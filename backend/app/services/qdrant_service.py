import logging
import re
import time
import uuid

import httpx
from typing import Any
from qdrant_client import (
    QdrantClient,
    models,
)

from app.ingestion.payload import (
    build_payload,
)

logger = logging.getLogger(__name__)

# Chunk ids look like `child_<32 hex>` / `parent_<32 hex>`
# (app/ingestion/chunker/base.py). Qdrant does not accept those as point
# ids: it only accepts unsigned integers or UUIDs. Derive a dashed UUID
# from the chunk id deterministically so re-upserts/re-ingests keep the
# exact same point id for a given chunk.
_HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")


def qdrant_point_id(chunk_id: str) -> str:
    suffix = chunk_id.rsplit("_", 1)[-1]
    if _HEX32.match(suffix):
        return str(uuid.UUID(suffix))
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


# Retryable client/network conditions (e.g. Qdrant Cloud read timeouts).
# Everything else -- validation 4xx, unexpected responses -- is permanent
# and must fail the batch immediately.
TRANSIENT_UPSERT_ERRORS = (
    httpx.TransportError,
    httpx.TimeoutException,
    ConnectionError,
)


class QdrantService:
    # The named sparse vector (BM25) used for hybrid dense+sparse search.
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        upsert_batch_size: int = 100,
        upsert_max_retries: int = 2,
        upsert_retry_delay_seconds: float = 1.0,
    ):

        self.collection_name = collection_name
        self.upsert_batch_size = upsert_batch_size
        self.upsert_max_retries = upsert_max_retries
        self.upsert_retry_delay_seconds = upsert_retry_delay_seconds

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

        self._ensure_collection(
            vector_size
        )

    def _ensure_collection(
        self,
        vector_size: int,
    ):

        if self.client.collection_exists(
            self.collection_name
        ):
            # If the existing collection was created with a different
            # vector dimension (e.g. 2048 before switching to BGE 768),
            # delete it and recreate — Qdrant collection vector size is
            # fixed at creation and cannot be changed.
            try:
                info = self.client.get_collection(
                    self.collection_name
                )
                existing_size = getattr(
                    info.config.params.vectors,
                    "size",
                    None,
                )
                if (
                    existing_size is not None
                    and existing_size != vector_size
                ):
                    logger.warning(
                        "Collection %s has vector size %d "
                        "but config requests %d — "
                        "deleting and recreating.",
                        self.collection_name,
                        existing_size,
                        vector_size,
                    )
                    self.client.delete_collection(
                        self.collection_name
                    )
            except Exception:  # noqa: BLE001
                pass

        if not self.client.collection_exists(
            self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: (
                        models.SparseVectorParams()
                    )
                },
            )
        else:
            self._ensure_sparse_config()

        self._create_payload_indexes()

    def _ensure_sparse_config(self):
        """Idempotently add the named sparse vector config to an existing
        (previously dense-only) collection so hybrid search can use it.
        Additive and safe to run on every init; does nothing if present."""
        try:
            info = self.client.get_collection(
                self.collection_name
            )
            sparse = getattr(
                info.config.params,
                "sparse_vectors",
                None,
            ) or {}
        except Exception:  # noqa: BLE001
            return

        if self.SPARSE_VECTOR_NAME in sparse:
            return

        # Qdrant cannot add a brand-new named vector via update_collection
        # (PATCH); that endpoint only alters existing vector configs.
        # Adding a sparse vector post-creation must go through the dedicated
        # named-vector endpoint (Qdrant >= 1.18), which the sdk exposes as
        # create_vector_name.
        self.client.create_vector_name(
            collection_name=self.collection_name,
            vector_name=self.SPARSE_VECTOR_NAME,
            vector_name_config=models.SparseVectorNameConfig(
                sparse=models.SparseVectorConfig(),
            ),
        )

    # Payload fields the app filters on, with their Qdrant index schema.
    # Qdrant indexes list-valued fields (pages, slides, section_path,
    # json_paths, ...) with the same plain schema, so only the scalar or
    # full-text case needs special params.
    PAYLOAD_INDEXES = (
        # Identity / tenant filtering
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("version_id", models.PayloadSchemaType.KEYWORD),
        ("parent_chunk_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        # Provenance
        ("filename", models.PayloadSchemaType.KEYWORD),
        ("parser", models.PayloadSchemaType.KEYWORD),
        ("mime_type", models.PayloadSchemaType.KEYWORD),
        ("strategy", models.PayloadSchemaType.KEYWORD),
        ("embedder", models.PayloadSchemaType.KEYWORD),
        ("section_path", models.PayloadSchemaType.KEYWORD),
        # Document structure / location
        ("root", models.PayloadSchemaType.KEYWORD),
        ("sheet", models.PayloadSchemaType.KEYWORD),
        ("sheets", models.PayloadSchemaType.KEYWORD),
        ("pages", models.PayloadSchemaType.INTEGER),
        ("slides", models.PayloadSchemaType.INTEGER),
        ("row_start", models.PayloadSchemaType.INTEGER),
        ("row_end", models.PayloadSchemaType.INTEGER),
        ("line_start", models.PayloadSchemaType.INTEGER),
        ("line_end", models.PayloadSchemaType.INTEGER),
        ("json_paths", models.PayloadSchemaType.KEYWORD),
        ("dom_paths", models.PayloadSchemaType.KEYWORD),
        ("cell_ranges", models.PayloadSchemaType.KEYWORD),
    )

    # Full-text keyword search over chunk text (hybrid retrieval).
    TEXT_INDEX_PARAMS = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
        min_token_len=2,
    )

    def _create_payload_indexes(self):
        # Additive and idempotent across re-inits: only create indexes
        # the collection does not already have.
        existing = set(
            self.client.get_collection(
                self.collection_name
            ).payload_schema.keys()
        )

        for field_name, schema in self.PAYLOAD_INDEXES:
            if field_name in existing:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:  # noqa: BLE001
                # A concurrent writer may have raced us to create it;
                # only swallow that case, surface anything else.
                payload_schema = self.client.get_collection(
                    self.collection_name
                ).payload_schema
                if field_name not in payload_schema:
                    raise

        if "text" not in existing:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="text",
                    field_schema=self.TEXT_INDEX_PARAMS,
                    wait=True,
                )
            except Exception:  # noqa: BLE001
                payload_schema = self.client.get_collection(
                    self.collection_name
                ).payload_schema
                if "text" not in payload_schema:
                    raise

    def delete_chunks_by_doc(
        self,
        doc_id: str,
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(
                                value=doc_id
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def upsert_chunks(
        self,
        chunks,
        vectors,
        sparse_vectors: list[dict] | None = None,
    ):

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunk/vector count mismatch: "
                f"{len(chunks)} chunks vs {len(vectors)} vectors"
            )

        if (
            sparse_vectors is not None
            and len(sparse_vectors) != len(chunks)
        ):
            raise ValueError(
                "chunk/sparse-vector count mismatch: "
                f"{len(chunks)} chunks vs {len(sparse_vectors)} "
                "sparse vectors"
            )

        points = []

        for i, chunk in enumerate(chunks):

            if sparse_vectors is None:
                vector = vectors[i]
            else:
                vector = {
                    # `""` is the reserved name for the default (unnamed)
                    # dense vector in a multi-vector point.
                    "": vectors[i],
                    self.SPARSE_VECTOR_NAME: (
                        sparse_vectors[i]
                    ),
                }

            points.append(
                models.PointStruct(
                    id=qdrant_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=build_payload(chunk),
                )
            )

        # Qdrant Cloud is sensitive to single large upsert requests: a
        # whole document in one wait=True call can exceed the HTTP read
        # timeout. Send points in bounded batches so each request stays
        # small, and retry a batch before giving up (upserts are
        # idempotent per point id, so retrying a batch is safe).
        results = []
        for start in range(
            0,
            len(points),
            self.upsert_batch_size,
        ):
            batch = points[
                start:start + self.upsert_batch_size
            ]
            for attempt in range(self.upsert_max_retries + 1):
                try:
                    result = self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True,
                    )
                    results.append(result)
                    break
                except TRANSIENT_UPSERT_ERRORS:
                    if (
                        attempt
                        >= self.upsert_max_retries
                    ):
                        raise
                    delay = min(
                        self.upsert_retry_delay_seconds
                        * (2**attempt),
                        10,
                    )
                    logger.warning(
                        "Qdrant upsert of %d points failed "
                        "(attempt %d); retrying in %.1fs",
                        len(batch),
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)

        return results

    # ------------------------------------------------------------------
    # Tenant-safe metadata filter
    # ------------------------------------------------------------------

    @staticmethod
    def tenant_filter(
        user_id: str,
        doc_ids: list[str] | None = None,
        chunk_type: str | None = "child",
    ) -> models.Filter:
        """Build the metadata Filter used for authorization + selection.

        Always ties results to ``user_id`` (sourced from the authenticated
        request, never client-supplied). Optionally restricts to a subset of
        documents and/or a chunk type (default: child/leaf chunks).
        """
        must = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]

        if doc_ids:
            must.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=list(doc_ids)),
                )
            )

        if chunk_type:
            must.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type),
                )
            )

        return models.Filter(must=must)

    @staticmethod
    def _sparse_query(sparse_vector: dict) -> models.SparseVector:
        """Build the named sparse vector query for a single modality prefix."""
        return models.SparseVector(
            indices=sparse_vector.get("indices", []),
            values=sparse_vector.get("values", []),
        )

    # ------------------------------------------------------------------
    # Hybrid dense + sparse (BM25) search, fused with RRF/DBSF
    # ------------------------------------------------------------------

    @staticmethod
    def _fusion_query(fusion: str | None) -> Any:
        """Build the Qdrant fusion query object for a modality prefix.

        Newer qdrant-client versions (>= 1.19) require wrapping the fusion
        enum in a ``FusionQuery``; older ones accept ``Fusion.RRF`` directly.
        """
        name = (fusion or "rrf").upper()
        choice = getattr(models.Fusion, name, models.Fusion.RRF)
        if "fusion" in getattr(models.FusionQuery, "model_fields", {}):
            return models.FusionQuery(fusion=choice)
        return choice

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict,
        *,
        user_id: str,
        doc_ids: list[str] | None = None,
        chunk_type: str | None = "child",
        top_k: int = 5,
        prefetch_dense: int = 20,
        prefetch_sparse: int = 20,
        fusion: str = "rrf",
    ) -> list[dict]:
        """Run dense + sparse hybrid retrieval over the tenant-filtered set.

        Both ``query_dense`` (list of floats) and ``query_sparse``
        ({"indices", "values"}) must be precomputed by the caller; this keeps
        the Qdrant layer independent of embedding providers.

        Returns the fused, top-``top_k`` scored points as plain dicts
        (payload + score), sorted by descending fused score.
        """
        qfilter = self.tenant_filter(
            user_id,
            doc_ids=doc_ids,
            chunk_type=chunk_type,
        )

        prefetch = [
            models.Prefetch(
                query=query_dense,
                using="",
                limit=prefetch_dense,
                filter=qfilter,
            ),
            models.Prefetch(
                query=self._sparse_query(query_sparse),
                using=self.SPARSE_VECTOR_NAME,
                limit=prefetch_sparse,
                filter=qfilter,
            ),
        ]

        fusion_query = self._fusion_query(fusion)

        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=fusion_query,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "payload": (point.payload or {}),
            }
            for point in result.points
        ]