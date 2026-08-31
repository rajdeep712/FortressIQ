import logging
import re
import time
import uuid

import httpx
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

        if not self.client.collection_exists(
            self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        self._create_payload_indexes()

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
    ):

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunk/vector count mismatch: "
                f"{len(chunks)} chunks vs {len(vectors)} vectors"
            )

        points = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):

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