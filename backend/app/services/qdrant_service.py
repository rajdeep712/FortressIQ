from qdrant_client import (
    QdrantClient,
    models,
)

from app.ingestion.payload import (
    build_payload,
)


class QdrantService:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
    ):

        self.collection_name = collection_name

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
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
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=build_payload(chunk),
                )
            )

        return self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )