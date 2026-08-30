from qdrant_client import (
    QdrantClient,
    models,
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

        if self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        self._create_payload_indexes()

    def _create_payload_indexes(self):

        indexes = [
            (
                "user_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "doc_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "version_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "chunk_type",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "parser",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "mime_type",
                models.PayloadSchemaType.KEYWORD,
            ),
        ]

        for field_name, schema in indexes:

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    def upsert_chunks(
        self,
        chunks,
        vectors,
    ):

        points = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):

            payload = {
                # Authorization / tenant filtering
                "user_id": chunk.user_id,
                "doc_id": chunk.doc_id,
                "version_id": chunk.version_id,

                # Retrieval organization
                "chunk_id": chunk.chunk_id,
                "parent_chunk_id": (
                    chunk.parent_chunk_id
                ),
                "chunk_type": chunk.chunk_type,

                # Document information
                "filename": chunk.metadata.get(
                    "filename"
                ),
                "parser": chunk.metadata.get(
                    "parser"
                ),

                # Structure
                "section_path": chunk.section_path,

                # Provenance
                "locations": [
                    location.__dict__
                    for location in chunk.locations
                ],

                # Extra metadata
                "metadata": chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )