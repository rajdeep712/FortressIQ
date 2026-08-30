from pathlib import Path

from app.ingestion.chunker import (
    HierarchicalChunker,
)
from app.ingestion.embedding import (
    EmbeddingService,
)
from app.repositories.chunk_repository import (
    ChunkRepository,
)
from app.services.qdrant_service import (
    QdrantService,
)


class IngestionPipeline:

    def __init__(
        self,
        chunker: HierarchicalChunker,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkRepository,
        qdrant_service: QdrantService,
    ):

        self.chunker = chunker

        self.embedding = (
            embedding_service
        )

        self.chunk_repository = (
            chunk_repository
        )

        self.qdrant = qdrant_service

    async def process(
        self,
        parser,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ):

        # -----------------------------------------
        # 1. Parse
        # -----------------------------------------

        document = await parser.parse(
            file_path=file_path,
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
        )

        # -----------------------------------------
        # 2. Hierarchical chunking
        # -----------------------------------------

        chunks = self.chunker.chunk(
            document
        )

        # -----------------------------------------
        # 3. Store chunk metadata
        # -----------------------------------------

        self.chunk_repository.create_many(
            chunks
        )

        # -----------------------------------------
        # 4. Embed CHILD chunks only
        # -----------------------------------------

        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
        ]

        vectors = self.embedding.embed(
            [
                chunk.text
                for chunk in child_chunks
            ]
        )

        # -----------------------------------------
        # 5. Store in Qdrant
        # -----------------------------------------

        self.qdrant.upsert_chunks(
            chunks=child_chunks,
            vectors=vectors,
        )

        return {
            "doc_id": doc_id,
            "version_id": version_id,
            "chunks": len(chunks),
            "embedded_chunks": len(
                child_chunks
            ),
        }