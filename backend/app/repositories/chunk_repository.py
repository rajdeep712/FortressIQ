from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.chunk import DocumentChunk
from app.ingestion.chunker import Chunk


class ChunkRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def create_many(
        self,
        chunks: list[Chunk],
    ):

        rows = []

        for chunk in chunks:

            rows.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    parent_chunk_id=(
                        chunk.parent_chunk_id
                    ),
                    doc_id=chunk.doc_id,
                    version_id=chunk.version_id,
                    user_id=chunk.user_id,
                    chunk_type=chunk.chunk_type,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    section_path=chunk.section_path,
                    locations=[
                        location.__dict__
                        for location
                        in chunk.locations
                    ],
                    metadata=chunk.metadata,
                )
            )

        self.db.add_all(rows)
        self.db.commit()

        return rows

    
    def get_chunk(
        self,
        chunk_id: str,
    ):
        statement = select(
            DocumentChunk
        ).where(
            DocumentChunk.chunk_id
            == chunk_id
        )

        return self.db.execute(
            statement
        ).scalar_one_or_none()


def get_parent(
        self,
        parent_chunk_id: str,
    ):
        return self.get_chunk(
            parent_chunk_id
        )