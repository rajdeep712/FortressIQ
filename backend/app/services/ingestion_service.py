import asyncio
import logging
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.ingestion.chunker import chunk_document
from app.ingestion.embedding import EmbeddingService
from app.ingestion.parsers import get_parser
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.qdrant_service import QdrantService
from app.services.s3_service import S3Service

logger = logging.getLogger(__name__)


class IngestionService:

    def __init__(self):
        self.s3 = S3Service()
        self.embedding = EmbeddingService()
        self.qdrant = QdrantService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            vector_size=settings.embedding_dimension,
            collection_name=settings.qdrant_collection_name,
            timeout_seconds=settings.qdrant_timeout_seconds,
            upsert_batch_size=settings.qdrant_upsert_batch_size,
        )

    def ingest_document(
        self,
        doc_id: str,
        user_id: str,
    ) -> dict:

        db: Session | None = None
        document_repository: DocumentRepository | None = None
        temp_path: Path | None = None

        try:
            db = SessionLocal()
            document_repository = DocumentRepository(db)
            chunk_repository = ChunkRepository(db)

            document = document_repository.get_by_id(doc_id)

            if document is None:
                raise ValueError(
                    f"Document {doc_id} not found."
                )

            if document.user_id != user_id:
                raise ValueError(
                    f"Document {doc_id} does not belong "
                    f"to user {user_id}."
                )

            if document.status == "COMPLETED":
                logger.info(
                    "Document %s already ingested; skipping",
                    doc_id,
                )
                return {
                    "doc_id": doc_id,
                    "skipped": True,
                    "chunks": chunk_repository.count_by_doc(
                        doc_id
                    ),
                }

            version_id = f"v_{uuid4().hex}"

            # --------------------------------------------------
            # Replace semantics: remove any prior artifacts for
            # this document so retries never duplicate data.
            # --------------------------------------------------

            chunk_repository.delete_by_doc(doc_id)
            self.qdrant.delete_chunks_by_doc(doc_id)

            document_repository.update_status(
                doc_id,
                "PROCESSING",
            )

            # --------------------------------------------------
            # Download + parse
            # --------------------------------------------------

            parser = get_parser(document.extension)
            content = self.s3.get_object(document.s3_key)

            fd, temp_name = tempfile.mkstemp(
                suffix=document.extension
            )
            os.close(fd)
            temp_path = Path(temp_name)

            temp_path.write_bytes(content)

            parsed = asyncio.run(
                parser.parse(
                    file_path=temp_path,
                    doc_id=doc_id,
                    version_id=version_id,
                    user_id=user_id,
                    filename=document.stored_filename,
                    mime_type=document.mime_type,
                )
            )

            # --------------------------------------------------
            # Chunk + embed children
            # --------------------------------------------------

            chunks = chunk_document(parsed)

            child_chunks = [
                chunk
                for chunk in chunks
                if chunk.chunk_type == "child"
            ]

            vectors = self.embedding.embed(
                [chunk.text for chunk in child_chunks]
            )

            # Record which embedding provider produced the
            # vectors so search can match query-time embeddings
            # with the same model.
            if (
                vectors
                and self.embedding.last_provider is not None
            ):
                for chunk in child_chunks:
                    chunk.metadata["embedder"] = (
                        self.embedding.last_provider
                    )

            # --------------------------------------------------
            # Persist metadata + store vectors in Qdrant
            # --------------------------------------------------

            chunk_repository.create_many(chunks)

            self.qdrant.upsert_chunks(
                chunks=child_chunks,
                vectors=vectors,
            )

            document_repository.update_status(
                doc_id,
                "COMPLETED",
            )

            logger.info(
                "Document %s ingested: "
                "%d chunks, %d embedded",
                doc_id,
                len(chunks),
                len(child_chunks),
            )

            return {
                "doc_id": doc_id,
                "version_id": version_id,
                "skipped": False,
                "chunks": len(chunks),
                "embedded_chunks": len(child_chunks),
            }

        except Exception:
            if document_repository is not None:
                try:
                    document_repository.update_status(
                        doc_id,
                        "FAILED",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Could not mark doc_id=%s as FAILED",
                        doc_id,
                    )
            logger.exception(
                "Document ingestion failed for doc_id=%s",
                doc_id,
            )
            raise

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if db is not None:
                db.close()