import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.ingestion.chunker import chunk_document, get_chunker_for
from app.ingestion.embedding import (
    EmbeddingService,
    SparseEmbeddingService,
)
from app.ingestion.parsers import get_parser
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.core.tracing import maybe_span, set_span_attributes
from app.services.qdrant_service import QdrantService
from app.services.s3_service import S3Service

logger = logging.getLogger(__name__)


class IngestionService:

    def __init__(self):
        self.s3 = S3Service()
        self.embedding = EmbeddingService()
        self.sparse_embedding = SparseEmbeddingService()
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

            parser = get_parser(document.extension)
            logger.info(
                "Ingesting document=%s user=%s file=%s parser=%s "
                "embed_provider=%s dimension=%d",
                doc_id,
                user_id,
                document.stored_filename,
                getattr(parser, "name", ""),
                getattr(self.embedding, "last_provider", "") or "",
                getattr(self.embedding, "dimension", 0) or 0,
            )

            # --------------------------------------------------
            # Download + parse
            # --------------------------------------------------

            t0 = time.monotonic()

            with maybe_span(
                "ingestion.download",
                kind="TOOL",
                doc_id=doc_id,
                s3_key=document.s3_key,
            ):
                content = self.s3.get_object(document.s3_key)

            fd, temp_name = tempfile.mkstemp(
                suffix=document.extension
            )
            os.close(fd)
            temp_path = Path(temp_name)

            temp_path.write_bytes(content)

            with maybe_span(
                "ingestion.parse",
                kind="TOOL",
                doc_id=doc_id,
                parser=getattr(parser, "name", ""),
                parser_version=getattr(parser, "version", ""),
                mime_type=document.mime_type,
                file_size_bytes=len(content),
            ):
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

            logger.info(
                "Parsed document=%s (%d bytes) with %s in %.1fs",
                doc_id,
                len(content),
                getattr(parser, "name", ""),
                time.monotonic() - t0,
            )

            # --------------------------------------------------
            # Chunk + embed children
            # --------------------------------------------------

            t0 = time.monotonic()

            with maybe_span(
                "ingestion.chunk",
                kind="CHAIN",
                doc_id=doc_id,
                chunker=getattr(
                    get_chunker_for(parsed),
                    "identifier",
                    "",
                ),
            ):
                chunks = chunk_document(parsed)

                child_chunks = [
                    chunk
                    for chunk in chunks
                    if chunk.chunk_type == "child"
                ]

                set_span_attributes(
                    total_chunks=len(chunks),
                    parent_count=len(chunks) - len(child_chunks),
                    child_count=len(child_chunks),
                )

            logger.info(
                "Chunked document=%s into %d chunks "
                "(%d parent, %d child) in %.1fs",
                doc_id,
                len(chunks),
                len(chunks) - len(child_chunks),
                len(child_chunks),
                time.monotonic() - t0,
            )

            with maybe_span(
                "ingestion.embed.dense",
                kind="EMBEDDING",
                doc_id=doc_id,
                provider=getattr(self.embedding, "last_provider", "") or "",
                model=getattr(self.embedding, "last_model", "") or "",
                text_count=len(child_chunks),
                dimension=getattr(self.embedding, "dimension", 0) or 0,
            ):
                t0 = time.monotonic()
                vectors = self.embedding.embed(
                    [chunk.text for chunk in child_chunks]
                )

            logger.info(
                "Dense-embedded %d child chunks with %s/%s "
                "(dim=%d) in %.1fs",
                len(child_chunks),
                self.embedding.last_provider or "",
                getattr(self.embedding, "last_model", "") or "",
                len(vectors[0]) if vectors else 0,
                time.monotonic() - t0,
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

            # Compute BM25 sparse vectors for every child chunk.
            # This is strict: if the sparse model is unavailable or the
            # produced vectors don't match the child chunks, ingest fails
            # rather than silently storing dense-only points (which would
            # be invisible to hybrid search's sparse arm).
            with maybe_span(
                "ingestion.embed.sparse",
                kind="EMBEDDING",
                doc_id=doc_id,
                model=getattr(
                    self.sparse_embedding,
                    "model_name",
                    "",
                ) or "",
                text_count=len(child_chunks),
            ):
                t0 = time.monotonic()
                sparse_vectors = self.sparse_embedding.embed(
                    [chunk.text for chunk in child_chunks]
                )

            logger.info(
                "Sparse-embedded %d child chunks with %s in %.1fs",
                len(child_chunks),
                getattr(self.sparse_embedding, "model_name", "") or "",
                time.monotonic() - t0,
            )

            if len(sparse_vectors) != len(child_chunks):
                raise RuntimeError(
                    "Sparse embedding produced "
                    f"{len(sparse_vectors)} vectors for "
                    f"{len(child_chunks)} child chunks; aborting ingest"
                )

            # --------------------------------------------------
            # Persist metadata + store vectors in Qdrant
            # --------------------------------------------------

            with maybe_span(
                "ingestion.persist_db",
                kind="CHAIN",
                doc_id=doc_id,
                chunk_count=len(chunks),
                parent_count=len(chunks) - len(child_chunks),
                child_count=len(child_chunks),
            ):
                chunk_repository.create_many(chunks)

            with maybe_span(
                "ingestion.upsert_qdrant",
                kind="RETRIEVER",
                doc_id=doc_id,
                collection=getattr(self.qdrant, "collection_name", ""),
                point_count=len(child_chunks),
                batch_size=getattr(self.qdrant, "upsert_batch_size", 0) or 0,
            ):
                t0 = time.monotonic()
                self.qdrant.upsert_chunks(
                    chunks=child_chunks,
                    vectors=vectors,
                    sparse_vectors=sparse_vectors,
                )

            logger.info(
                "Upserted %d points to Qdrant collection %s in %.1fs",
                len(child_chunks),
                getattr(self.qdrant, "collection_name", "") or "",
                time.monotonic() - t0,
            )

            document_repository.update_status(
                doc_id,
                "COMPLETED",
            )

            logger.info(
                "Document %s ingested: "
                "%d chunks, %d embedded "
                "(embedder=%s model=%s)",
                doc_id,
                len(chunks),
                len(child_chunks),
                self.embedding.last_provider or "",
                getattr(self.embedding, "last_model", "") or "",
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