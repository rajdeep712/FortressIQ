from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api.deps import get_current_user, require_verified_user
from app.core.config import settings
from app.core.database import get_db
from app.core.tracing import maybe_span, turn_context
from app.models.user import User
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentRetryResponse,
)
from app.services.document_service import DocumentService
from app.services.malware_scanner import MalwareScanner
from app.services.s3_service import S3Service

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


@router.post(   ## Final URL with the prefix becomes -> POST /api/v1/documents/upload
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_verified_user),
):
    user_id = current_user.user_id

    repository = DocumentRepository(db)
    malware_scanner = MalwareScanner()

    document_service = DocumentService(
        repository=repository,
        s3_service=S3Service(),
        malware_scanner=malware_scanner,
    )

    try:
        with turn_context(chat_id="", user_id=user_id):
            with maybe_span(
                "upload_document",
                kind="CHAIN",
                user_id=user_id,
                filename=file.filename or "",
                content_type=file.content_type or "",
            ):
                document = await document_service.upload_document(
                    upload_file=file,
                    user_id=user_id,
                )

            return DocumentUploadResponse(
                doc_id=document.doc_id,
                user_id=document.user_id,
                filename=document.original_filename,
                mime_type=document.mime_type,
                size=document.file_size,
                sha256=document.sha256,
                s3_key=document.s3_key,
                status=document.status,
                created_at=document.created_at,
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="This document has already been uploaded.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document upload failed.",
        )


async def _enqueue_ingestion(
    doc_id: str,
    user_id: str,
) -> None:
    from arq.connections import (
        RedisSettings,
        create_pool,
    )

    if not settings.redis_url:
        raise HTTPException(
            status_code=503,
            detail="Redis is not configured.",
        )

    redis = await create_pool(
        settings_=RedisSettings.from_dsn(
            settings.redis_url
        )
    )

    try:
        await redis.enqueue_job(
            "ingest_document",
            doc_id,
            user_id,
        )
    finally:
        await redis.aclose()


@router.get(
    "/{doc_id}",
    response_model=DocumentStatusResponse,
)
async def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = DocumentRepository(db).get_by_id(doc_id)

    if document is None or document.user_id != current_user.user_id:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentStatusResponse(
        doc_id=document.doc_id,
        filename=document.original_filename,
        status=document.status,
        chunk_count=ChunkRepository(db).count_by_doc(
            doc_id
        ),
        s3_key=document.s3_key,
        created_at=document.created_at,
    )


@router.post(
    "/{doc_id}/retry",
    response_model=DocumentRetryResponse,
)
async def retry_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repository = DocumentRepository(db)
    document = repository.get_by_id(doc_id)

    if document is None or document.user_id != current_user.user_id:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if document.status == "PROCESSING":
        raise HTTPException(
            status_code=409,
            detail="Document is already being processed.",
        )

    repository.update_status(doc_id, "UPLOADED")

    await _enqueue_ingestion(
        doc_id,
        document.user_id,
    )

    return DocumentRetryResponse(
        doc_id=doc_id,
        status="UPLOADED",
        message=(
            "Ingestion job enqueued. Track progress via "
            "GET /api/v1/documents/{doc_id}."
        ),
    )
