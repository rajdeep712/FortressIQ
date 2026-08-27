from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.database import get_db
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentUploadResponse
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
    db: Session = Depends(get_db),   ## using Depends automatically calls get_db and injects it here.(which creates the session and provides it to the function).
):
    user_id = settings.mock_user_id   ## For now we are using mock user id.

    repository = DocumentRepository(db)
    malware_scanner = MalwareScanner()

    document_service = DocumentService(
        repository=repository,
        s3_service=S3Service(),
        malware_scanner=malware_scanner,
    )

    try:
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
