import logging
from uuid import uuid4
import tempfile
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.file_validator import (
    get_extension,
    is_supported_extension,
    check_file_signature,
    detect_mime_type,
    check_mime_type,
    validate_office_package,
    check_zip_limits,
)
from app.services.hash_service import calculate_sha256
from app.core.tracing import maybe_span
from app.services.malware_scanner import MalwareScanner
from app.services.s3_service import S3Service
from app.utils.filenames import sanitize_filename

logger = logging.getLogger(__name__)


class DocumentService:

    def __init__(
        self,
        repository: DocumentRepository,
        s3_service: S3Service,
        malware_scanner: MalwareScanner,
    ):
        self.repository = repository   ## Database
        self.s3 = s3_service     ## S3
        self.malware_scanner = malware_scanner

    async def upload_document(
        self,
        upload_file: UploadFile,
        user_id: str,
    ):
        original_filename = upload_file.filename

        if not original_filename:
            raise ValueError("Filename is missing.")

        extension = get_extension(original_filename)

        # --------------------------------------------------
        # 1. Extension check
        # --------------------------------------------------

        if not is_supported_extension(original_filename):
            raise ValueError(f"Unsupported file type: {extension}")

        logger.info(
            "Upload started: filename=%s user_id=%s",
            original_filename,
            user_id,
        )

        # --------------------------------------------------
        # 2. Temporary file
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp:
            temp_path = Path(temp.name)
            total_size = 0

            while chunk := await upload_file.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > settings.max_file_size_mb * 1024 * 1024:
                    temp_path.unlink(missing_ok=True)
                    raise ValueError(
                        "File exceeds maximum allowed size."
                    )

                temp.write(chunk)

        try:

            # --------------------------------------------------
            # 3. Signature check
            # --------------------------------------------------

            if not check_file_signature(temp_path, extension):
                raise ValueError(
                    "File content does not match its extension."
                )

            # --------------------------------------------------
            # 4. MIME detection (single call)
            # --------------------------------------------------

            mime_type = detect_mime_type(temp_path)

            if not check_mime_type(mime_type, extension):
                raise ValueError(
                    f"Detected MIME type '{mime_type}' "
                    f"does not match extension '{extension}'."
                )

            # --------------------------------------------------
            # 5. Office package validation
            # --------------------------------------------------

            if not validate_office_package(temp_path, extension):
                raise ValueError("Invalid Office document.")

            # --------------------------------------------------
            # 6. ZIP limits
            # --------------------------------------------------

            if extension in {".docx", ".pptx", ".xlsx"}:
                if not check_zip_limits(temp_path):
                    raise ValueError(
                        "Office package exceeds safety limits."
                    )

            # --------------------------------------------------
            # 7. Malware scan
            # --------------------------------------------------

            is_safe = self.malware_scanner.scan(temp_path)

            if not is_safe:
                raise ValueError("File failed malware scanning.")

            # --------------------------------------------------
            # 8. SHA-256
            # --------------------------------------------------

            sha256 = calculate_sha256(temp_path)

            # --------------------------------------------------
            # 9. Duplicate content check
            # --------------------------------------------------

            existing_document = self.repository.find_by_hash(
                user_id=user_id,
                sha256=sha256,
            )

            if existing_document:
                raise ValueError(
                    "This document has already been uploaded."
                )

            # --------------------------------------------------
            # 10. Generate doc_id
            # --------------------------------------------------

            doc_id = generate_doc_id()

            # --------------------------------------------------
            # 11. Sanitize filename
            # --------------------------------------------------

            safe_filename = sanitize_filename(original_filename)

            # --------------------------------------------------
            # 12. Generate S3 key
            # --------------------------------------------------

            s3_key = (
                f"{user_id}/"
                f"{doc_id}_"
                f"{safe_filename}"
            )

            # --------------------------------------------------
            # 13. Upload to S3
            # --------------------------------------------------

            with maybe_span(
                "upload_document.s3_upload",
                kind="TOOL",
                doc_id=doc_id,
                s3_key=s3_key,
                content_type=mime_type,
                file_size_bytes=total_size,
            ):
                try:
                    self.s3.upload_file(
                        file_path=temp_path,
                        s3_key=s3_key,
                        content_type=mime_type,
                    )
                except Exception:
                    logger.exception(
                        "S3 upload failed for doc_id=%s",
                        doc_id,
                    )
                    raise ValueError("Failed to store document.")

            logger.info(
                "S3 upload complete: doc_id=%s s3_key=%s",
                doc_id,
                s3_key,
            )

            # --------------------------------------------------
            # 14. Store metadata
            # --------------------------------------------------

            document = Document(
                doc_id=doc_id,
                user_id=user_id,
                original_filename=original_filename,
                stored_filename=safe_filename,
                extension=extension,
                mime_type=mime_type,
                file_size=total_size,
                sha256=sha256,
                s3_key=s3_key,
                status="UPLOADED",
            )

            try:
                with maybe_span(
                    "upload_document.persist_db",
                    kind="CHAIN",
                    doc_id=doc_id,
                ):
                    document = self.repository.create(document)
            except IntegrityError:
                # Race condition: another request uploaded the
                # same content simultaneously. Clean up S3 object.
                self.s3.client.delete_object(
                    Bucket=self.s3.bucket,
                    Key=s3_key,
                )
                raise ValueError(
                    "This document has already been uploaded."
                )

            logger.info(
                "Upload complete: doc_id=%s filename=%s",
                doc_id,
                original_filename,
            )

            return document

        finally:
            temp_path.unlink(missing_ok=True)


def generate_doc_id() -> str:
    return f"doc_{uuid4().hex}"
