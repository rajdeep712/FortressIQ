from datetime import datetime

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    doc_id: str
    user_id: str
    filename: str
    mime_type: str
    size: int
    sha256: str
    s3_key: str
    status: str
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    s3_key: str
    created_at: datetime


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    file_size: int
    mime_type: str
    created_at: datetime


class DocumentRetryResponse(BaseModel):
    doc_id: str
    status: str
    message: str
