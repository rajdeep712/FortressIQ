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
