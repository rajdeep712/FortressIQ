from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:

    def __init__(self, db: Session):
        self.db = db

    def find_by_hash(
        self,
        user_id: str,
        sha256: str,
    ) -> Document | None:

        statement = select(Document).where(
            Document.user_id == user_id,
            Document.sha256 == sha256,
        )

        return self.db.execute(
            statement
        ).scalar_one_or_none()

    def create(
        self,
        document: Document,
    ) -> Document:

        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        return document