from sqlalchemy import select  
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    def __init__(self, db: Session):  ## Connects to the database using SQLAlchemy.
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

    def get_by_id(
        self,
        doc_id: str,
    ) -> Document | None:

        statement = select(Document).where(
            Document.doc_id == doc_id
        )

        return self.db.execute(
            statement
        ).scalar_one_or_none()

    def update_status(
        self,
        doc_id: str,
        status: str,
    ) -> Document | None:

        document = self.get_by_id(doc_id)

        if document is None:
            return None

        document.status = status
        self.db.commit()
        self.db.refresh(document)

        return document

    def create(
        self,
        document: Document,
    ) -> Document:

        self.db.add(document)   ## This tells "I want to insert this document."
        self.db.commit()   ## This actually commits the transaction to the database.
        self.db.refresh(document)  ## This reloads the document from the database.

        return document