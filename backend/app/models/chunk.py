from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
    JSON,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.models.document import Base


class DocumentChunk(Base):

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    chunk_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    doc_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    chunk_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    section_path: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    locations: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    metadata: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )