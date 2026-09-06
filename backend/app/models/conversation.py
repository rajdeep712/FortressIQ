from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
    Index,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.models.document import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    chat_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_chat_conversations_user_updated", "user_id", "updated_at"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    message_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    chat_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        # user | assistant | system
        String(30),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Precomputed dense embedding for history similarity search. Persisted by
    # the background worker so query-time search never embeds on the hot path.
    embedding_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # JSON-serialized citations for assistant messages.
    citations_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Arbitrary per-message metadata (e.g. latency, token counts).
    metadata_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_chat_messages_chat_created", "chat_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<ChatMessage {self.message_id} role={self.role} "
            f"chat_id={self.chat_id}>"
        )
