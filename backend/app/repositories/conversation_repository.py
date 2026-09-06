"""Persistence for chat conversations + per-message stored embeddings.

The message embeddings are written by the background worker (see
app/ingestion/worker.py) so the retriever can run history similarity search
without embedding on the request hot path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.models.conversation import ChatConversation, ChatMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


class ConversationRepository:

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    def create_conversation(
        self,
        chat_id: str,
        user_id: str,
        title: str | None = None,
    ) -> ChatConversation:
        row = ChatConversation(
            chat_id=chat_id,
            user_id=user_id,
            title=title,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_conversation(
        self,
        chat_id: str,
        user_id: str,
    ) -> ChatConversation | None:
        stmt = select(ChatConversation).where(
            ChatConversation.chat_id == chat_id,
            ChatConversation.user_id == user_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[ChatConversation]:
        stmt = (
            select(ChatConversation)
            .where(ChatConversation.user_id == user_id)
            .order_by(desc(ChatConversation.updated_at))
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def update_conversation_time(self, chat_id: str) -> None:
        row = self.db.execute(
            select(ChatConversation).where(ChatConversation.chat_id == chat_id)
        ).scalar_one_or_none()
        if row is not None:
            row.updated_at = _utcnow()
            self.db.commit()

    def delete_conversation(self, chat_id: str, user_id: str) -> bool:
        row = self.get_conversation(chat_id, user_id)
        if row is None:
            return False
        self.db.execute(delete(ChatMessage).where(ChatMessage.chat_id == chat_id))
        self.db.delete(row)
        self.db.commit()
        return True

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def add_message(
        self,
        message_id: str,
        chat_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        citations: Any = None,
        metadata: Any = None,
    ) -> ChatMessage:
        row = ChatMessage(
            message_id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            embedding_json=_dump(embedding),
            citations_json=_dump(citations),
            metadata_json=_dump(metadata),
            created_at=_utcnow(),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        self.update_conversation_time(chat_id)
        return row

    def get_messages(
        self,
        chat_id: str,
        limit: int = 40,
    ) -> list[ChatMessage]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def count_messages(self, chat_id: str) -> int:
        return (
            self.db.scalar(
                select(func.count())
                .select_from(ChatMessage)
                .where(ChatMessage.chat_id == chat_id)
            )
            or 0
        )

    def get_message(self, message_id: str) -> ChatMessage | None:
        stmt = select(ChatMessage).where(ChatMessage.message_id == message_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def update_embedding(self, message_id: str, embedding: list[float]) -> bool:
        row = self.get_message(message_id)
        if row is None:
            return False
        row.embedding_json = _dump(embedding)
        self.db.commit()
        return True
