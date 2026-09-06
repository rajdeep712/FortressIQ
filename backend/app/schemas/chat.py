from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class CitationOut(BaseModel):
    """A citation that maps an inline marker to a highlighted document region.

    The fields align with the frontend's ``Citation`` type so the UI can
    open the exact highlighted portion in the right-side document viewer.
    """

    id: int
    doc_id: str
    doc_title: str = ""
    page: int = 1
    snippet: str = ""
    chunk_title: str = ""
    confidence: float = 0.0
    score: float = 0.0
    bounding_box: dict[str, Any] = {}  # {top,left,width,height} percentages

    model_config = ConfigDict(extra="allow")


class ChatMessageOut(BaseModel):
    message_id: str
    chat_id: str
    role: str
    content: str
    citations: list[CitationOut] = []
    created_at: datetime

    model_config = ConfigDict(extra="allow")


class ConversationOut(BaseModel):
    chat_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="allow")


class ChatRequest(BaseModel):
    chat_id: str | None = None
    message: str
    selected_doc_ids: list[str] | None = None

    model_config = ConfigDict(extra="ignore")
