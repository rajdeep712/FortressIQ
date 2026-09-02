"""Step 2: Conversation context search.

Given the current query embedding and the recent conversation turns, find the
most relevant prior messages. This is pure in-memory vector similarity over
the *conversation memory* -- it does NOT touch Qdrant (documents).

The conversation turns are supplied as an argument so the step is agnostic to
where they came from (currently the calling LangGraph node loads them from its
PostgreSQL persistence checkpointer; a unit test can pass synthetic turns).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

# A lightweight structural contract for an embedding provider. The real
# implementation is app.ingestion.embedding.EmbeddingService (which exposes
# `.embed(texts) -> list[list[float]]` and `.dimension`); tests inject a fake
# with the same shape, so this step stays decoupled from torch loading.
#
# This is duck-typed on purpose: importing EmbeddingService here would pull in
# sentence-transformers at import time for every consumer of the router.
HasEmbedder = Any


# Map langchain message `type` values onto our role vocabulary.
_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


class ConversationMessage(BaseModel):
    """A single conversation turn (Pydantic-view of a LangGraph message).

    Fields deliberately mirror what a LangGraph persisted message carries so a
    future node can map `BaseMessage` -> this model without loss.
    """

    message_id: str = ""
    role: str = "user"          # user | assistant | system
    content: str = ""
    timestamp: datetime | None = None
    embedding: list[float] | None = Field(default=None, repr=False)

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def build(cls, message: Any) -> "ConversationMessage":
        """Normalize a message from several sources without importing LangGraph.

        Accepts:
          * a ConversationMessage (returned unchanged)
          * a dict / mapping (keys: message_id|id, role|type, content, ...)
          * a langchain-style object exposing .id / .type / .content
        """
        if isinstance(message, ConversationMessage):
            return message

        if isinstance(message, Mapping):
            data = dict(message)
            role = data.get("role") or data.get("type") or "user"
            return cls(
                message_id=str(data.get("message_id") or data.get("id") or ""),
                role=str(role),
                content=str(data.get("content") or ""),
                timestamp=data.get("timestamp"),
                embedding=data.get("embedding"),
            )

        # langchain BaseMessage-like duck type
        role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
        role = _ROLE_ALIASES.get(str(role), str(role))
        content = getattr(message, "content", None) or ""
        if isinstance(content, list):  # content blocks in newer langchain
            content = " ".join(
                str(block.get("text"))
                for block in content
                if isinstance(block, Mapping) and block.get("text")
            )
        return cls(
            message_id=str(getattr(message, "id", None) or ""),
            role=str(role),
            content=str(content),
        )

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()


class ConversationHit(BaseModel):
    message: ConversationMessage
    score: float

    model_config = ConfigDict(extra="forbid")


class ConversationSearchResult(BaseModel):
    hits: list[ConversationHit] = Field(default_factory=list)
    best_score: float = 0.0
    mean_score: float = 0.0
    num_searched: int = 0
    num_message_turns: int = 0

    model_config = ConfigDict(extra="forbid")

    @property
    def conversation(self) -> list[ConversationHit]:
        """Alias used by the future context-fusion step."""
        return self.hits


def _embed(embedder: HasEmbedder, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return embedder.embed(texts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Robust cosine similarity (safe for zero vectors / mismatched dims)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search_conversation(
    messages: Iterable[Any],
    query: str | Sequence[float],
    embedder: HasEmbedder,
    *,
    window: int = 40,
    top_k: int = 5,
) -> ConversationSearchResult:
    """Search recent conversation turns by embedding similarity.

    Parameters
    ----------
    messages:
        List of conversation turns in *chronological* order (the last
        ``window`` of them are the search candidates). Accepts
        ConversationMessage, dicts, or langchain-style objects.
    query:
        The query to match. Either a raw string (embedded here) or a
        pre-computed embedding (list of floats).
    embedder:
        Object exposing ``.embed(texts) -> list[list[float]]`` and
        ``.dimension`` (real: EmbeddingService; fake in tests).
    window:
        How many of the most recent turns to consider (default 40).
    top_k:
        How many best-matching turns to return (default 5).

    Returns
    -------
    ConversationSearchResult sorted by descending score.
    """
    # Normalize the candidate list and keep only the most recent `window`.
    turns = [ConversationMessage.build(m) for m in messages]
    turns = [t for t in turns if not t.is_empty][-window:] if window >= 0 else []

    result = ConversationSearchResult(
        num_message_turns=len(turns),
    )

    if not turns:
        return result

    # Resolve the query vector (embed a string, or use the provided one).
    if isinstance(query, str):
        if not query.strip():
            return result
        query_vec = _embed(embedder, [query])[0]
    else:
        query_vec = list(query)

    if not query_vec:
        return result

    # Embed any candidate turns that do not already carry an embedding.
    missing = [i for i, t in enumerate(turns) if not t.embedding]
    if missing:
        vectors = _embed(embedder, [turns[i].content for i in missing])
        for i, vec in zip(missing, vectors):
            turns[i].embedding = vec

    scored: list[ConversationHit] = []
    for turn in turns:
        if not turn.embedding:
            continue
        score = _cosine(query_vec, turn.embedding)
        scored.append(ConversationHit(message=turn, score=score))

    scored.sort(key=lambda h: h.score, reverse=True)

    result.hits = scored[: max(top_k, 0)]
    result.best_score = result.hits[0].score if result.hits else 0.0
    result.mean_score = (
        sum(h.score for h in result.hits) / len(result.hits)
        if result.hits
        else 0.0
    )
    result.num_searched = len(scored)

    return result
