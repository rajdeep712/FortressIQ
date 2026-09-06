"""Chat endpoints.

``POST /api/v1/chat`` streams an SSE answer for a user message by running
the LangGraph RAG pipeline (retrieval -> routing -> generation) with a
Postgres checkpointer keyed on the chat id (persistent per-chat memory).

Conversation history similarity search reads precomputed embeddings that were
written to the relational DB by the background worker (``embed_chat_message``);
message embedding is enqueued here so it happens off the request hot path.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.tracing import turn_context
from app.graph.checkpointer import build_checkpointer
from app.graph.nodes import GraphDeps
from app.graph.graph import build_graph
from app.ingestion.embedding import EmbeddingService, SparseEmbeddingService
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.retrieval.reranker import FlashRankReranker
from app.schemas.chat import ChatMessageOut, ChatRequest, ConversationOut
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)

# ---------------------------------------------------------------------------
# Singleton services reused across requests. Built lazily so the in-memory
# models (FlashRank, sentence-transformers fallback) load once.
# ---------------------------------------------------------------------------
_embedder = None
_sparse = None
_reranker = None
_qdrant = None


def _get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder


def _get_sparse() -> SparseEmbeddingService:
    global _sparse
    if _sparse is None:
        _sparse = SparseEmbeddingService()
    return _sparse


def _get_reranker() -> FlashRankReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlashRankReranker()
    return _reranker


def _get_qdrant() -> QdrantService | None:
    global _qdrant
    if not settings.qdrant_url:
        return None
    if _qdrant is None:
        _qdrant = QdrantService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            vector_size=settings.embedding_dimension,
            collection_name=settings.qdrant_collection_name,
            timeout_seconds=settings.qdrant_timeout_seconds,
            upsert_batch_size=settings.qdrant_upsert_batch_size,
        )
    return _qdrant


def _make_deps(db: Session) -> GraphDeps:
    repo = ConversationRepository(db)
    return GraphDeps(
        qdrant=_get_qdrant(),
        embedder=_get_embedder(),
        sparse_embedder=_get_sparse(),
        reranker=_get_reranker(),
        llm=LLMService(),
        conversation_repository=repo,
    )


async def _enqueue_embed(message_id: str, content: str) -> None:
    from arq.connections import RedisSettings, create_pool

    if not settings.redis_url or not content.strip():
        return
    redis = await create_pool(
        settings_=RedisSettings.from_dsn(settings.redis_url)
    )
    try:
        await redis.enqueue_job("embed_chat_message", message_id, content)
    except Exception:  # best-effort; never break the chat on enqueue failure
        logger.warning("Failed to enqueue embedding for message %s", message_id)
    finally:
        await redis.aclose()


def _to_message_out(row: Any) -> ChatMessageOut:
    return ChatMessageOut(
        message_id=row.message_id,
        chat_id=row.chat_id,
        role=row.role,
        content=row.content,
        citations=json.loads(row.citations_json) if row.citations_json else [],
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = ConversationRepository(db)
    rows = repo.list_conversations(current_user.user_id)
    return [
        ConversationOut(
            chat_id=r.chat_id,
            title=r.title,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@router.get("/{chat_id}/messages", response_model=list[ChatMessageOut])
async def get_messages(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = ConversationRepository(db)
    conv = repo.get_conversation(chat_id, current_user.user_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    rows = repo.get_messages(chat_id)
    return [_to_message_out(r) for r in rows]


@router.delete("/{chat_id}")
async def delete_conversation(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = ConversationRepository(db)
    if not repo.delete_conversation(chat_id, current_user.user_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

@router.post("/")
async def chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    repo = ConversationRepository(db)

    # Resolve the conversation: reuse an owned chat id or create a new one.
    chat_id = body.chat_id
    if chat_id:
        conv = repo.get_conversation(chat_id, current_user.user_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        title = body.message.strip()[:60]
        conv = repo.create_conversation(
            chat_id=uuid.uuid4().hex,
            user_id=current_user.user_id,
            title=title,
        )
        chat_id = conv.chat_id

    checkpointer = await build_checkpointer()
    try:
        app = build_graph(_make_deps(db), checkpointer=checkpointer)

        async def _stream() -> AsyncIterator[str]:
            config = {"configurable": {"thread_id": chat_id}}
            input_state = {
                "query": body.message,
                "user_id": current_user.user_id,
                "chat_id": chat_id,
                "selected_doc_ids": body.selected_doc_ids,
                "messages": [],
            }

            meta: dict[str, Any] = {
                "chat_id": chat_id,
                "route_decision": None,
                "route_reason": None,
                "num_message_turns": 0,
                "retrieval_latency_ms": 0.0,
                "has_grounding_context": False,
            }

            error: str | None = None
            meta_emitted = False
            try:
                with turn_context(chat_id=chat_id, user_id=current_user.user_id):
                    async for mode, data in app.astream(
                        input_state,
                        config=config,
                        stream_mode=["custom", "updates"],
                    ):
                        if mode == "updates":
                            _absorb_updates(meta, data)
                            continue
                        # custom => token chunk; emit the meta event just before
                        # the first token so the UI can show route info early.
                        if not meta_emitted:
                            meta_emitted = True
                            yield _sse("meta", meta)
                        yield _sse("token", {"token": data})

                    snap = await app.aget_state(config)
                    meta.update(_final_meta(snap))
            except Exception as exc:  # keep the user message even on failure
                error = str(exc)
                logger.exception("Chat generation failed for %s", chat_id)

            try:
                user_msg_id = _persist_user(repo, chat_id, body.message)
                assistant_id, assistant = _persist_assistant(
                    repo,
                    chat_id,
                    meta,
                    error,
                )
                await _enqueue_embed(user_msg_id, body.message)
                if assistant:
                    await _enqueue_embed(assistant_id, assistant)
            except Exception:
                logger.exception("Failed to persist chat messages for %s", chat_id)

            meta["has_grounding_context"] = bool(
                meta.get("parents") or meta.get("children")
            )
            state = {
                "answer": meta.get("answer", ""),
                "citations": meta.get("citations", []),
            }
            if error is not None:
                state["error"] = error
            if not meta_emitted:
                yield _sse("meta", meta)
            yield _sse("done", state)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    finally:
        conn = getattr(checkpointer, "conn", None)
        if conn is not None:
            await conn.close()


def _absorb_updates(meta: dict[str, Any], updates: dict[str, Any]) -> None:
    """Fold per-node update chunks into lightweight metadata for SSE."""

    def _node(name: str) -> dict | None:
        chunk = updates.get(name)
        return chunk if isinstance(chunk, dict) else None

    search = _node("search_conversation")
    if search is not None:
        summary = search.get("conversation_search") or {}
        meta["num_message_turns"] = summary.get("num_message_turns", 0)
        meta["retrieval_latency_ms"] = search.get("retrieval_latency_ms", 0.0)

    route = _node("route")
    if route is not None:
        meta["route_decision"] = route.get("route_decision")
        meta["route_reason"] = route.get("route_reason")

    preprocess = _node("preprocess")
    if preprocess is not None:
        meta["query_analysis"] = preprocess.get("analysis")

    enrich = _node("enrich")
    if enrich is not None:
        meta["parents"] = enrich.get("parents", [])
        meta["children"] = enrich.get("children", [])
        meta["has_grounding_context"] = bool(
            enrich.get("parents") or enrich.get("children")
        )


def _final_meta(snap: Any) -> dict[str, Any]:
    return {
        "answer": snap.values.get("answer", ""),
        "citations": snap.values.get("citations", []),
        "model": snap.values.get("model", ""),
        "route_reason": snap.values.get("route_reason"),
        "route_decision": snap.values.get("route_decision"),
    }


def _persist_user(repo: ConversationRepository, chat_id: str, content: str) -> str:
    msg_id = uuid.uuid4().hex
    repo.add_message(msg_id, chat_id, "user", content)
    return msg_id


def _persist_assistant(
    repo: ConversationRepository,
    chat_id: str,
    meta: dict[str, Any],
    error: str | None,
) -> tuple[str, str | None]:
    msg_id = uuid.uuid4().hex
    if error is not None:
        content = (
            "I couldn't generate an answer for that request. "
            "Please try again. %s" % (f"({error})" if error else "")
        )
        repo.add_message(msg_id, chat_id, "assistant", content, citations=[])
        return msg_id, None
    content = meta.get("answer", "")
    citations = meta.get("citations", [])
    repo.add_message(msg_id, chat_id, "assistant", content, citations=citations)
    return msg_id, content


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"
