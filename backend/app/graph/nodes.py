"""Graph nodes: the retrieval + generation steps of the RAG pipeline.

Kept free of FastAPI/Router imports so the graph is unit-testable with fakes.
Shared services live in ``GraphDeps`` (constructed in the API layer).
"""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.types import StreamWriter

from app.core.config import settings
from app.core.tracing import maybe_span, set_span_attributes
from app.ingestion.embedding import EmbeddingService, SparseEmbeddingService
from app.models.conversation import ChatMessage
from app.repositories.conversation_repository import ConversationRepository
from app.retrieval.conversation_search import (
    ConversationMessage,
    ConversationSearchResult,
    search_conversation,
)
from app.retrieval.document_search import retrieve_documents
from app.retrieval.fusion import FusionItem, rrf_fuse
from app.retrieval.query import analyze_query
from app.retrieval.reranker import FlashRankReranker
from app.retrieval.route import decide
from app.services.enrichment import enrich_chunks
from app.services.llm_service import SYSTEM_PROMPT, LLMService, parse_citations
from app.services.qdrant_service import QdrantService


class GraphDeps:
    """Wires the services the graph needs. Constructed once in the API layer
    (or in tests with fakes swapped in)."""

    def __init__(
        self,
        qdrant: QdrantService | None = None,
        embedder: Any | None = None,
        sparse_embedder: Any | None = None,
        reranker: Any | None = None,
        llm: LLMService | None = None,
        conversation_repository: ConversationRepository | None = None,
        conversation_window: int | None = None,
        conversation_top_k: int | None = None,
    ):
        self.embedder = embedder if embedder is not None else EmbeddingService()
        self.sparse_embedder = (
            sparse_embedder if sparse_embedder is not None else SparseEmbeddingService()
        )
        self.qdrant = qdrant
        self.reranker = reranker if reranker is not None else FlashRankReranker()
        self.llm = llm if llm is not None else LLMService()
        self.conversation_repository = conversation_repository
        self.conversation_window = (
            conversation_window
            if conversation_window is not None
            else settings.conversation_window
        )
        self.conversation_top_k = (
            conversation_top_k
            if conversation_top_k is not None
            else settings.conversation_top_k
        )


def _db_message_to_conversation(msg: ChatMessage) -> ConversationMessage:
    embedding: list[float] | None = None
    if msg.embedding_json:
        try:
            embedding = json.loads(msg.embedding_json)
        except json.JSONDecodeError:
            embedding = None
    return ConversationMessage(
        message_id=msg.message_id,
        role=msg.role,
        content=msg.content,
        timestamp=msg.created_at,
        embedding=embedding,
    )


async def preprocess_node(state: dict, deps: GraphDeps) -> dict:
    with maybe_span(
        "graph.preprocess",
        kind="CHAIN",
        query_length=len(state.get("query", "")),
    ):
        try:
            analysis = analyze_query(state.get("query", ""))
        except Exception:
            analysis = analyze_query("")
    return {"analysis": analysis.model_dump()}


async def search_conversation_node(state: dict, deps: GraphDeps) -> dict:
    """Load recent history (with stored embeddings) and run similarity search."""
    chat_id = state.get("chat_id")
    with maybe_span("graph.search_conversation", kind="CHAIN", chat_id=chat_id or ""):
        start = time.perf_counter()

        messages: list[ConversationMessage] = []
        if deps.conversation_repository is not None and chat_id:
            rows = deps.conversation_repository.get_messages(
                chat_id, limit=deps.conversation_window
            )
            messages = [_db_message_to_conversation(m) for m in rows]

        result: ConversationSearchResult = search_conversation(
            messages,
            state.get("query", ""),
            deps.embedder,
            window=deps.conversation_window,
            top_k=deps.conversation_top_k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        lean_hits = [
            {
                "message": h.message.model_dump(exclude={"embedding"}),
                "score": h.score,
            }
            for h in result.hits
        ]
        set_span_attributes(
            num_message_turns=result.num_message_turns,
            num_searched=result.num_searched,
            best_score=result.best_score,
            hits=len(lean_hits),
        )
    return {
        "conversation_hits": lean_hits,
        "conversation_search": {
            "hits": lean_hits,
            "best_score": result.best_score,
            "mean_score": result.mean_score,
            "num_searched": result.num_searched,
            "num_message_turns": result.num_message_turns,
        },
        "conversation_dependency": result.best_score,
        "retrieval_latency_ms": elapsed_ms,
    }


def _decode_conversation_search(raw: Any) -> ConversationSearchResult:
    if not raw:
        return ConversationSearchResult()
    return ConversationSearchResult.model_validate(raw)


async def route_node(state: dict, deps: GraphDeps) -> dict:
    from app.retrieval.query import QueryAnalysis

    with maybe_span("graph.route", kind="CHAIN"):
        analysis = QueryAnalysis.model_validate(state["analysis"])
        search = _decode_conversation_search(state.get("conversation_search"))
        decision = decide(analysis, search)
    return {
        "route_decision": decision.path,
        "route_reason": decision.reason,
        "conversation_dependency": decision.conversation_dependency,
    }


async def retrieve_documents_node(state: dict, deps: GraphDeps) -> dict:
    if deps.qdrant is None:
        return {"document_hits": []}

    with maybe_span(
        "graph.retrieve_documents",
        kind="CHAIN",
        user_id=state.get("user_id", ""),
        selected_doc_count=len(state.get("selected_doc_ids") or []),
    ):
        result = retrieve_documents(
            state.get("query", ""),
            user_id=state.get("user_id", ""),
            selected_doc_ids=state.get("selected_doc_ids"),
            embedder=deps.embedder,
            sparse_embedder=deps.sparse_embedder,
            qdrant=deps.qdrant,
            reranker=deps.reranker,
            top_k=deps.conversation_top_k,
        )
        set_span_attributes(hits=len(result.hits))
    return {"document_hits": [d.model_dump() for d in result.hits]}


async def fuse_node(state: dict, deps: GraphDeps) -> dict:
    """RRF-fuse conversation hits + document hits (hybrid path only)."""
    documents = state.get("document_hits", [])
    with maybe_span(
        "graph.fuse",
        kind="CHAIN",
        document_count=len(documents),
        hybrid=state.get("route_decision") == "hybrid",
    ):
        if state.get("route_decision") != "hybrid":
            # qdrant_only: documents pass through unfused.
            return {
                "fusible": False,
                "fused_hits": [
                    {
                        "item_id": d["chunk_id"],
                        "source": "document",
                        "payload": d,
                        "rrf_score": None,
                    }
                    for d in documents
                ],
            }

        doc_items = [
            FusionItem(item_id=d["chunk_id"], source="document", payload=d)
            for d in documents
        ]
        prev = state.get("conversation_search")
        conv_result = _decode_conversation_search(prev)
        conv_items = [
            FusionItem(
                item_id=h.message.message_id or f"conv-{i}",
                source="conversation",
                payload={
                    "role": h.message.role,
                    "content": h.message.content,
                },
            )
            for i, h in enumerate(conv_result.hits[:2])
        ]

        fused = rrf_fuse(
            conv_items,
            doc_items,
            k=settings.fusion_k,
            conversation_boost=1.0,
        )
        set_span_attributes(fused_count=len(fused))
    return {
        "fusible": True,
        "fused_hits": [
            {
                "item_id": f.item_id,
                "source": f.source,
                "payload": f.payload,
                "rrf_score": f.score,
            }
            for f in fused
        ],
    }


async def rerank_node(state: dict, deps: GraphDeps) -> dict:
    fused = state.get("fused_hits", [])
    # Only document payloads become citation candidates; conversation stays
    # as conversational context only.
    doc_payloads = [f["payload"] for f in fused if f.get("source") == "document"]

    if not doc_payloads or not deps.reranker.available:
        return {"document_hits": doc_payloads}

    with maybe_span(
        "graph.rerank",
        kind="RERANKER",
        input_count=len(doc_payloads),
    ):
        query = state.get("query", "")
        texts = [d.get("text") or "" for d in doc_payloads]
        scored_ranks = _rerank_with_scores(
            deps.reranker, query, texts, top_k=deps.conversation_top_k
        )
        ordered = []
        for i, score in scored_ranks:
            d = dict(doc_payloads[i])
            d["rerank_score"] = score
            ordered.append(d)
        if not ordered:
            ordered = doc_payloads
        set_span_attributes(output_count=len(ordered))
    return {"document_hits": ordered[: deps.conversation_top_k]}


def _rerank_with_scores(
    reranker: Any,
    query: str,
    texts: list[str],
    top_k: int,
) -> list[tuple[int, float]]:
    """Return ``(index, score)`` pairs for reranked passage texts.

    Prefers ``rerank_with_scores`` when available (real cross-encoder
    scores); otherwise falls back to plain ``rerank`` with synthetic 0.0
    scores so logging keeps a uniform shape.
    """
    scored = getattr(reranker, "rerank_with_scores", None)
    if callable(scored):
        return scored(query, texts, top_k)
    indices = reranker.rerank(query, texts, top_k)
    return [(i, 0.0) for i in indices]


async def enrich_node(state: dict, deps: GraphDeps) -> dict:
    if deps.conversation_repository is None:
        return {"parents": [], "children": []}

    child_ids = [d.get("chunk_id") for d in state.get("document_hits", [])]
    with maybe_span(
        "graph.enrich",
        kind="CHAIN",
        child_count=len(child_ids),
    ):
        ctx = enrich_chunks(deps.conversation_repository.db, child_ids)
        set_span_attributes(
            parents=len(ctx.parents),
            children=len(ctx.children),
        )
    return {"parents": ctx.parents, "children": ctx.children}


async def generate_node(
    state: dict,
    deps: GraphDeps,
    writer: StreamWriter | None = None,
) -> dict:
    """Build the prompt, stream the LLM answer, and parse citations.

    Yields token chunks through the injected ``writer`` (SSE). The final
    return carries the clean answer text + parsed citations.
    """
    query = state.get("query", "")
    parents = state.get("parents", [])
    children = state.get("children", [])
    conv_result = _decode_conversation_search(state.get("conversation_search"))

    with maybe_span(
        "graph.generate",
        kind="CHAIN",
        model=getattr(deps.llm, "model", "") or "",
        grounding_count=len(parents),
    ):
        user_prompt = _build_prompt(query, parents, children, conv_result)

        full = ""
        async for chunk in deps.llm.stream(SYSTEM_PROMPT, user_prompt):
            full += chunk
            if writer is not None:
                writer(chunk)

        text, citations = parse_citations(full)
        if not citations and children:
            # Fall back to the retrieved children as citation metadata so the
            # UI still gets highlightable pointers.
            citations = [
                _child_as_citation(c, idx)
                for idx, c in enumerate(children, start=1)
            ]

        set_span_attributes(
            output_chars=len(full),
            answer_chars=len(text),
            citations=len(citations),
        )

    return {
        "answer": text,
        "citations": citations,
        "model": deps.llm.model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }


async def persist_node(state: dict, deps: GraphDeps) -> dict:
    """Persistence hook. The API layer writes user/assistant messages and
    enqueues background embedding before/after invoking the graph; this node
    exists so the graph topology includes a persist step."""
    return {}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _build_prompt(
    query: str,
    parents: list[dict],
    children: list[dict],
    conversation: ConversationSearchResult,
) -> str:
    lines: list[str] = []

    if conversation.hits:
        lines.append("Recent conversation (for context):")
        for h in conversation.hits[:3]:
            lines.append(f"- ({h.message.role}) {h.message.content}")
        lines.append("")

    if parents:
        lines.append("Grounded context (use ONLY this):")
        for i, p in enumerate(parents, start=1):
            lines.append(f"[{i}] {p.get('text', '')}")
        lines.append("")

        cit_map = []
        for idx, child in enumerate(children, start=1):
            cit_map.append(_child_citation_meta(child, idx))
        if cit_map:
            lines.append("Your citation ids must map to these positions:")
            lines.append(json.dumps(cit_map))
            lines.append("")

    lines.append(f"User question: {query}")
    return "\n".join(lines)


def _child_citation_meta(child: dict, idx: int) -> dict:
    return {
        "id": idx,
        "doc_id": child.get("doc_id", ""),
        "page": child.get("page", 1),
        "chunk_title": (child.get("section_path") or [""])[-1] or "Document passage",
        "snippet": (child.get("text") or "")[:300],
        "confidence": 0.0,
        "score": 0.0,
        "bounding_box": child.get("bounding_box")
        or {"top": 0, "left": 0, "width": 100, "height": 100},
    }


def _child_as_citation(child: dict, idx: int) -> dict:
    return _child_citation_meta(child, idx)
