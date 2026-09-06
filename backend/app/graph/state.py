"""LangGraph state for the RAG query pipeline.

``messages`` is the LangGraph-managed conversation memory (persisted by the
Postgres checkpointer, thread id = chat_id). The remaining channels carry
per-request retrieval + generation data.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RAGState(TypedDict):
    # Conversation memory (LangGraph / checkpointer).
    messages: Annotated[list, add_messages]

    # Request context.
    query: str
    user_id: str
    chat_id: str
    selected_doc_ids: list[str] | None

    # Retrieval pipeline outputs.
    analysis: dict[str, Any] | None
    route_decision: str | None
    route_reason: str | None
    conversation_dependency: float
    conversation_hits: list[dict[str, Any]]
    conversation_search: dict[str, Any] | None
    document_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    fusible: bool  # whether RRF merged conversation + documents

    # Enriched context handed to the LLM.
    parents: list[dict[str, Any]]
    children: list[dict[str, Any]]

    # Generation + metadata.
    answer: str
    citations: list[dict[str, Any]]
    retrieval_latency_ms: float
    model: str

    # Prompt trace (captured so benchmarks/logs can reproduce exactly what
    # was sent to the final LLM after enrichment).
    system_prompt: str
    user_prompt: str
