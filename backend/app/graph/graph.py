"""LangGraph pipeline for a single RAG query turn.

Flow::

    preprocess -> search_conversation -> route
        route == conversation_only  -> generate
        else                        -> retrieve_documents -> fuse -> rerank -> enrich -> generate
    generate -> persist

``messages`` channel uses ``add_messages`` so the compiled graph is runnable
with a checkpointer keyed on ``thread_id = chat_id`` for per-chat persistent
memory (Postgres via langgraph-checkpoint-postgres).
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamWriter

from app.graph.nodes import (
    GraphDeps,
    enrich_node,
    fuse_node,
    generate_node,
    persist_node,
    preprocess_node,
    rerank_node,
    retrieve_documents_node,
    route_node,
    search_conversation_node,
)
from app.graph.state import RAGState


def build_graph(
    deps: GraphDeps,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the RAG StateGraph with the given deps + optional checkpointer."""

    builder = StateGraph(RAGState)

    builder.add_node(
        "preprocess",
        _bind_async(preprocess_node, deps),
    )
    builder.add_node(
        "search_conversation",
        _bind_async(search_conversation_node, deps),
    )
    builder.add_node("route", _bind_async(route_node, deps))
    builder.add_node("retrieve_documents", _bind_async(retrieve_documents_node, deps))
    builder.add_node("fuse", _bind_async(fuse_node, deps))
    builder.add_node("rerank", _bind_async(rerank_node, deps))
    builder.add_node("enrich", _bind_async(enrich_node, deps))
    builder.add_node("generate", _generate(deps))
    builder.add_node("persist", _bind_async(persist_node, deps))

    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "search_conversation")
    builder.add_edge("search_conversation", "route")

    def _after_route(state: dict) -> str:
        if state.get("route_decision") == "conversation_only":
            return "conversation_only"
        return "documents"

    builder.add_conditional_edges(
        "route",
        _after_route,
        {
            "conversation_only": "generate",
            "documents": "retrieve_documents",
        },
    )

    builder.add_edge("retrieve_documents", "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_edge("rerank", "enrich")
    builder.add_edge("enrich", "generate")
    builder.add_edge("generate", "persist")
    builder.add_edge("persist", END)

    return builder.compile(checkpointer=checkpointer)


def _bind_async(fn: Callable, deps: GraphDeps) -> Callable:
    async def node(state: dict) -> dict:
        return await fn(state, deps)
    return node


def _generate(deps: GraphDeps) -> Callable:
    """generate node with StreamWriter injection for token SSE streaming.

    Uses the exact protocol signature (keyword-only ``writer: StreamWriter``)
    so LangGraph injects the writer and emits ``custom`` token events.
    """

    async def node(state: dict, *, writer: StreamWriter) -> dict:
        return await generate_node(state, deps, writer)
    return node
