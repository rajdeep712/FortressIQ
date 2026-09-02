"""Retrieval pipeline primitives.

These are the pure, dependency-light steps that a future LangGraph retrieval
query handles use. Each step is a small Pydantic-modeled, unit-testable
function; nothing here talks to Qdrant or an LLM yet.

Step 1: Query preprocessing  (app.retrieval.query)
Step 2: Conversation search  (app.retrieval.conversation_search)
Step 3: Sufficiency routing   (app.retrieval.route)
Step 4: Document retrieval    (app.retrieval.document_search) + rerank
"""

from app.retrieval.query import QueryAnalysis, QueryContext, analyze_query
from app.retrieval.conversation_search import (
    ConversationHit,
    ConversationMessage,
    ConversationSearchResult,
    search_conversation,
)
from app.retrieval.route import RouterDecision, decide
from app.retrieval.document_search import (
    DocumentHit,
    DocumentSearchResult,
    retrieve_documents,
)
from app.retrieval.reranker import FlashRankReranker, Reranker

__all__ = [
    "QueryAnalysis",
    "QueryContext",
    "analyze_query",
    "ConversationHit",
    "ConversationMessage",
    "ConversationSearchResult",
    "search_conversation",
    "RouterDecision",
    "decide",
    "DocumentHit",
    "DocumentSearchResult",
    "retrieve_documents",
    "FlashRankReranker",
    "Reranker",
]
