"""RetrieverService: orchestrates the first three retrieval steps.

This is the integration boundary a future LangGraph query node calls:

    RetrieverService.run(raw_query, recent_conversation) -> RouterDecision

It wires together:
  1. query preprocessing   -> QueryAnalysis
  2. conversation search   -> ConversationSearchResult
  3. sufficiency routing   -> RouterDecision (conversation-only / hybrid / doc-only)

The decision's `.search` (conversation hits) and the future document hits are
what a context-fusion + reranking step will consume.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from app.core.config import settings
from app.ingestion.embedding import EmbeddingService, SparseEmbeddingService
from app.retrieval import (
    ConversationSearchResult,
    DocumentSearchResult,
    QueryAnalysis,
    RouterDecision,
    analyze_query,
    decide,
    retrieve_documents,
    search_conversation,
)
from app.retrieval.reranker import FlashRankReranker
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(
        self,
        embedder: Any | None = None,
        *,
        sparse_embedder: Any | None = None,
        qdrant: Any | None = None,
        reranker: Any | None = None,
        window: int | None = None,
        top_k: int | None = None,
        min_hits: int | None = None,
        overlap_threshold: float | None = None,
        conversation_only_threshold: float | None = None,
        hybrid_threshold: float | None = None,
        reference_boost: float | None = None,
    ):
        self.embedder = embedder if embedder is not None else EmbeddingService()
        self.sparse_embedder = (
            sparse_embedder if sparse_embedder is not None else SparseEmbeddingService()
        )
        self.qdrant = qdrant
        self.reranker = reranker if reranker is not None else FlashRankReranker()
        self.window = window if window is not None else settings.conversation_search_window
        self.top_k = top_k if top_k is not None else settings.conversation_search_top_k
        self.min_hits = min_hits if min_hits is not None else settings.conversation_min_hits
        self.overlap_threshold = (
            overlap_threshold
            if overlap_threshold is not None
            else settings.conversation_overlap_threshold
        )
        self.conversation_only_threshold = (
            conversation_only_threshold
            if conversation_only_threshold is not None
            else settings.conversation_only_threshold
        )
        self.hybrid_threshold = (
            hybrid_threshold
            if hybrid_threshold is not None
            else settings.hybrid_threshold
        )
        self.reference_boost = (
            reference_boost
            if reference_boost is not None
            else settings.conversation_reference_boost
        )

    # -- Individual steps (exposed for testability / reuse) -------------------

    def analyze(self, raw_query: str | None) -> QueryAnalysis:
        return analyze_query(raw_query)

    def search(
        self,
        query: str,
        messages: Iterable[Any] | None,
    ) -> ConversationSearchResult:
        if not messages:
            return ConversationSearchResult()
        return search_conversation(
            messages,
            query,
            self.embedder,
            window=self.window,
            top_k=self.top_k,
        )

    def route(
        self,
        analysis: QueryAnalysis,
        search: ConversationSearchResult,
    ) -> RouterDecision:
        return decide(
            analysis,
            search,
            min_hits=self.min_hits,
            overlap_threshold=self.overlap_threshold,
            conversation_only_threshold=self.conversation_only_threshold,
            hybrid_threshold=self.hybrid_threshold,
            reference_boost=self.reference_boost,
        )

    def retrieve_documents(
        self,
        query: str,
        *,
        user_id: str,
        selected_doc_ids: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> DocumentSearchResult:
        """Step 4: Qdrant hybrid (dense + BM25) retrieval + rerank.

        ``user_id`` must come from the authenticated context (never the
        client); ``selected_doc_ids`` come from the frontend. Requires a
        Qdrant client; raise a clear error when none was injected.
        """
        if self.qdrant is None:
            raise RuntimeError(
                "RetrieverService.retrieve_documents requires a Qdrant client"
            )
        return retrieve_documents(
            query,
            user_id=user_id,
            selected_doc_ids=selected_doc_ids,
            embedder=self.embedder,
            sparse_embedder=self.sparse_embedder,
            qdrant=self.qdrant,
            reranker=self.reranker,
            top_k=top_k,
        )

    # -- End-to-end ------------------------------------------------------------

    def run(
        self,
        raw_query: str | None,
        messages: Iterable[Any] | None,
    ) -> RouterDecision:
        """Preprocess + conversation search + route, returning a decision."""
        try:
            analysis = self.analyze(raw_query)
        except Exception as exc:  # never let a malformed query kill the chain
            logger.warning("Query preprocessing failed: %s", exc)
            analysis = analyze_query("")

        search = self.search(analysis.normalized_query, messages)
        return self.route(analysis, search)
