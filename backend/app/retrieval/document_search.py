"""Step 4: Document (Qdrant) hybrid retrieval + rerank.

Given the normalized query, embeds it both densely and sparsely, runs a
tenant-safe Qdrant hybrid search (dense + BM25 sparse fused with RRF/DBSF),
then reranks the fused candidates with a cross-encoder (FlashRank), falling
back to the fusion order if the reranker is unavailable.

Authorization is enforced here and in the Qdrant layer: ``user_id`` is taken
from the authenticated context at the API boundary (never client-supplied)
and ``selected_doc_ids`` come from the frontend. The Qdrant ``Filter``
restricts results to ``user_id == X`` AND ``doc_id IN selected_doc_ids`` AND
``chunk_type == "child"`` (the embedded leaf chunks).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.ingestion.embedding import EmbeddingService, SparseEmbeddingService
from app.retrieval.reranker import FlashRankReranker
from app.services.qdrant_service import QdrantService


class DocumentHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str = ""
    filename: str = ""
    section_path: list[str] = Field(default_factory=list)
    locations: list[dict] = Field(default_factory=list)
    score: float = 0.0
    rerank_score: float | None = None

    model_config = ConfigDict(extra="forbid")


class DocumentSearchResult(BaseModel):
    query: str
    hits: list[DocumentHit] = Field(default_factory=list)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    reranked: int = 0
    rerank_used: bool = False

    model_config = ConfigDict(extra="forbid")


def _payload_to_hit(point: dict, rerank_score: float | None = None) -> DocumentHit:
    payload = point.get("payload", {})
    location = payload.get("locations")
    if isinstance(location, list):
        locations = [
            item for item in location
            if isinstance(item, dict)
        ]
    else:
        locations = []

    section = payload.get("section_path", [])
    return DocumentHit(
        chunk_id=str(payload.get("chunk_id", point.get("id", ""))),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        filename=str(payload.get("filename", "")),
        section_path=list(section) if isinstance(section, list) else [],
        locations=locations,
        score=float(point.get("score", 0.0)),
        rerank_score=rerank_score,
    )


def retrieve_documents(
    query_text: str,
    *,
    user_id: str,
    selected_doc_ids: Sequence[str] | None = None,
    embedder: Any | None = None,
    sparse_embedder: Any | None = None,
    qdrant: Any | None = None,
    reranker: Any | None = None,
    top_k: int | None = None,
    prefetch_dense: int | None = None,
    prefetch_sparse: int | None = None,
    fusion: str | None = None,
    rerank_required: bool | None = None,
) -> DocumentSearchResult:
    """Run the full document retrieval pipeline for a single query."""
    embedder = embedder or EmbeddingService()
    sparse_embedder = sparse_embedder or SparseEmbeddingService()
    reranker = reranker or FlashRankReranker()

    top_k = top_k if top_k is not None else settings.conversation_search_top_k
    prefetch_dense = (
        prefetch_dense
        if prefetch_dense is not None
        else settings.retrieval_prefetch_dense
    )
    prefetch_sparse = (
        prefetch_sparse
        if prefetch_sparse is not None
        else settings.retrieval_prefetch_sparse
    )
    fusion = fusion if fusion is not None else settings.retrieval_fusion
    rerank_required = (
        rerank_required
        if rerank_required is not None
        else settings.rerank_required
    )

    sparse_vectors = sparse_embedder.query_embed(
        query_text
    )

    dense_vectors = embedder.embed([query_text])
    query_dense = (
        dense_vectors[0]
        if dense_vectors
        else [0.0] * embedder.dimension
    )

    points = qdrant.hybrid_search(
        query_dense,
        sparse_vectors,
        user_id=user_id,
        doc_ids=list(selected_doc_ids) if selected_doc_ids else None,
        chunk_type="child",
        top_k=max(top_k, prefetch_dense, prefetch_sparse),
        prefetch_dense=prefetch_dense,
        prefetch_sparse=prefetch_sparse,
        fusion=fusion,
    )

    fused: list[DocumentHit] = [_payload_to_hit(p) for p in points]
    dense_candidates = 0
    sparse_candidates = 0

    order: list[int]
    if reranker.available:
        indices = reranker.rerank(query_text, fused, top_k)
        if indices:
            ordered = [fused[i] for i in indices]
            reranked = len(ordered)
        elif rerank_required:
            raise RuntimeError(
                "Reranker required but returned no results"
            )
        else:
            ordered = fused[:top_k]
            reranked = 0
        rerank_used = bool(indices)
    else:
        if rerank_required:
            raise RuntimeError(
                "Reranker is required (rerank_required=True) but unavailable"
            )
        ordered = fused[:top_k]
        reranked = 0
        rerank_used = False

    return DocumentSearchResult(
        query=query_text,
        hits=ordered[:top_k],
        dense_candidates=dense_candidates,
        sparse_candidates=sparse_candidates,
        reranked=reranked,
        rerank_used=rerank_used,
    )
