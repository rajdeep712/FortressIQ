"""Reciprocal Rank Fusion (RRF) over conversation + document hits.

Qdrant already fuses *dense + BM25* inside its hybrid search (RRF/DBSF).
This is a *second* fusion step that merges the top conversation hits with
the Qdrant document hits before cross-encoder reranking.

RRF score: sum over the two ranked lists of 1/(k + rank). Items seen in
only one list still contribute; items in both are boosted by the extra
occurrence.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel


class FusionItem(BaseModel):
    """A rankable item with an optional source tag (conversation|document)."""

    item_id: str
    source: str = "document"  # 'document' | 'conversation'
    payload: dict = {}
    score: float | None = None  # RRF score, set by rrf_fuse


def _rrf_score(
    rank: int,
    k: int,
) -> float:
    return 1.0 / (k + rank)


def rrf_fuse(
    conversation: Sequence[FusionItem],
    documents: Sequence[FusionItem],
    *,
    k: int = 60,
    conversation_boost: float = 1.0,
) -> list[FusionItem]:
    """Fuse two ranked lists into a merged RRF ordering (descending score).

    ``k`` is the standard RRF constant (default 60). Items that appear in
    both lists accumulate score from each occurrence. Tie-breaks prefer the
    higher individual rank (lower index).
    """
    scores: dict[str, float] = {}

    for rank, item in enumerate(conversation, start=1):
        scores[item.item_id] = (
            scores.get(item.item_id, 0.0)
            + conversation_boost * _rrf_score(rank, k)
        )

    for rank, item in enumerate(documents, start=1):
        scores[item.item_id] = (
            scores.get(item.item_id, 0.0) + _rrf_score(rank, k)
        )

    # Keep the first occurrence's source/payload for each id.
    seen: dict[str, FusionItem] = {}
    for item in [*conversation, *documents]:
        seen.setdefault(item.item_id, item)

    ordered = sorted(
        seen.values(),
        key=lambda it: scores[it.item_id],
        reverse=True,
    )
    return [
        FusionItem(
            item_id=it.item_id,
            source=it.source,
            payload=it.payload,
            score=scores[it.item_id],
        )
        for it in ordered
    ]
