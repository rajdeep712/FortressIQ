"""Step 3: Context sufficiency decision (router).

Given the query analysis (step 1) and the conversation search result (step 2),
decide how much to lean on conversation memory versus document (Qdrant)
retrieval. This is heuristic -- NO LLM, NO Qdrant -- and routes into one of
three bands via a continuous *conversation dependency* score:

    dependency  >= conversation_only_threshold  -> conversation_only
    dependency  >= hybrid_threshold             -> hybrid   (both, -> fusion/rerank)
    otherwise                                   -> qdrant_only

The thresholds are tunable knobs (constructor / settings), because a single
magic similarity threshold cannot be trusted blindly; tune against a labelled
test dataset.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.retrieval.conversation_search import ConversationSearchResult
from app.retrieval.query import QueryAnalysis


class RouterDecision(BaseModel):
    path: Literal["conversation_only", "hybrid", "qdrant_only"]
    use_conversation: bool
    retrieve_documents: bool
    conversation_dependency: float
    reason: str
    best_score: float = 0.0
    hits_selected: int = 0
    analysis: QueryAnalysis | None = None
    search: ConversationSearchResult | None = None

    model_config = ConfigDict(extra="forbid")


def _conversation_dependency(
    analysis: QueryAnalysis,
    search: ConversationSearchResult | None,
    *,
    min_hits: int,
    overlap_threshold: float,
    reference_boost: float,
) -> float:
    """Compose the continuous conversation-dependency score in [0, 1].

    Higher means the conversation is very likely sufficient to answer alone.
    """
    if (
        search is None
        or search.num_message_turns == 0
        or not search.hits
    ):
        return 0.0

    best = search.best_score
    strong = sum(
        1 for h in search.hits
        if h.score >= overlap_threshold
    )
    count_factor = min(strong / max(min_hits, 1), 1.0)

    dep = 0.5 * best + 0.3 * count_factor

    if analysis.has_conversation_reference:
        dep += reference_boost

    # A question that explicitly targets a document leans away from relying
    # on conversation alone.
    if analysis.explicit_document_reference:
        dep -= 0.15

    return max(0.0, min(1.0, dep))


def decide(
    analysis: QueryAnalysis,
    search: ConversationSearchResult | None,
    *,
    min_hits: int = 1,
    overlap_threshold: float = 0.6,
    conversation_only_threshold: float = 0.8,
    hybrid_threshold: float = 0.55,
    reference_boost: float = 0.15,
) -> RouterDecision:
    """Route a query to conversation-only / hybrid / document-only.

    All thresholds are injectable so callers (and tests) can tune them.
    """
    dep = _conversation_dependency(
        analysis,
        search,
        min_hits=min_hits,
        overlap_threshold=overlap_threshold,
        reference_boost=reference_boost,
    )

    if search is None or search.num_message_turns == 0 or not search.hits:
        return RouterDecision(
            path="qdrant_only",
            use_conversation=False,
            retrieve_documents=True,
            conversation_dependency=dep,
            reason="no usable conversation history; falling back to document retrieval",
            best_score=search.best_score if search else 0.0,
            hits_selected=len(search.hits) if search else 0,
            analysis=analysis,
            search=search,
        )

    if dep >= conversation_only_threshold:
        path: Literal["conversation_only", "hybrid", "qdrant_only"] = "conversation_only"
        use_conversation = True
        retrieve_documents = False
        reason = (
            f"conversation is highly sufficient (dependency={dep:.3f}); "
            "skipping document retrieval"
        )
    elif dep >= hybrid_threshold:
        path = "hybrid"
        use_conversation = True
        retrieve_documents = True
        reason = (
            f"conversation is moderately sufficient (dependency={dep:.3f}); "
            "fusing conversation context with document retrieval"
        )
    else:
        path = "qdrant_only"
        use_conversation = False
        retrieve_documents = True
        reason = (
            f"conversation is not sufficient (dependency={dep:.3f}); "
            "retrieving from documents only"
        )

    return RouterDecision(
        path=path,
        use_conversation=use_conversation,
        retrieve_documents=retrieve_documents,
        conversation_dependency=dep,
        reason=reason,
        best_score=search.best_score if search else 0.0,
        hits_selected=len(search.hits) if search else 0,
        analysis=analysis,
        search=search,
    )
