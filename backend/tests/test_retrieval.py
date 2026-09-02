"""Tests for the first three retrieval steps (preprocess, conversation
search, sufficiency routing) and the RetrieverService orchestrator.

Uses a deterministic FakeEmbeddingService so scores are fully predictable:
each string maps to a normalized indicator vector over a fixed vocabulary, so
cosine similarity is exactly the token-overlap ratio (no torch / real model).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest

from app.retrieval import (
    ConversationMessage,
    ConversationSearchResult,
    DocumentSearchResult,
    RouterDecision,
    analyze_query,
    decide,
    retrieve_documents,
    search_conversation,
)
from app.retrieval.query import QueryAnalysis
from app.services.retriever_service import RetrieverService

VOCABULARY = [
    "redis", "qdrant", "faiss", "chunk", "size", "document", "uploaded",
    "pipeline", "job", "state", "retry", "worker", "architecture", "page",
]


class FakeEmbeddingService:
    """Deterministic embedder: token-indicator vector -> exact cosines."""

    def __init__(self, vocabulary: list[str] | None = None):
        self.vocabulary = vocabulary if vocabulary is not None else VOCABULARY
        self.dimension = len(self.vocabulary) + 1  # last slot = noise/empty
        self.embed_calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        words = set(re.findall(r"[a-z0-9']+", text.lower()))
        v = [0.0] * self.dimension
        for idx, tok in enumerate(self.vocabulary):
            if tok in words:
                v[idx] = 1.0
        if not any(v[: len(self.vocabulary)]):
            v[-1] = 1.0  # noise slot so unmatched text is orthogonal to all
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vector(t) for t in texts]


def make_msgs(*pairs: str) -> list[dict]:
    """Build conversation turns: ("user", "text"), ("assistant", "text"), ..."""
    msgs = []
    for i, (role, content) in enumerate(pairs):
        msgs.append({"role": role, "content": content, "id": f"m{i}"})
    return msgs


# ---------------------------------------------------------------------------
# Step 1: Query preprocessing
# ---------------------------------------------------------------------------


class TestAnalyzeQuery:
    def test_normalizes_whitespace_and_punctuation(self):
        result = analyze_query("  Why   did we use Redis??? ")
        assert result.normalized_query == "Why did we use Redis?"
        assert result.original_query == "  Why   did we use Redis??? "

    def test_handles_none_blank_input(self):
        for raw in (None, "", "   \t  "):
            result = analyze_query(raw)
            assert result.normalized_query == ""
            assert result.original_query == (raw or "")
            assert result.has_conversation_reference is False
            assert result.explicit_document_reference is False

    def test_detects_conversational_anaphora(self):
        assert analyze_query("Why did we use Redis?").has_conversation_reference is True
        assert analyze_query("Explain that in more detail.").has_conversation_reference is True
        assert analyze_query("Continue from earlier.").has_conversation_reference is True
        assert analyze_query("What did you recommend last time?").has_conversation_reference is True

    def test_plain_question_has_no_conversation_reference(self):
        assert analyze_query("What is the capital of France?").has_conversation_reference is False

    def test_detects_explicit_document_reference(self):
        assert analyze_query(
            "According to the uploaded PDF what is the chunk size?"
        ).explicit_document_reference is True
        assert analyze_query(
            "What does the architecture document say?"
        ).explicit_document_reference is True
        assert analyze_query(
            "The report mentions a parent chunk size."
        ).explicit_document_reference is True

    def test_doc_reference_flags_truly_signal_only(self):
        # A conversational "why did we" question is NOT a document reference.
        result = analyze_query("Why did we choose Qdrant for ingestion?")
        assert result.explicit_document_reference is False

    def test_both_flags_combine(self):
        result = analyze_query(
            "Earlier you mentioned the document recommends a worker count."
        )
        assert result.has_conversation_reference is True
        assert result.explicit_document_reference is True


# ---------------------------------------------------------------------------
# Step 2: Conversation search
# ---------------------------------------------------------------------------


class TestSearchConversation:
    def test_empty_input_returns_empty_result(self):
        result = search_conversation([], "hello", FakeEmbeddingService())
        assert result.hits == []
        assert result.best_score == 0.0
        assert result.num_message_turns == 0

    def test_ranks_matching_messages_by_cosine(self):
        embedder = FakeEmbeddingService()
        msgs = make_msgs(
            ("assistant", "Redis will maintain job state and retries"),
            ("user", "We are building a pipeline using Redis"),
        )
        result = search_conversation(
            msgs, "why did we use Redis for job state", embedder
        )
        assert result.hits
        assert result.hits[0].score == pytest.approx(1.0)
        assert result.hits[0].message.role == "assistant"
        assert result.best_score == pytest.approx(1.0)

    def test_top_k_limits_returned_hits(self):
        embedder = FakeEmbeddingService()
        msgs = [{"role": "assistant", "content": f"topic {i}"} for i in range(8)]
        strong = [{"role": "user", "content": f"topic {i}"} for i in range(8)]
        all_msgs = msgs + strong
        result = search_conversation(all_msgs, "topic x", embedder, top_k=5)
        assert len(result.hits) == 5
        assert result.num_searched == len(all_msgs)

    def test_window_limits_to_recent_messages(self):
        embedder = FakeEmbeddingService()
        # position 0 matches, but it is outside the last-10 window.
        msgs = make_msgs(
            *[("user", f"m{i}") for i in range(20)]
        )
        msgs[0]["content"] = "Redis job state"
        msgs[19]["content"] = "Redis job state"
        result = search_conversation(msgs, "Redis job state", embedder, window=10)
        assert result.num_searched == 10
        returned_ids = {h.message.message_id for h in result.hits}
        assert "m19" in returned_ids
        assert "m0" not in returned_ids

    def test_reuses_stored_embeddings_when_present(self):
        embedder = FakeEmbeddingService()
        query_vec = embedder.embed(["redis job state"])[0]
        msgs = [
            ConversationMessage(
                message_id="m1",
                role="assistant",
                content="Redis job state",
                embedding=query_vec,
            )
        ]
        before = len(embedder.embed_calls)
        result = search_conversation(msgs, query_vec, embedder)
        # The message already had an embedding, so no message embedding call.
        assert result.hits[0].score == pytest.approx(1.0)
        assert len(embedder.embed_calls) == before

    def test_build_accepts_mappings_and_langgraph_like_objects(self):
        mapped = ConversationMessage.build({"role": "assistant", "content": "hi", "id": "a"})
        assert mapped.role == "assistant" and mapped.content == "hi"
        obj = ConversationMessage.build(
            SimpleNamespace(id="b", type="human", content="hello there")
        )
        assert obj.role == "user" and obj.message_id == "b"
        assert ConversationMessage.build(ConversationMessage(content="x")).content == "x"


# ---------------------------------------------------------------------------
# Step 3: Sufficiency routing
# ---------------------------------------------------------------------------


def routed(raw_query: str, msgs: list[dict], **kwargs) -> RouterDecision:
    analysis = analyze_query(raw_query)
    search = search_conversation(msgs, analysis.normalized_query, FakeEmbeddingService())
    return decide(analysis, search, **kwargs)


class TestDecide:
    def test_high_dependency_routes_conversation_only(self):
        decision = routed(
            "why did we use Redis for job state",
            make_msgs(
                ("assistant", "Redis will maintain job state and retries"),
            ),
        )
        assert decision.path == "conversation_only"
        assert decision.use_conversation is True
        assert decision.retrieve_documents is False

    def test_medium_dependency_routes_hybrid(self):
        decision = routed(
            "what chunk size should we use",
            make_msgs(
                ("assistant", "Chunk and size the document"),
            ),
        )
        assert decision.path == "hybrid"
        assert decision.use_conversation is True
        assert decision.retrieve_documents is True

    def test_low_dependency_routes_qdrant_only(self):
        decision = routed(
            "recommend an embedding model with high accuracy",
            make_msgs(
                ("assistant", "We recommend Redis for worker state"),
            ),
        )
        assert decision.path == "qdrant_only"
        assert decision.use_conversation is False
        assert decision.retrieve_documents is True

    def test_empty_history_routes_qdrant_only(self):
        analysis = analyze_query("why did we use Redis")
        search = search_conversation([], analysis.normalized_query, FakeEmbeddingService())
        decision = decide(analysis, search)
        assert decision.path == "qdrant_only"
        assert decision.retrieve_documents is True

    def test_explicit_document_reference_lowers_reliance(self):
        # Hold the conversation search fixed and toggle only the doc-reference
        # signal: an explicit document reference must drop the dependency out of
        # the conversation-only band (-> hybrid), because convo alone can't answer.
        from app.retrieval import ConversationHit
        from app.retrieval.query import QueryAnalysis

        message = ConversationMessage(
            message_id="m1", role="assistant", content="chunk and size and document"
        )
        search = ConversationSearchResult(
            hits=[ConversationHit(message=message, score=1.0)],
            best_score=1.0,
            mean_score=1.0,
            num_searched=1,
            num_message_turns=1,
        )

        no_doc_analysis = QueryAnalysis(
            original_query="chunk and size",
            normalized_query="chunk and size",
            has_conversation_reference=False,
            explicit_document_reference=False,
        )
        with_doc_analysis = no_doc_analysis.model_copy(
            update={"explicit_document_reference": True}
        )

        no_doc = decide(no_doc_analysis, search)
        with_doc = decide(with_doc_analysis, search)

        assert no_doc.path == "conversation_only"
        assert with_doc.path == "hybrid"
        assert with_doc.conversation_dependency < no_doc.conversation_dependency

    def test_dangling_conversation_reference_does_not_force_conversation(self):
        # Anaphora ("that") with no matching message => dependency stays low.
        decision = routed(
            "explain that in detail",
            make_msgs(("assistant", "Redis is unrelated content here")),
        )
        assert decision.path == "qdrant_only"

    def test_thresholds_are_injectable(self):
        decision = routed(
            "what chunk size should we use",
            make_msgs(("assistant", "Chunk and size the document")),
            conversation_only_threshold=0.5,
            hybrid_threshold=0.4,
        )
        assert decision.path == "conversation_only"


# ---------------------------------------------------------------------------
# RetrieverService orchestrator
# ---------------------------------------------------------------------------


class TestRetrieverService:
    def test_run_returns_decision_end_to_end(self):
        service = RetrieverService(
            embedder=FakeEmbeddingService(),
            min_hits=1,
            overlap_threshold=0.6,
            conversation_only_threshold=0.8,
            hybrid_threshold=0.55,
        )
        decision = service.run(
            "why did we use Redis for job state",
            make_msgs(("assistant", "Redis will maintain job state and retries")),
        )
        assert isinstance(decision, RouterDecision)
        assert decision.analysis is not None
        assert decision.analysis.normalized_query == "why did we use Redis for job state"
        assert decision.search is not None and decision.search.hits
        assert decision.path == "conversation_only"

    def test_run_tolerates_malformed_query(self):
        service = RetrieverService(embedder=FakeEmbeddingService())
        decision = service.run(None, [])
        assert decision.path == "qdrant_only"
        assert isinstance(decision, RouterDecision)

    def test_search_exposes_conversation_hits_for_fusion(self):
        service = RetrieverService(embedder=FakeEmbeddingService())
        analysis = service.analyze("why did we use Redis")
        assert isinstance(analysis, QueryAnalysis)
        search = service.search(
            analysis.normalized_query,
            make_msgs(("assistant", "Redis job state")),
        )
        assert search.conversation == search.hits
        assert search.hits


# ---------------------------------------------------------------------------
# Step 4: Document (Qdrant) hybrid retrieval + rerank
# ---------------------------------------------------------------------------


class FakeSparseEmbedder:
    """Deterministic BM25-ish sparse embedder for tests."""

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(("embed", list(texts)))
        return [self._enc(t) for t in texts]

    def query_embed(self, text):
        self.calls.append(("query_embed", text))
        return self._enc(text)

    @staticmethod
    def _enc(text):
        words = re.findall(r"[a-z0-9']+", text.lower())
        indices = sorted({hash(w) % 100 for w in words})
        return {
            "indices": indices,
            "values": [1.0] * len(indices),
        }


class FakeReranker:
    def __init__(self, available=True, order=None, raise_on_first=False):
        self.available_flag = available
        self.order = order
        self.raise_on_first = raise_on_first
        self.calls = 0

    @property
    def available(self):
        return self.available_flag

    def rerank(self, query, passages, top_k):
        self.calls += 1
        if self.raise_on_first:
            self.raise_on_first = False
            return []
        if self.order is not None:
            return [i for i in self.order if i < len(passages)][:top_k]
        # default: reverse order (worst first)
        return list(range(len(passages) - 1, -1, -1))[:top_k]


class FakeQdrant:
    def __init__(self, points, required_fields=None, raise_error=None):
        self.points = points
        self.required_fields = required_fields or {}
        self.raise_error = raise_error
        self.hybrid_calls = []

    def hybrid_search(self, query_dense, query_sparse, **kwargs):
        self.hybrid_calls.append(
            (query_dense, query_sparse, kwargs)
        )
        if self.raise_error:
            raise self.raise_error
        for key, value in self.required_fields.items():
            assert kwargs[key] == value, (key, kwargs[key], value)
        return self.points


def _point(chunk_id, score=0.5, text="hello", doc_id="d1", filename="f.pdf"):
    return {
        "id": chunk_id,
        "score": score,
        "payload": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "filename": filename,
            "section_path": ["Intro"],
            "locations": [{"type": "pdf_bbox", "page": 1}],
        },
    }


def test_document_search_passes_tenant_filter_and_returns_hits():
    embedder = FakeEmbeddingService()
    sparse = FakeSparseEmbedder()
    qdrant = FakeQdrant(
        points=[_point("c1", 0.8), _point("c2", 0.6)],
        required_fields={
            "user_id": "u1",
            "doc_ids": ["d1", "d2"],
            "chunk_type": "child",
        },
    )
    reranker = FakeReranker(order=[1, 0])

    result = retrieve_documents(
        "why qdrant",
        user_id="u1",
        selected_doc_ids=["d1", "d2"],
        embedder=embedder,
        sparse_embedder=sparse,
        qdrant=qdrant,
        reranker=reranker,
        top_k=2,
    )

    assert sparse.calls and sparse.calls[0][0] == "query_embed"
    assert isinstance(result, DocumentSearchResult)
    assert [h.chunk_id for h in result.hits] == ["c2", "c1"]
    assert result.rerank_used is True
    assert result.reranked == 2
    assert result.hits[0].doc_id == "d1"
    assert result.hits[0].section_path == ["Intro"]


def test_document_search_reranker_absent_falls_back_to_fusion_order():
    qdrant = FakeQdrant(
        points=[_point("c1", 0.8), _point("c2", 0.6), _point("c3", 0.4)]
    )
    result = retrieve_documents(
        "why qdrant",
        user_id="u1",
        embedder=FakeEmbeddingService(),
        sparse_embedder=FakeSparseEmbedder(),
        qdrant=qdrant,
        reranker=FakeReranker(available=False),
        top_k=2,
    )
    assert [h.chunk_id for h in result.hits] == ["c1", "c2"]
    assert result.rerank_used is False


def test_document_search_raises_when_rerank_required_and_unavailable():
    with pytest.raises(RuntimeError, match="rerank"):
        retrieve_documents(
            "why qdrant",
            user_id="u1",
            embedder=FakeEmbeddingService(),
            sparse_embedder=FakeSparseEmbedder(),
            qdrant=FakeQdrant(points=[_point("c1")]),
            reranker=FakeReranker(available=False),
            rerank_required=True,
        )


def test_retriever_service_retrieve_documents_requires_qdrant():
    service = RetrieverService(embedder=FakeEmbeddingService())
    with pytest.raises(RuntimeError, match="Qdrant"):
        service.retrieve_documents("why qdrant", user_id="u1")


def test_retriever_service_retrieve_documents_end_to_end():
    qdrant = FakeQdrant(points=[_point("c1", 0.9)])
    service = RetrieverService(
        embedder=FakeEmbeddingService(),
        sparse_embedder=FakeSparseEmbedder(),
        qdrant=qdrant,
        reranker=FakeReranker(),
    )
    result = service.retrieve_documents(
        "why qdrant",
        user_id="u1",
        selected_doc_ids=["d1"],
    )
    assert isinstance(result, DocumentSearchResult)
    assert [h.chunk_id for h in result.hits] == ["c1"]
