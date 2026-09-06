"""End-to-end graph tests: routing branches + token streaming + citations.

Runs the compiled StateGraph (with a MemorySaver checkpointer) and the node
functions directly, using deterministic fakes so scores are predictable.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.graph import build_graph
from app.graph.nodes import GraphDeps, generate_node, rerank_node
from app.models.conversation import ChatMessage
from app.retrieval.query import analyze_query

VOCABULARY = [
    "redis", "qdrant", "faiss", "chunk", "size", "document", "uploaded",
    "pipeline", "job", "state", "retry", "worker", "architecture", "page",
]


class FakeEmbeddingService:
    def __init__(self):
        self.dimension = len(VOCABULARY) + 1
        self.last_provider = "fake"

    def _vector(self, text):
        words = set(re.findall(r"[a-z0-9']+", text.lower()))
        v = [0.0] * self.dimension
        for idx, tok in enumerate(VOCABULARY):
            if tok in words:
                v[idx] = 1.0
        if not any(v[: len(VOCABULARY)]):
            v[-1] = 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts):
        return [self._vector(t) for t in texts]


class FakeSparseEmbedder:
    def __init__(self):
        self.last_provider = "fake"
        self.query_called = 0

    def query_embed(self, text):
        self.query_called += 1
        words = re.findall(r"[a-z0-9']+", text.lower())
        indices = sorted({hash(w) % 100 for w in words})
        return {
            "indices": indices,
            "values": [1.0] * (len(indices) or 1),
        }


class FakeReranker:
    available = False
    def rerank(self, *a, **k): return []


class FakeLLM:
    model = "fake-model"
    async def stream(self, system, user):
        for tok in ["Grounded", " answer"]:
            yield tok


class FakeRepo:
    """Stand-in for ConversationRepository used by the graph nodes."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.db = None

    def get_messages(self, chat_id, limit=40):
        return self.rows[:limit]


def chat_msg(mid, role, content, embedding=None):
    return ChatMessage(
        message_id=mid,
        chat_id="c",
        role=role,
        content=content,
        embedding_json=json.dumps(embedding) if embedding else None,
        created_at=datetime.now(timezone.utc),
    )


def make_deps(embedder, rows, qdrant_points=None):
    return GraphDeps(
        qdrant=None if qdrant_points is None else FakeQdrant(qdrant_points),
        embedder=embedder,
        sparse_embedder=FakeSparseEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        conversation_repository=FakeRepo(rows),
    )


class FakeQdrant:
    def __init__(self, points):
        self.points = points
        self.calls = 0

    def hybrid_search(self, *a, **k):
        self.calls += 1
        return self.points


def _matching_history(query, embedder):
    vec = embedder.embed([query])[0]
    return [chat_msg("h1", "assistant", query, embedding=vec)]


def _run(graph, query, chat_id="chat-t"):
    import asyncio

    async def _go():
        config = {"configurable": {"thread_id": chat_id}}
        inp = {
            "query": query,
            "user_id": "u1",
            "chat_id": chat_id,
            "selected_doc_ids": None,
            "messages": [],
        }
        tokens = []
        updates = []
        async for mode, data in graph.astream(
            inp, config=config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                tokens.append(data)
            else:
                updates.append(data)
        state = await graph.aget_state(config)
        return tokens, updates, state.values

    return asyncio.run(_go())


class TestGraphRouting:
    def test_conversation_only_skips_document_retrieval(self):
        embedder = FakeEmbeddingService()
        deps = make_deps(embedder, _matching_history(
            "why did we use Redis for job state", embedder
        ))
        graph = build_graph(deps, checkpointer=MemorySaver())
        tokens, updates, state = _run(graph, "why did we use Redis for job state")

        assert state["route_decision"] == "conversation_only"
        node_names = {list(u.keys())[0] for u in updates}
        assert "retrieve_documents" not in node_names and "enrich" not in node_names
        assert "".join(tokens) == "Grounded answer"
        assert state["answer"] == "Grounded answer"

    def test_qdrant_only_passes_documents_through(self):
        embedder = FakeEmbeddingService()
        deps = make_deps(embedder, [], qdrant_points=[])
        graph = build_graph(deps, checkpointer=MemorySaver())
        tokens, updates, state = _run(graph, "recommend an embedding model")

        assert state["route_decision"] == "qdrant_only"
        # No conversation fusion; fusible stays False.
        assert state["fusible"] is False
        assert state["answer"] == "Grounded answer"

    def test_hybrid_fuses_conversation_and_runs_retrieval(self):
        embedder = FakeEmbeddingService()
        history = _matching_history("Chunk and size the document", embedder)
        deps = make_deps(embedder, history, qdrant_points=[])
        graph = build_graph(deps, checkpointer=MemorySaver())
        tokens, updates, state = _run(graph, "what chunk size should we use")

        assert state["route_decision"] == "hybrid"
        node_names = {list(u.keys())[0] for u in updates}
        assert "retrieve_documents" in node_names and "enrich" in node_names
        assert state["fusible"] is True
        assert "".join(tokens) == "Grounded answer"

    def test_persist_node_hook_runs(self):
        embedder = FakeEmbeddingService()
        deps = make_deps(embedder, [])
        graph = build_graph(deps, checkpointer=MemorySaver())
        _, updates, _ = _run(graph, "hello world")
        node_names = {list(u.keys())[0] for u in updates}
        assert "persist" in node_names


class TestGenerateNode:
    def test_streams_tokens_and_parses_citations(self):
        embedder = FakeEmbeddingService()
        deps = make_deps(embedder, [])

        async def llm_stream(system, user):
            yield "See source [1]\n\n"
            yield "START_CITATIONS\n"
            yield '{"citations":[{"id":1,"doc_id":"d1","page":2}]}\n'
            yield "END_CITATIONS\n"

        deps.llm.stream = llm_stream
        deps.llm.model = "fake-model"

        children = [{
            "chunk_id": "c1", "doc_id": "d1", "text": "snippet",
            "section_path": ["Intro"], "page": 2,
            "bounding_box": {"top": 0, "left": 0, "width": 100, "height": 100},
        }]

        state = {
            "query": "q", "parents": [], "children": children,
            "conversation_search": {
                "hits": [], "best_score": 0.0, "mean_score": 0.0,
                "num_searched": 0, "num_message_turns": 0,
            },
        }

        chunks = []
        # run directly with a writer stub
        class W:
            def __call__(self, c): chunks.append(c)
        out = _run_generate(state, deps, W())
        assert "".join(chunks) == (
            "See source [1]\n\nSTART_CITATIONS\n"
            '{"citations":[{"id":1,"doc_id":"d1","page":2}]}\nEND_CITATIONS\n'
        )
        assert "START_CITATIONS" not in out["answer"]
        assert out["citations"][0]["doc_id"] == "d1"
        assert out["model"] == "fake-model"
        # The exact final prompt is captured for benchmark logging.
        assert out["system_prompt"] and "research assistant" in out["system_prompt"]
        assert out["user_prompt"] == "User question: q"

    def test_citation_fallback_to_children_when_llm_omits_citations(self):
        embedder = FakeEmbeddingService()
        deps = make_deps(embedder, [])
        children = [{
            "chunk_id": "c1", "doc_id": "d1", "text": "snippet",
            "section_path": ["Intro"], "page": 2,
            "bounding_box": {"top": 0, "left": 0, "width": 100, "height": 100},
        }]
        state = {
            "query": "q", "parents": [], "children": children,
            "conversation_search": {
                "hits": [], "best_score": 0.0, "mean_score": 0.0,
                "num_searched": 0, "num_message_turns": 0,
            },
        }
        out = _run_generate(state, deps, None)
        assert out["citations"]
        assert out["citations"][0]["doc_id"] == "d1"


def _run_generate(state, deps, writer):
    import asyncio
    return asyncio.run(generate_node(state, deps, writer))


class TestRerankNode:


    class _ScoredReranker:
        available = True

        def rerank_with_scores(self, query, texts, top_k):
            return [(1, 0.92), (0, 0.71)]  # index, score


    def test_rerank_node_attaches_scores(self):
        import asyncio

        deps = GraphDeps(
            qdrant=None,
            embedder=FakeEmbeddingService(),
            sparse_embedder=FakeSparseEmbedder(),
            reranker=self._ScoredReranker(),
            llm=FakeLLM(),
            conversation_repository=FakeRepo([]),
        )
        state = {
            "query": "q",
            "fused_hits": [
                {
                    "item_id": "c1",
                    "source": "document",
                    "payload": {"chunk_id": "c1", "text": "one"},
                    "rrf_score": 0.5,
                },
                {
                    "item_id": "c2",
                    "source": "document",
                    "payload": {"chunk_id": "c2", "text": "two"},
                    "rrf_score": 0.4,
                },
            ],
        }
        out = asyncio.run(rerank_node(state, deps))
        ids = [d["chunk_id"] for d in out["document_hits"]]
        assert ids == ["c2", "c1"]
        assert out["document_hits"][0]["rerank_score"] == 0.92
        assert out["document_hits"][1]["rerank_score"] == 0.71
