"""Tests for the chat API routing helpers (SSE framing, metadata folding,
persistence). Graph streaming itself is covered in test_graph.py; these focus
on the router glue that turns graph output into SSE + stored messages.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api.chat import (
    _absorb_updates,
    _persist_assistant,
    _persist_user,
    _sse,
    _to_message_out,
)
from app.models.conversation import ChatMessage
from app.models.document import Base
from app.repositories.conversation_repository import ConversationRepository


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()
    engine.dispose()


class TestSse:
    def test_frame_format(self):
        frame = _sse("token", {"token": "hi"})
        assert frame == 'event: token\ndata: {"token": "hi"}\n\n'

    def test_frame_serializes_json(self):
        frame = _sse("meta", {"n": 1, "ok": True})
        parsed = frame.split("\n")[1].replace("data: ", "")
        assert json.loads(parsed) == {"n": 1, "ok": True}


class TestAbsorbUpdates:
    def _meta(self):
        return {
            "chat_id": "c",
            "route_decision": None,
            "route_reason": None,
            "num_message_turns": 0,
            "retrieval_latency_ms": 0.0,
            "has_grounding_context": False,
        }

    def test_search_meta(self):
        meta = self._meta()
        _absorb_updates(meta, {
            "search_conversation": {
                "conversation_search": {"num_message_turns": 3},
                "retrieval_latency_ms": 12.5,
            }
        })
        assert meta["num_message_turns"] == 3
        assert meta["retrieval_latency_ms"] == 12.5

    def test_route_meta(self):
        meta = self._meta()
        _absorb_updates(meta, {
            "route": {"route_decision": "hybrid", "route_reason": "medium"}
        })
        assert meta["route_decision"] == "hybrid"
        assert meta["route_reason"] == "medium"

    def test_enrich_meta_sets_grounding(self):
        meta = self._meta()
        _absorb_updates(meta, {
            "enrich": {
                "parents": [{"text": "p"}],
                "children": [{"chunk_id": "c1"}],
            }
        })
        assert meta["has_grounding_context"] is True
        assert meta["parents"] == [{"text": "p"}]

    def test_ignore_unknown_nodes(self):
        meta = self._meta()
        _absorb_updates(meta, {"persist": None})
        assert meta["route_decision"] is None


class TestPersistence:
    def _repo(self, session):
        repo = ConversationRepository(session)
        repo.create_conversation("chat-1", "u1", title="Hi")
        return repo

    def test_persist_user_then_read_back(self, session):
        repo = self._repo(session)
        msg_id = _persist_user(repo, "chat-1", "hello there")
        rows = session.execute(
            select(ChatMessage)
        ).scalars().all()
        out = _to_message_out(rows[0])
        assert out.role == "user"
        assert out.content == "hello there"
        assert rows[0].message_id == msg_id

    def test_persist_assistant_with_citations(self, session):
        repo = self._repo(session)
        meta = {
            "answer": "The answer text [1]",
            "citations": [{"id": 1, "doc_id": "d1", "page": 2}],
        }
        msg_id, content = _persist_assistant(repo, "chat-1", meta, None)
        assert content == "The answer text [1]"
        row = repo.get_message(msg_id)
        assert row.role == "assistant"
        assert json.loads(row.citations_json)[0]["doc_id"] == "d1"
        out = _to_message_out(row)
        assert out.citations[0].doc_id == "d1"
        assert out.content == "The answer text [1]"

    def test_persist_assistant_on_error_writes_polite_message(self, session):
        repo = self._repo(session)
        msg_id, content = _persist_assistant(
            repo, "chat-1", {"answer": "", "citations": []}, "oops"
        )
        assert content is None
        row = repo.get_message(msg_id)
        assert row.role == "assistant"
        assert "couldn't generate" in row.content.lower()

    def test_count_and_list_after_persistence(self, session):
        repo = self._repo(session)
        _persist_user(repo, "chat-1", "q1")
        _persist_assistant(repo, "chat-1", {"answer": "a1", "citations": []}, None)
        _persist_user(repo, "chat-1", "q2")
        assert repo.count_messages("chat-1") == 3
        convs = repo.list_conversations("u1")
        assert convs[0].chat_id == "chat-1"
