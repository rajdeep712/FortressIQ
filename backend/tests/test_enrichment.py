"""Tests for parent/child enrichment (children = UI locators, parents = LLM)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.chunker import Chunk
from app.ingestion.models import SourceLocation
from app.models.document import Base
from app.repositories.chunk_repository import ChunkRepository
from app.services.enrichment import enrich_chunks


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


def _loc(page, top=10, left=20, width=30, height=40):
    return SourceLocation(
        type="pdf_bbox",
        page=page,
        bbox={"top": top, "left": left, "width": width, "height": height},
    )


def _make_chunks(doc_id="doc1"):
    parent = Chunk(
        chunk_id="parent_1",
        parent_chunk_id=None,
        chunk_type="parent",
        text="Parent context paragraph",
        chunk_index=0,
        doc_id=doc_id,
        version_id="v1",
        user_id="u1",
        section_path=["Intro"],
        locations=[],
        metadata={},
    )
    child = Chunk(
        chunk_id="child_1",
        parent_chunk_id="parent_1",
        chunk_type="child",
        text="Child exact sentence",
        chunk_index=1,
        doc_id=doc_id,
        version_id="v1",
        user_id="u1",
        section_path=["Intro"],
        locations=[_loc(3)],
        metadata={"filename": "report.pdf"},
    )
    return [parent, child]


def _seed(session):
    ChunkRepository(session).create_many(_make_chunks())


class TestEnrichChunks:
    def test_resolves_child_and_parent(self, session):
        _seed(session)
        ctx = enrich_chunks(session, ["child_1"])
        assert len(ctx.children) == 1
        assert len(ctx.parents) == 1

        child = ctx.children[0]
        assert child["chunk_id"] == "child_1"
        assert child["doc_id"] == "doc1"
        assert child["page"] == 3
        assert child["bounding_box"] == {
            "top": 10,
            "left": 20,
            "width": 30,
            "height": 40,
        }
        assert child["section_path"] == ["Intro"]

        parent = ctx.parents[0]
        assert parent["parent_chunk_id"] == "parent_1"
        assert "Parent context paragraph" in parent["text"]

    def test_children_carry_citation_locators(self, session):
        _seed(session)
        ctx = enrich_chunks(session, ["child_1"])
        c = ctx.children[0]
        assert c["page"] == 3
        assert "bounding_box" in c
        assert c["bounding_box"]["width"] == 30

    def test_missing_child_is_skipped(self, session):
        _seed(session)
        ctx = enrich_chunks(session, ["child_1", "does_not_exist"])
        assert len(ctx.children) == 1

    def test_empty_input(self, session):
        ctx = enrich_chunks(session, [])
        assert ctx.children == []
        assert ctx.parents == []

    def test_to_dict_round_trip(self, session):
        _seed(session)
        ctx = enrich_chunks(session, ["child_1"])
        d = ctx.to_dict()
        assert d["children"] == ctx.children
        assert d["parents"] == ctx.parents
