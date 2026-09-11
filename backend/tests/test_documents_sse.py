import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.documents as documents_module
from app.api.deps import get_current_user
from app.api.documents import _document_event_stream
from app.core.database import get_db
from app.main import app
from app.models.document import Base, Document
from app.models.user import User
from app.repositories.document_repository import DocumentRepository

USER_ID = "u1"
DOC_ID = "doc1"


@pytest.fixture
def db_session():
    # StaticPool: share a single SQLite in-memory connection across all
    # threads/connections so the stream's per-poll sessions and TestClient
    # requests all see the same tables.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    yield TestSession
    engine.dispose()


@pytest.fixture
def client(db_session, monkeypatch):
    def override_get_db():
        db = db_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = (
        lambda: User(user_id=USER_ID, email="a@b.com")
    )
    monkeypatch.setattr(
        documents_module,
        "SessionLocal",
        db_session,
    )
    client = TestClient(app, follow_redirects=False)
    yield client
    app.dependency_overrides.clear()


def make_document(doc_id=DOC_ID, status="UPLOADED", user_id=USER_ID):
    return Document(
        doc_id=doc_id,
        user_id=user_id,
        original_filename="x.txt",
        stored_filename="x.txt",
        extension=".txt",
        mime_type="text/plain",
        file_size=5,
        sha256=f"hash-{doc_id}",
        s3_key=f"{user_id}/{doc_id}_x.txt",
        status=status,
    )


def persist_document(db_session, doc_id=DOC_ID, status="UPLOADED", user_id=USER_ID):
    session = db_session()
    try:
        return DocumentRepository(session).create(
            make_document(doc_id=doc_id, status=status, user_id=user_id)
        )
    finally:
        session.close()


def set_status(db_session, status):
    session = db_session()
    try:
        DocumentRepository(session).update_status(DOC_ID, status)
    finally:
        session.close()


class FakeRequest:
    async def is_disconnected(self):
        return False


def parse_frames(body):
    frames = []
    for raw in body.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        event = "message"
        data = None
        for line in raw.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        frames.append((event, data))
    return frames


def test_event_stream_follows_status_transitions(db_session):
    persist_document(db_session, status="UPLOADED")

    async def drive():
        gen = _document_event_stream(
            DOC_ID,
            USER_ID,
            FakeRequest(),
            session_factory=db_session,
        )
        frames = []
        frames.append(await asyncio.wait_for(anext(gen), 10))
        for status in ("PARSED", "CHUNKED", "EMBEDDED", "COMPLETED"):
            set_status(db_session, status)
            frames.append(await asyncio.wait_for(anext(gen), 20))
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(gen), 20)
        return frames

    frames = parse_frames("\n".join(asyncio.run(drive())))

    assert frames[0][0] == "connected"
    assert json.loads(frames[0][1])["status"] == "UPLOADED"

    status_events = [
        (event, json.loads(data)["status"])
        for event, data in frames
        if event == "status"
    ]
    assert status_events == [
        ("status", "PARSED"),
        ("status", "CHUNKED"),
        ("status", "EMBEDDED"),
        ("status", "COMPLETED"),
    ]


def test_event_stream_ends_on_terminal_snapshot(db_session):
    persist_document(db_session, status="COMPLETED")

    async def drive():
        gen = _document_event_stream(
            DOC_ID,
            USER_ID,
            FakeRequest(),
            session_factory=db_session,
        )
        frames = [await asyncio.wait_for(anext(gen), 10)]
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(gen), 20)
        return frames

    frames = parse_frames("\n".join(asyncio.run(drive())))
    assert frames == [
        ("connected", json.dumps({"doc_id": DOC_ID, "filename": "x.txt", "status": "COMPLETED", "chunk_count": 0}))
    ]


def test_sse_streams_completed_document(client, db_session):
    persist_document(db_session, status="COMPLETED")

    with client.stream("GET", f"/api/v1/documents/{DOC_ID}/events") as response:
        assert response.status_code == 200
        body = response.read().decode()

    frames = parse_frames(body)
    assert len(frames) == 1
    event, data = frames[0]
    assert event == "connected"
    assert json.loads(data)["status"] == "COMPLETED"


def test_sse_returns_404_for_other_users_document(client, db_session):
    persist_document(db_session, status="UPLOADED", user_id="u2")

    response = client.get(f"/api/v1/documents/{DOC_ID}/events")
    assert response.status_code == 404


def test_list_documents_returns_own_docs_newest_first(client, db_session):
    persist_document(db_session, status="COMPLETED")
    persist_document(db_session, doc_id="doc2", status="COMPLETED")
    persist_document(db_session, doc_id="doc3", user_id="u2")

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    items = response.json()
    assert [item["doc_id"] for item in items] == ["doc2", DOC_ID]
    assert items[0]["filename"] == "x.txt"
    assert items[0]["status"] == "COMPLETED"
    assert items[0]["file_size"] == 5
    assert items[0]["chunk_count"] == 0


def test_list_documents_empty_when_no_docs(client):
    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == []