import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.chunker import Chunk
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.models.document import Base, Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services import ingestion_service as ingestion_module
from app.services.ingestion_service import IngestionService


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    yield Session
    engine.dispose()


def make_document(status="UPLOADED"):
    return Document(
        doc_id="doc1",
        user_id="u1",
        original_filename="x.txt",
        stored_filename="x.txt",
        extension=".txt",
        mime_type="text/plain",
        file_size=5,
        sha256="hash",
        s3_key="u1/doc1_x.txt",
        status=status,
    )


def persist_document(session_factory, status="UPLOADED"):
    session = session_factory()
    try:
        return DocumentRepository(session).create(
            make_document(status)
        )
    finally:
        session.close()


class FakeS3:
    def __init__(self):
        self.content = b"line one\nline two\n"
        self.got = None

    def get_object(self, s3_key):
        self.got = s3_key
        return self.content


class FakeEmbedding:
    def __init__(self):
        self.calls = []
        self.last_provider = "sentence-transformer"

    def embed(self, texts):
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.deleted = []
        self.upserted = []

    def delete_chunks_by_doc(self, doc_id):
        self.deleted.append(doc_id)

    def upsert_chunks(self, chunks, vectors):
        self.upserted.append((len(chunks), len(vectors)))


class FakeParser:
    async def parse(
        self,
        file_path,
        doc_id,
        version_id,
        user_id,
        filename,
        mime_type,
    ):
        self.kwargs = {
            "doc_id": doc_id,
            "version_id": version_id,
            "user_id": user_id,
            "filename": filename,
            "mime_type": mime_type,
        }
        elements = [
            ParsedElement(
                element_id=f"e{i}",
                element_type="text",
                text=f"This is line {i} of the fake document.",
                order=i,
                locations=[
                    SourceLocation(
                        type="text_offset",
                        line_start=i + 1,
                        line_end=i + 1,
                    )
                ],
            )
            for i in range(5)
        ]
        return ParsedDocument(
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
            parser_name="fake",
            parser_version="1.0",
            elements=elements,
            metadata={"char_count": 200},
        )


class FailingParser:
    async def parse(self, **kwargs):
        raise RuntimeError("boom")


def make_service(session_factory, monkeypatch):
    svc = object.__new__(IngestionService)
    svc.s3 = FakeS3()
    svc.embedding = FakeEmbedding()
    svc.qdrant = FakeQdrant()
    monkeypatch.setattr(
        ingestion_module,
        "SessionLocal",
        session_factory,
    )
    monkeypatch.setattr(
        ingestion_module,
        "get_parser",
        lambda extension: FakeParser(),
    )
    return svc


def test_ingest_completes_and_persists(session_factory, monkeypatch):
    persist_document(session_factory)

    svc = make_service(session_factory, monkeypatch)
    result = svc.ingest_document("doc1", "u1")

    assert result["skipped"] is False
    assert result["chunks"] >= 1
    assert result["embedded_chunks"] >= 1
    assert svc.s3.got == "u1/doc1_x.txt"
    assert svc.qdrant.deleted == ["doc1"]
    assert svc.qdrant.upserted == [
        (
            result["embedded_chunks"],
            result["embedded_chunks"],
        )
    ]

    session = session_factory()
    try:
        document = DocumentRepository(session).get_by_id("doc1")
        assert document.status == "COMPLETED"
        assert (
            ChunkRepository(session).count_by_doc("doc1")
            == result["chunks"]
        )
    finally:
        session.close()


def test_ingest_skips_completed_documents(
    session_factory,
    monkeypatch,
):
    persist_document(session_factory, status="COMPLETED")

    svc = make_service(session_factory, monkeypatch)
    result = svc.ingest_document("doc1", "u1")

    assert result["skipped"] is True
    assert svc.s3.got is None
    assert svc.qdrant.deleted == []
    assert svc.qdrant.upserted == []


def test_retry_replaces_prior_artifacts(session_factory, monkeypatch):
    doc = persist_document(session_factory, status="FAILED")

    session = session_factory()
    try:
        old_parent = Chunk(
            chunk_id="old_parent",
            parent_chunk_id=None,
            chunk_type="parent",
            text="old",
            chunk_index=0,
            doc_id="doc1",
            version_id="v_old",
            user_id="u1",
            section_path=[],
            locations=[],
            metadata={},
        )
        ChunkRepository(session).create_many([old_parent])
        assert ChunkRepository(session).count_by_doc("doc1") == 1
    finally:
        session.close()

    svc = make_service(session_factory, monkeypatch)
    result = svc.ingest_document("doc1", "u1")

    assert result["skipped"] is False
    assert svc.qdrant.deleted == ["doc1"]

    session = session_factory()
    try:
        repository = ChunkRepository(session)
        assert repository.count_by_doc("doc1") == result["chunks"]
        assert repository.get_chunk("old_parent") is None
    finally:
        session.close()


def test_ingest_unknown_document_raises(
    session_factory,
    monkeypatch,
):
    svc = make_service(session_factory, monkeypatch)
    with pytest.raises(ValueError):
        svc.ingest_document("missing", "u1")


def test_parse_failure_marks_document_failed(
    session_factory,
    monkeypatch,
):
    persist_document(session_factory)

    svc = make_service(session_factory, monkeypatch)
    monkeypatch.setattr(
        ingestion_module,
        "get_parser",
        lambda extension: FailingParser(),
    )

    with pytest.raises(RuntimeError):
        svc.ingest_document("doc1", "u1")

    session = session_factory()
    try:
        assert (
            DocumentRepository(session)
            .get_by_id("doc1")
            .status
            == "FAILED"
        )
    finally:
        session.close()


def test_ingest_tags_embedder_on_children(
    session_factory,
    monkeypatch,
):
    from app.models.chunk import DocumentChunk

    persist_document(session_factory)

    svc = make_service(session_factory, monkeypatch)
    svc.ingest_document("doc1", "u1")

    session = session_factory()
    try:
        children = (
            session.query(DocumentChunk)
            .filter_by(
                doc_id="doc1",
                chunk_type="child",
            )
            .all()
        )
        assert children
        for child in children:
            assert (
                (child.metadata_json or {})
                .get("embedder")
                == "sentence-transformer"
            )
    finally:
        session.close()