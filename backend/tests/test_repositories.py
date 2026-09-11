import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.chunker import Chunk
from app.models.chunk import DocumentChunk
from app.models.document import Base, Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository


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


def make_document(doc_id="doc1", status="UPLOADED"):
    return Document(
        doc_id=doc_id,
        user_id="u1",
        original_filename="x.txt",
        stored_filename="x.txt",
        extension=".txt",
        mime_type="text/plain",
        file_size=3,
        sha256=f"hash-{doc_id}",
        s3_key=f"u1/{doc_id}_x.txt",
        status=status,
    )


def make_chunk(chunk_id, index, parent_id=None):
    return Chunk(
        chunk_id=chunk_id,
        parent_chunk_id=parent_id,
        chunk_type="parent" if parent_id is None else "child",
        text="hello" * 20,
        chunk_index=index,
        doc_id="doc1",
        version_id="v1",
        user_id="u1",
        section_path=[],
        locations=[],
        metadata={},
    )


def add_chunks(session, n):
    chunks = [
        make_chunk(f"parent_{i}", i)
        for i in range(n)
    ]
    ChunkRepository(session).create_many(chunks)
    return chunks


def test_get_parent_returns_parent_row(session):
    dr = DocumentRepository(session)
    dr.create(make_document())

    parent = make_chunk("parent_0", 0)
    child = make_chunk("child_0", 1, parent_id="parent_0")
    ChunkRepository(session).create_many([parent, child])

    got = ChunkRepository(session).get_parent("parent_0")
    assert got is not None
    assert got.chunk_id == "parent_0"
    assert got.chunk_type == "parent"


def test_count_by_doc(session):
    dr = DocumentRepository(session)
    dr.create(make_document())
    add_chunks(session, 3)

    assert ChunkRepository(session).count_by_doc("doc1") == 3
    assert ChunkRepository(session).count_by_doc("missing") == 0


def test_delete_by_doc_only_removes_target(session):
    dr = DocumentRepository(session)
    dr.create(make_document("doc1"))
    dr.create(make_document("doc2"))

    add_chunks(session, 2)
    other = make_chunk("other_0", 0)
    other.doc_id = "doc2"
    ChunkRepository(session).create_many([other])

    deleted = ChunkRepository(session).delete_by_doc("doc1")

    assert deleted == 2
    assert ChunkRepository(session).count_by_doc("doc1") == 0
    assert ChunkRepository(session).count_by_doc("doc2") == 1


def test_document_get_by_id_and_update_status(session):
    dr = DocumentRepository(session)
    dr.create(make_document())

    assert dr.get_by_id("doc1") is not None
    assert dr.get_by_id("missing") is None

    updated = dr.update_status("doc1", "PROCESSING")
    assert updated.status == "PROCESSING"

    assert dr.update_status("missing", "PROCESSING") is None


def test_list_by_user_scopes_and_orders_newest_first(session):
    dr = DocumentRepository(session)
    for doc_id in ("doc_oldest", "doc_middle", "doc_newest"):
        dr.create(make_document(doc_id))

    dr.create(
        make_document("other_user")
    )
    other = dr.get_by_id("other_user")
    other.user_id = "u2"

    listed = [doc.doc_id for doc in dr.list_by_user("u1")]

    assert listed == ["doc_newest", "doc_middle", "doc_oldest"]
    assert dr.list_by_user("u2") == [other]