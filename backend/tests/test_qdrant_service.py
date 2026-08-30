from app.ingestion.chunker import Chunk
from app.ingestion.models import SourceLocation
from app.ingestion.payload import build_payload
from app.services.qdrant_service import QdrantService


def make_chunk(
    metadata=None,
    locations=None,
    section_path=None,
):
    return Chunk(
        chunk_id="child_x",
        parent_chunk_id="parent_y",
        chunk_type="child",
        text="hello chunk",
        chunk_index=4,
        doc_id="doc",
        version_id="v1",
        user_id="u1",
        section_path=section_path or [],
        locations=locations or [],
        metadata=metadata or {},
    )


def test_build_payload_promotes_filter_fields_to_top_level():
    chunk = make_chunk(
        metadata={
            "filename": "x.pdf",
            "parser": "odl",
            "mime_type": "application/pdf",
            "strategy": "pdf",
            "section_path": ["Intro"],
            "pages": [1, 2],
            "row_start": 3,
            "row_end": 9,
        },
        section_path=["Intro"],
    )
    payload = build_payload(chunk)

    assert payload["user_id"] == "u1"
    assert payload["doc_id"] == "doc"
    assert payload["version_id"] == "v1"
    assert payload["chunk_id"] == "child_x"
    assert payload["parent_chunk_id"] == "parent_y"
    assert payload["chunk_type"] == "child"
    assert payload["chunk_index"] == 4
    assert payload["text"] == "hello chunk"
    assert payload["filename"] == "x.pdf"
    assert payload["parser"] == "odl"
    assert payload["mime_type"] == "application/pdf"
    assert payload["strategy"] == "pdf"
    assert payload["section_path"] == ["Intro"]
    assert payload["pages"] == [1, 2]
    assert payload["row_start"] == 3
    assert payload["row_end"] == 9

    assert "metadata" not in payload


def test_build_payload_keeps_unknown_keys_out():
    chunk = make_chunk(
        metadata={
            "custom": "xyz",
            "mime_type": "text/plain",
        }
    )
    payload = build_payload(chunk)
    assert "custom" not in payload
    assert "metadata" not in payload
    assert payload["mime_type"] == "text/plain"


def test_build_payload_missing_filter_fields_stay_absent():
    chunk = make_chunk(metadata={})
    payload = build_payload(chunk)
    assert "pages" not in payload
    assert "row_start" not in payload
    assert payload["section_path"] == []


def test_build_payload_locations_serialized():
    chunk = make_chunk(
        locations=[
            SourceLocation(
                type="pdf_bbox",
                page=3,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        ]
    )
    payload = build_payload(chunk)
    assert payload["locations"]
    assert payload["locations"][0]["type"] == "pdf_bbox"
    assert payload["locations"][0]["page"] == 3


class FakeQdrantClient:
    def __init__(self):
        self.indexes = {}

    def get_collection(self, collection_name):
        return type(
            "_Info",
            (),
            {"payload_schema": dict(self.indexes)},
        )()

    def create_payload_index(
        self,
        collection_name,
        field_name,
        field_schema,
        wait=False,
        **kwargs,
    ):
        self.indexes[field_name] = field_schema


def make_service(client):
    svc = object.__new__(QdrantService)
    svc.collection_name = "probe"
    svc.client = client
    return svc


def test_create_payload_indexes_covers_every_upserted_field():
    from app.ingestion.payload import FILTER_FIELDS

    client = FakeQdrantClient()
    make_service(client)._create_payload_indexes()

    created = set(client.indexes)
    assert created == {name for name, _ in QdrantService.PAYLOAD_INDEXES} | {"text"}
    assert set(FILTER_FIELDS).issubset(created)
    assert {
        "user_id",
        "doc_id",
        "version_id",
        "parent_chunk_id",
        "chunk_type",
        "filename",
    }.issubset(created)


def test_create_payload_indexes_is_idempotent():
    client = FakeQdrantClient()
    svc = make_service(client)
    svc._create_payload_indexes()
    first = dict(client.indexes)
    svc._create_payload_indexes()
    assert client.indexes == first


def test_upsert_chunks_rejects_mismatched_counts():
    svc = make_service(FakeQdrantClient())
    try:
        svc.upsert_chunks([make_chunk()], [])
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")