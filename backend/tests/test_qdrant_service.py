import uuid

import httpx
from qdrant_client import models

from app.ingestion.chunker import Chunk
from app.ingestion.models import SourceLocation
from app.ingestion.payload import build_payload
from app.services.qdrant_service import (
    QdrantService,
    qdrant_point_id,
)


def make_chunk(
    metadata=None,
    locations=None,
    section_path=None,
    chunk_id="child_x",
):
    return Chunk(
        chunk_id=chunk_id,
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


def test_delete_chunks_by_doc_builds_doc_filter():
    client = FakeQdrantClient()
    calls = []
    client.delete = lambda **kwargs: calls.append(kwargs)

    make_service(client).delete_chunks_by_doc("doc_abc123")

    kwargs = calls[0]
    assert kwargs["collection_name"] == "probe"
    assert kwargs["wait"] is True

    selector = kwargs["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    must = selector.filter.must
    assert len(must) == 1
    assert isinstance(must[0], models.FieldCondition)
    assert must[0].key == "doc_id"
    assert must[0].match.value == "doc_abc123"


def make_batched_service(client, batch_size=3):
    svc = make_service(client)
    svc.upsert_batch_size = batch_size
    svc.upsert_max_retries = 2
    svc.upsert_retry_delay_seconds = 0
    return svc


def test_qdrant_point_id_from_hex_suffix_is_dashed_uuid():
    chunk_id = "child_6be9d94841814fa48177c1a06ebfc927"
    point_id = qdrant_point_id(chunk_id)

    assert point_id == "6be9d948-4181-4fa4-8177-c1a06ebfc927"
    assert uuid.UUID(point_id)

    # Deterministic for the same chunk id (idempotent re-upserts).
    assert qdrant_point_id(chunk_id) == point_id
    assert qdrant_point_id("child_9e3a9d3e36d04b24b3e6f5d6125e9bcd") != point_id


def test_qdrant_point_id_fallback_for_non_hex_chunk_ids():
    point_id = qdrant_point_id("chunk-abc-custom")
    assert uuid.UUID(point_id)
    assert qdrant_point_id("chunk-abc-custom") == point_id


def test_upsert_chunks_uses_qdrant_valid_point_ids():
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return None

    chunk = make_chunk(
        chunk_id="child_6be9d94841814fa48177c1a06ebfc927",
    )
    client = FakeQdrantClient()
    client.upsert = record

    svc = make_batched_service(client)
    svc.upsert_chunks(
        [chunk],
        [[0.0] * 768],
    )

    point = calls[0]["points"][0]
    # Qdrant serializes UUID point ids as dashed strings.
    assert point.id == "6be9d948-4181-4fa4-8177-c1a06ebfc927"
    # The payload keeps the original human-readable chunk id.
    assert point.payload["chunk_id"] == "child_6be9d94841814fa48177c1a06ebfc927"
    assert point.payload["parent_chunk_id"] == "parent_y"


def test_upsert_chunks_splits_into_batches():
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return None

    client = FakeQdrantClient()
    client.upsert = record

    svc = make_batched_service(client)
    chunks = [make_chunk() for _ in range(7)]
    vectors = [[0.0] * 768 for _ in range(7)]

    results = svc.upsert_chunks(chunks, vectors)

    assert [len(call["points"]) for call in calls] == [3, 3, 1]
    assert all(call["wait"] is True for call in calls)
    assert all(call["collection_name"] == "probe" for call in calls)

    # Qdrant serializes UUID point ids as strings; every chunk lands in
    # exactly one batch, ids intact.
    sent = [
        str(point.id)
        for call in calls
        for point in call["points"]
    ]
    expected = sorted(
        qdrant_point_id(c.chunk_id)
        for c in chunks
    )
    assert sorted(sent) == expected
    assert len(results) == 3


def test_upsert_chunks_retries_then_succeeds():
    attempts = {"count": 0}
    calls = []

    def flaky(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("read timeout")
        calls.append(kwargs["points"])
        return None

    client = FakeQdrantClient()
    client.upsert = flaky

    svc = make_batched_service(client)
    svc.upsert_chunks(
        [make_chunk()],
        [[0.0] * 768],
    )

    assert attempts["count"] == 2
    assert len(calls) == 1
    assert len(calls[0]) == 1


def test_upsert_chunks_gives_up_after_max_retries():
    attempts = {"count": 0}

    def failing(**kwargs):
        attempts["count"] += 1
        raise httpx.ReadTimeout("read timeout")

    client = FakeQdrantClient()
    client.upsert = failing

    svc = make_batched_service(client)
    svc.upsert_max_retries = 1

    try:
        svc.upsert_chunks(
            [make_chunk()],
            [[0.0] * 768],
        )
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError("expected httpx.ReadTimeout")

    assert attempts["count"] == 2


def test_upsert_chunks_does_not_retry_non_transient_errors():
    attempts = {"count": 0}

    def rejecting(**kwargs):
        attempts["count"] += 1
        raise ValueError("permanent error")

    client = FakeQdrantClient()
    client.upsert = rejecting

    svc = make_batched_service(client)
    svc.upsert_max_retries = 2

    try:
        svc.upsert_chunks(
            [make_chunk()],
            [[0.0] * 768],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    assert attempts["count"] == 1