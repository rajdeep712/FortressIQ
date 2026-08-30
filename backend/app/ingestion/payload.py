"""
Chunk -> vector-store payload builder (pure Python, no Qdrant deps).

Qdrant payload filtering only works on top-level scalar fields, so the
locator/strategy fields stored inside ``Chunk.metadata`` are promoted
to top-level payload keys here. The raw locations are kept as well so
the exact document position is available for provenance/citation.
"""


# Fields promoted from Chunk.metadata to top-level payload keys so the
# vector store can index and filter on them.
FILTER_FIELDS = (
    "section_path",
    "mime_type",
    "strategy",
    "root",
    "sheet",
    "sheets",
    "pages",
    "slides",
    "row_start",
    "row_end",
    "line_start",
    "line_end",
    "json_paths",
    "dom_paths",
    "cell_ranges",
)


def build_payload(chunk) -> dict:
    """Vector-store point payload for a chunk: tenant/identity fields,
    the filterable fields (promoted from metadata, so Qdrant can index
    and filter them) and the raw source locations."""
    meta = chunk.metadata or {}

    payload = {
        # Authorization / tenant filtering
        "user_id": chunk.user_id,
        "doc_id": chunk.doc_id,
        "version_id": chunk.version_id,
        "chunk_id": chunk.chunk_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "chunk_type": chunk.chunk_type,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "filename": meta.get("filename"),
        "parser": meta.get("parser"),
        "section_path": list(chunk.section_path),
        "locations": [
            location.__dict__
            for location in chunk.locations
        ],
    }

    for field in FILTER_FIELDS:
        if field in meta:
            payload[field] = meta[field]

    return payload