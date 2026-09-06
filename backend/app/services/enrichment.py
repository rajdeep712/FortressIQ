"""Parent/child enrichment for retrieved chunks.

After reranking we hold a list of *child* chunks (the embedded leaf nodes).
For the LLM we want the richer parent context; for the UI we want each
child as a citation pointer that highlights the source region.

This module loads both from the relational ``document_chunks`` table:
  - children  -> cited in the answer, carry the bounding-box/page locators
  - parents   -> passed to the LLM as grounded context
"""

from __future__ import annotations

from typing import Any, Sequence

from app.models.chunk import DocumentChunk
from app.repositories.chunk_repository import ChunkRepository


class EnrichedContext:
    """Container for the children + parents we hand to the LLM and the UI."""

    def __init__(
        self,
        children: list[dict],
        parents: list[dict],
    ):
        self.children = children
        self.parents = parents

    def to_dict(self) -> dict:
        return {
            "children": self.children,
            "parents": self.parents,
        }


def _location_box(locations: Any) -> dict[str, Any]:
    """Return a frontend-style bounding box from a chunk's locations.

    Prefers the first location carrying a page + bbox; otherwise returns
    an empty box (the UI then falls back to a general page highlight).
    """
    if not locations:
        return {}

    for loc in locations:
        if not isinstance(loc, dict):
            continue
        bbox = loc.get("bbox")
        if isinstance(bbox, dict) and bbox:
            return {
                "top": bbox.get("top", 0),
                "left": bbox.get("left", 0),
                "width": bbox.get("width", 100),
                "height": bbox.get("height", 100),
            }
    return {}


def _page_of(locations: Any, fallback: int = 1) -> int:
    if not locations:
        return fallback
    for loc in locations:
        if isinstance(loc, dict) and loc.get("page"):
            return int(loc["page"])
    return fallback


def enrich_chunks(
    db: Any,
    child_chunk_ids: Sequence[str],
) -> EnrichedContext:
    """Given reranked child chunk ids, load children + their parents.

    ``db`` is a SQLAlchemy Session (used to build the ChunkRepository).
    Child ids that are not found (e.g. a Qdrant hit whose relational row
    was purged) are skipped gracefully.
    """
    repo = ChunkRepository(db)

    children: list[dict] = []
    parents: list[dict] = []
    parent_ids: set[str] = set()

    # First pass: resolve children, collect parent references.
    resolved: list[DocumentChunk] = []
    for child_id in child_chunk_ids:
        row = repo.get_chunk(child_id)
        if row is None:
            continue
        resolved.append(row)
        if row.parent_chunk_id:
            parent_ids.add(row.parent_chunk_id)

    # Second pass: load parent context in one batch.
    parent_map: dict[str, DocumentChunk] = {}
    for pid in parent_ids:
        parent = repo.get_parent(pid)
        if parent is not None:
            parent_map[pid] = parent

    for row in resolved:
        children.append(_child_to_dict(row))
        parent = parent_map.get(row.parent_chunk_id or "")
        parent_dict = _parent_to_dict(parent) if parent else None
        if parent_dict:
            parents.append(parent_dict)

    return EnrichedContext(children=children, parents=parents)


def _child_to_dict(row: DocumentChunk) -> dict:
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "text": row.text,
        "filename": (row.metadata_json or {}).get("filename"),
        "section_path": row.section_path or [],
        "locations": row.locations or [],
        "page": _page_of(row.locations),
        "bounding_box": _location_box(row.locations),
    }


def _parent_to_dict(row: DocumentChunk) -> dict:
    return {
        "parent_chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "text": row.text,
        "section_path": row.section_path or [],
    }
