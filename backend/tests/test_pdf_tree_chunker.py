"""Tests for the structure-driven PDF tree chunker.

These cover the OpenDocumentLoader-shaped hierarchy the user's real
PDFs expose: numbered project items (root ``list item``s) whose kids
hold nested sub-lists (a/b/c, papers i/ii). Each major item must become
one parent; each ODL kid must become one child chunk (1:1), even when
small — no character-window packing.
"""

import pytest

from app.ingestion.chunker.base import (
    Chunk,
)
from app.ingestion.chunker.factory import (
    get_chunker_for,
)
from app.ingestion.chunker.pdf_chunker import (
    PdfChunker,
)
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)


def pdf_el(
    value,
    text,
    etype="paragraph",
    parent=None,
    section_path=None,
    bbox=None,
    metadata=None,
    page=1,
):
    locations = []
    if bbox is not None:
        locations.append(
            SourceLocation(
                type="pdf_bbox",
                page=page,
                bbox={
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
            )
        )
    return ParsedElement(
        element_id=value,
        element_type=etype,
        text=text,
        order=0,
        section_path=list(section_path or []),
        parent_element_id=parent,
        locations=locations,
        metadata=dict(metadata or {}),
    )


def document(elements):
    return ParsedDocument(
        doc_id="doc",
        version_id="v1",
        user_id="u1",
        filename="projects.pdf",
        mime_type="application/pdf",
        parser_name="open_document_loader_pdf",
        parser_version="1.0",
        elements=elements,
        metadata={},
    )


def parents_children(chunks):
    parents = [
        c for c in chunks
        if c.chunk_type == "parent"
    ]
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    return parents, children


def assert_invariants(chunks):
    parents, children = parents_children(chunks)

    parent_ids = {p.chunk_id for p in parents}

    assert len(parent_ids) == len(parents)

    indexes = [c.chunk_index for c in chunks]
    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)

    for child in children:
        assert child.parent_chunk_id in parent_ids

    assert children, "expect at least one child"


def test_mimics_user_project_list_tree():
    root = pdf_el(
        "p2",
        "2. Graph Neural Networks (GNNs) have achieved remarkable "
        "success across various graph analysis tasks.",
        etype="list_item",
    )
    kids = [
        pdf_el(
            "a1",
            "a. The project may explore possibility of adversarial "
            "generation of nodes",
            etype="list_item",
            parent="p2",
        ),
        pdf_el(
            "b1",
            "b. May explore the possibility of graph clustering in "
            "GraphUAT framework",
            etype="list_item",
            parent="p2",
        ),
        pdf_el(
            "c1",
            "c. May explore applications in imbalanced node "
            "classification in specific domains like medical science",
            etype="list_item",
            parent="p2",
        ),
    ]
    doc = document([root] + kids)

    chunks = PdfChunker().chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)

    assert len(parents) == 1
    assert len(children) == 3

    assert parents[0].text.startswith(
        "2. Graph Neural Networks"
    )
    assert parents[0].metadata["root_element_id"] == "p2"

    assert [c.text for c in children] == [
        k.text for k in kids
    ]
    assert all(
        c.parent_chunk_id == parents[0].chunk_id
        for c in children
    )


def test_heading_plus_paragraph_is_one_parent_one_child():
    heading = pdf_el(
        "h1",
        "Introduction",
        etype="h1",
        section_path=["Introduction"],
    )
    body = pdf_el(
        "b1",
        "Some body text under the heading.",
        parent="h1",
        section_path=["Introduction"],
    )
    doc = document([heading, body])

    parents, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].text == (
        "Introduction\n\nSome body text "
        "under the heading."
    )
    assert children[0].text == (
        "Some body text under the heading."
    )


def test_flat_root_leaves_merge_into_one_run_parent():
    doc = document(
        [
            pdf_el("l1", "line one"),
            pdf_el("l2", "line two"),
            pdf_el("l3", "line three"),
        ]
    )
    parents, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 1
    # 1:1 children, unlike SectionChunker window packing.
    assert [c.text for c in children] == [
        "line one",
        "line two",
        "line three",
    ]
    assert parents[0].metadata["kind"] == "run"


def test_numbered_flat_items_get_own_parent_regardless_of_kids():
    doc = document(
        [
            pdf_el("n1", "1. First topic"),
            pdf_el("n2", "2. Second topic"),
        ]
    )
    parents, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 2
    assert len(children) == 2
    assert {p.text for p in parents} == {
        "1. First topic",
        "2. Second topic",
    }
    for parent, child in zip(parents, children):
        assert child.parent_chunk_id == parent.chunk_id
        assert parent.text == child.text


def test_top_level_unnumbered_list_item_is_major():
    root = pdf_el(
        "s1",
        "Technical Skills",
        etype="list_item",
    )
    kid = pdf_el(
        "s2",
        "Python, PyTorch, LangGraph",
        etype="list_item",
        parent="s1",
    )
    doc = document([root, kid])
    parents, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 1
    assert parents[0].text == (
        "Technical Skills\n\n"
        "Python, PyTorch, LangGraph"
    )
    assert len(children) == 1


def test_tiny_kid_folds_into_next_sibling():
    root = pdf_el(
        "p1",
        "1. Model",
        etype="list_item",
    )
    tiny = pdf_el(
        "t1",
        "short",
        etype="list_item",
        parent="p1",
    )
    normal = pdf_el(
        "n1",
        "A substantially long listing item text well over the "
        "sixty-character merge threshold.",
        etype="list_item",
        parent="p1",
    )
    doc = document([root, tiny, normal])

    _, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(children) == 1
    assert children[0].text == (
        "short\n\n"
        "A substantially long listing item text well over the "
        "sixty-character merge threshold."
    )


def test_trailing_tiny_kid_folds_into_previous():
    root = pdf_el(
        "p1",
        "1. Model",
        etype="list_item",
    )
    normal = pdf_el(
        "n1",
        "A substantially long listing item text well over the "
        "sixty-character merge threshold.",
        etype="list_item",
        parent="p1",
    )
    tail = pdf_el(
        "t1",
        "zzz",
        etype="list_item",
        parent="p1",
    )
    doc = document([root, normal, tail])

    _, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(children) == 1
    assert children[0].text == (
        "A substantially long listing item text well over the "
        "sixty-character merge threshold.\n\nzzz"
    )


def test_all_tiny_kids_kept_separate():
    root = pdf_el(
        "p1",
        "1. Model",
        etype="list_item",
    )
    kids = [
        pdf_el("k1", "a", etype="list_item", parent="p1"),
        pdf_el("k2", "b", etype="list_item", parent="p1"),
        pdf_el("k3", "c", etype="list_item", parent="p1"),
    ]
    doc = document([root] + kids)

    _, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert [c.text for c in children] == ["a", "b", "c"]


def test_empty_text_kid_skipped():
    root = pdf_el(
        "p1",
        "1. Model",
        etype="list_item",
    )
    empty = pdf_el(
        "e1",
        "",
        etype="list_item",
        parent="p1",
    )
    real = pdf_el(
        "r1",
        "The real content of the item.",
        etype="list_item",
        parent="p1",
    )
    doc = document([root, empty, real])

    _, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert [c.text for c in children] == [
        "The real content of the item."
    ]


def test_oversized_major_item_stays_one_parent():
    root = pdf_el(
        "big",
        "2. " + "x" * 8000,
        etype="list_item",
    )
    kid = pdf_el(
        "k",
        "a. " + "y" * 500,
        etype="list_item",
        parent="big",
    )
    doc = document([root, kid])

    parents, _ = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 1
    assert len(parents[0].text) > 7500
    assert "part" not in parents[0].metadata


def test_bbox_sibling_sort_is_opt_in():
    root = pdf_el(
        "p1",
        "1. Model",
        etype="list_item",
    )
    expected_order = [
        pdict
        for pdict in (
            {"bottom": "Z bottom", "bbox": (100, 420, 400, 430)},
            {"bottom": "Y middle", "bbox": (100, 320, 400, 330)},
            {"bottom": "X top", "bbox": (100, 220, 400, 230)},
        )
    ]
    kids = [
        pdf_el(
            f"k{i}",
            expected_order[i]["bottom"] + " with enough text",
            etype="list_item",
            parent="p1",
            bbox=expected_order[i]["bbox"],
        )
        for i in range(3)
    ]
    doc = document([root] + kids)

    _, children = parents_children(
        PdfChunker().chunk(doc)
    )
    # Default: ODL insertion order is preserved (logical order).
    assert [c.text for c in children] == [
        k.text for k in kids
    ]

    sorted_chunker = PdfChunker(
        sort_siblings_by_bbox=True,
    )
    _, sorted_children = parents_children(
        sorted_chunker.chunk(doc)
    )
    # Reading order: ascending y.
    assert sorted_children[0].text == kids[2].text
    assert sorted_children[1].text == kids[1].text
    assert sorted_children[2].text == kids[0].text


def test_structured_false_restores_window_packing():
    doc = document(
        [
            pdf_el("l1", "line one"),
            pdf_el("l2", "line two"),
            pdf_el("l3", "line three"),
        ]
    )
    chunker = PdfChunker(structured=False)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert len(parents) == 1
    # SectionChunker packs short content into a single child window.
    assert len(children) == 1
    assert children[0].text == "line one\n\nline two\n\nline three"


def test_factory_returns_structured_pdf_chunker_by_default():
    chunker = get_chunker_for(document([]))
    assert isinstance(chunker, PdfChunker)
    assert chunker.structured is True


def test_no_elements_produces_no_chunks():
    chunks = PdfChunker().chunk(document([]))
    assert chunks == []


def test_nested_descendants_flatten_into_children():
    # Sub-sub nesting (item -> list -> list item) still yields 1:1
    # children on the owning major parent.
    root = pdf_el(
        "p1",
        "1. Project",
        etype="list_item",
    )
    wrapper = pdf_el(
        "member",
        "Papers",
        etype="list_item",
        parent="p1",
    )
    sub_a = pdf_el(
        "sub1",
        "i. Something about a method",
        etype="list_item",
        parent="member",
    )
    sub_b = pdf_el(
        "sub2",
        "ii. Something about another method",
        etype="list_item",
        parent="member",
    )
    doc = document([root, wrapper, sub_a, sub_b])

    parents, children = parents_children(
        PdfChunker().chunk(doc)
    )
    assert len(parents) == 1
    assert [c.text for c in children] == [
        "Papers",
        "i. Something about a method",
        "ii. Something about another method",
    ]
    assert all(
        isinstance(c, Chunk) for c in children
    )