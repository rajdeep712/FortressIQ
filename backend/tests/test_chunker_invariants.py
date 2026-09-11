from uuid import uuid4

import pytest

from app.ingestion.chunker import (
    chunk_document,
)
from app.ingestion.chunker.csv_chunker import CsvChunker
from app.ingestion.chunker.tokenizer import estimate_tokens
from app.ingestion.chunker.xlsx_chunker import XlsxChunker
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)

CHILD_TOKEN_MAX = 250
PARENT_TOKEN_MAX = 1800
CHILD_MIN = 4
CHILD_MAX = 8
OVERLAP_MIN = 200
OVERLAP_MAX = 300


def element(text, element_type="paragraph", section_path=None, metadata=None):
    return ParsedElement(
        element_id=f"el_{uuid4().hex}",
        element_type=element_type,
        text=text,
        order=0,
        section_path=section_path or [],
        locations=[],
        metadata=metadata or {},
    )


def document(elements, mime_type, filename):
    return ParsedDocument(
        doc_id="doc",
        version_id="v1",
        user_id="u1",
        filename=filename,
        mime_type=mime_type,
        parser_name="x",
        parser_version="1.0",
        elements=elements,
        metadata={},
    )


def parents_children(chunks):
    return (
        [c for c in chunks if c.chunk_type == "parent"],
        [c for c in chunks if c.chunk_type == "child"],
    )


def assert_token_caps(chunks):
    parents, children = parents_children(chunks)
    assert parents
    assert children
    for p in parents:
        assert estimate_tokens(p.text) <= PARENT_TOKEN_MAX, (
            f"parent over max tokens: {estimate_tokens(p.text)}"
        )
    for c in children:
        assert estimate_tokens(c.text) <= CHILD_TOKEN_MAX, (
            f"child over max tokens: {estimate_tokens(c.text)}"
        )


def sibling_overlaps(chunks):
    parents, children = parents_children(chunks)
    by_parent = {p.chunk_id: p for p in parents}
    overlaps = []
    for parent_id in by_parent:
        kids = sorted(
            (c for c in children if c.parent_chunk_id == parent_id),
            key=lambda c: c.chunk_index,
        )
        for a, b in zip(kids, kids[1:]):
            shared = 0
            for n in range(1, len(a.text)):
                if b.text.startswith(a.text[-n:]):
                    shared = n
            overlaps.append(shared)
    return overlaps


def test_child_token_cap_holds_across_standard_doc_types():
    cases = [
        (
            [
                element(
                    "A moderately long heading paragraph with several "
                    "distinct words in it. " * 2
                ),
                element(
                    "Another body paragraph that keeps the section long "
                    "enough to require window packing. " * 3
                ),
            ]
            * 10,
            "text/markdown",
            "invariant.md",
        ),
        (
            [
                element(
                    "Paragraph with several words that is substantial. " * 3
                )
                for _ in range(30)
            ],
            "text/plain",
            "invariant.txt",
        ),
        (
            [
                ParsedElement(
                    element_id=f"p_{i}",
                    element_type="major_heading",
                    text=f"Section {i}",
                    order=i,
                    section_path=[f"Section {i}"],
                )
                for i in range(8)
            ]
            + [
                ParsedElement(
                    element_id=f"b_{i}",
                    element_type="paragraph",
                    text="Some body words that add up across the section. " * 4,
                    order=100 + i,
                    section_path=[f"Section {i % 8}"],
                )
                for i in range(8)
            ],
            "text/html",
            "invariant.html",
        ),
        (
            [
                ParsedElement(
                    element_id=f"el_{i}",
                    element_type="paragraph",
                    text="Wide text that becomes several windows. " * 5,
                    order=i,
                    section_path=[],
                    locations=[
                        SourceLocation(type="slide_bbox", slide=1)
                    ],
                    metadata={"slide_number": 1},
                )
                for i in range(24)
            ],
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "invariant.pptx",
        ),
    ]

    for elements, mime, filename in cases:
        doc = document(elements, mime, filename)
        chunks = chunk_document(doc)
        assert_token_caps(chunks), filename


def test_overlap_stays_within_min_max_on_content_rich_parent():
    paras = [
        element(f"Sentence with some distinctive words {i} here to add length. " * 3)
        for i in range(36)
    ]
    chunks = chunk_document(
        document(paras, "text/markdown", "overlap.md")
    )
    overlaps = sibling_overlaps(chunks)
    assert overlaps, "expected multiple children"
    assert max(overlaps) <= OVERLAP_MAX, (
        f"overlap exceeds max: {max(overlaps)}"
    )
    assert min(overlaps) >= OVERLAP_MIN, (
        f"overlap under min: {min(overlaps)}"
    )


def test_parent_token_budget_splits_huge_markdown_section():
    paras = [
        element(f"Sentence with some distinctive words {i} here to add length. " * 8)
        for i in range(120)
    ]
    chunks = chunk_document(
        document(paras, "text/markdown", "huge.md")
    )
    parents, children = parents_children(chunks)
    assert len(parents) >= 2
    for p in parents:
        assert estimate_tokens(p.text) <= PARENT_TOKEN_MAX
    # no content silently dropped by truncation
    assert all("\u2026[truncated]" not in p.text for p in parents)


def test_children_per_parent_within_soft_range_when_filled():
    paras = [
        element(f"Sentence with some distinctive words {i} here to add length. " * 4)
        for i in range(20)
    ]
    chunks = chunk_document(
        document(paras, "text/markdown", "medium.md")
    )
    parents, children = parents_children(chunks)
    assert len(parents) == 1
    counts = [
        sum(1 for c in children if c.parent_chunk_id == p.chunk_id)
        for p in parents
    ]
    # soft 4-8 target with headroom for a filled parent
    assert min(counts) >= 1
    assert max(counts) <= 12


def test_dp_coalesces_content_blocks_into_banded_parents():
    paragraphs = [
        element(f"Sentence block {i} with a useful payload. " * 3)
        for i in range(60)
    ]
    chunks = chunk_document(
        document(paragraphs, "text/plain", "dp.txt")
    )
    parents, children = parents_children(chunks)
    counts = [
        sum(1 for c in children if c.parent_chunk_id == p.chunk_id)
        for p in sorted(parents, key=lambda p: p.chunk_index)
    ]
    # DP coalesces sibling content blocks into the minimum number of
    # parents; every non-final parent must hold children_min..children_max
    # children, and only the final parent may fall short.
    assert len(counts) >= 2, "content long enough for several parents"
    for i, count in enumerate(counts[:-1]):
        assert CHILD_MIN <= count <= CHILD_MAX, (
            f"non-final parent {i} has {count} children"
        )
    assert 1 <= counts[-1] <= CHILD_MAX, (
        f"final parent has {counts[-1]} children"
    )
    assert_token_caps(chunks)


def test_dp_merges_many_tiny_runs_into_few_banded_parents():
    paragraphs = [
        element(f"tiny run {i} of loose words. " * 3)
        for i in range(80)
    ]
    chunks = chunk_document(
        document(paragraphs, "text/plain", "many.txt")
    )
    parents, children = parents_children(chunks)
    assert len(parents) >= 2, "many runs need more than one parent"
    counts = [
        sum(1 for c in children if c.parent_chunk_id == p.chunk_id)
        for p in sorted(parents, key=lambda p: p.chunk_index)
    ]
    for count in counts:
        assert CHILD_MIN <= count <= CHILD_MAX, (
            f"parent has {count} children (wants min {CHILD_MIN}"
            f", max {CHILD_MAX})"
        )
    assert_token_caps(chunks)


def test_merged_parent_does_not_brew_empty_or_wafer_children():
    paragraphs = [
        element(f"Timely vivid paragraph {i} with real content. " * 4)
        for i in range(40)
    ]
    chunks = chunk_document(
        document(paragraphs, "text/html", "body.html")
    )
    parents, children = parents_children(chunks)
    assert parents
    by_parent = {
        p.chunk_id: [
            c for c in children if c.parent_chunk_id == p.chunk_id
        ]
        for p in parents
    }
    for p in parents:
        kids = by_parent[p.chunk_id]
        assert kids, f"parent {p.chunk_id} has no children"
        assert all(k.text.strip() for k in kids), (
            f"parent {p.chunk_id} owns an empty-text child"
        )
    assert_token_caps(chunks)


def test_csv_children_preserve_legacy_char_limits():
    doc = document(
        [
            ParsedElement(
                element_id=f"r{i}",
                element_type="row",
                text=f"c1,c2,c3",
                order=i,
                section_path=[],
                locations=[
                    SourceLocation(type="csv_row", row_start=i, row_end=i)
                ],
            )
            for i in range(5)
        ],
        "text/csv",
        "invariant.csv",
    )
    chunks = CsvChunker().chunk(doc)
    _, children = parents_children(chunks)
    assert children
    assert all(len(c.text) <= 2000 for c in children)
    assert all(len(c.text) <= 20000 for c in chunks)


def test_xlsx_children_preserve_legacy_char_limits():
    doc = document(
        [
            ParsedElement(
                element_id=f"c{i}",
                element_type="cell",
                text="header",
                order=i,
                section_path=[],
            )
            for i in range(5)
        ],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "invariant.xlsx",
    )
    chunks = XlsxChunker().chunk(doc)
    _, children = parents_children(chunks)
    assert children
    assert all(len(c.text) <= 2000 for c in children)