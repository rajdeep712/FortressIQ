from collections import Counter
from uuid import uuid4

import pytest

from app.ingestion.chunker import (
    Chunk,
    chunk_document,
    get_chunker_for,
)
from app.ingestion.chunker.csv_chunker import CsvChunker
from app.ingestion.chunker.json_chunker import JsonChunker
from app.ingestion.chunker.markdown_chunker import MarkdownChunker
from app.ingestion.chunker.pptx_chunker import PptxChunker
from app.ingestion.chunker.section_chunker import SectionChunker
from app.ingestion.chunker.tokenizer import estimate_tokens
from app.ingestion.chunker.txt_chunker import TxtChunker
from app.ingestion.chunker.xlsx_chunker import XlsxChunker
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)


def element(
    text,
    element_type="paragraph",
    section_path=None,
    locations=None,
    metadata=None,
    value=None,
    parent=None,
):
    return ParsedElement(
        element_id=value or f"el_{uuid4().hex}",
        element_type=element_type,
        text=text,
        order=0,
        section_path=section_path or [],
        parent_element_id=parent,
        locations=locations or [],
        metadata=metadata or {},
    )


def text_offset(start, end):
    return SourceLocation(
        type="text_offset",
        line_start=start,
        line_end=end,
    )


def csv_row(start, end):
    return SourceLocation(
        type="csv_row",
        row_start=start,
        row_end=end,
    )


def document(elements, mime_type, filename, parser_name="x", metadata=None):
    return ParsedDocument(
        doc_id="doc",
        version_id="v1",
        user_id="u1",
        filename=filename,
        mime_type=mime_type,
        parser_name=parser_name,
        parser_version="1.0",
        elements=elements,
        metadata=metadata or {},
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

    assert len(parent_ids) == len(parents), "parent ids unique"

    indexes = [c.chunk_index for c in chunks]
    assert indexes == sorted(indexes), "indexes continuous"
    assert len(set(indexes)) == len(indexes), "indexes unique"

    for child in children:
        assert child.parent_chunk_id in parent_ids

    assert children, "expect at least one child"


def test_mime_dispatch_chooses_per_type_chunker():
    cases = [
        ("application/pdf", "x.pdf", "pdf"),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "x.docx",
            "docx",
        ),
        ("text/plain", "x.txt", "txt"),
        ("text/markdown", "x.md", "markdown"),
        ("text/csv", "x.csv", "csv"),
        (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "x.xlsx",
            "xlsx",
        ),
        ("application/json", "x.json", "json"),
        ("text/html", "x.html", "html"),
        (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "x.pptx",
            "pptx",
        ),
        ("application/octet-stream", "x.unknown", "section"),
    ]
    for mime, filename, expected in cases:
        doc = document([], mime, filename)
        chunker = get_chunker_for(doc)
        assert chunker.identifier == expected, (mime, expected)


def test_no_headings_produce_one_unsectioned_parent():
    doc = document(
        [
            element("line one", section_path=[]),
            element("line two", section_path=[]),
            element("line three", section_path=[]),
        ],
        "text/markdown",
        "simple.md",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) == 1
    assert parents[0].section_path == []
    assert len(children) == 1


def test_content_before_first_heading_is_root_parent_then_sections():
    doc = document(
        [
            element("preamble", section_path=[]),
            element("First Heading", section_path=["First Heading"]),
            element("body under first", section_path=["First Heading"]),
            element("Second Heading", section_path=["Second Heading"]),
            element("body under second", section_path=["Second Heading"]),
        ],
        "text/html",
        "page.html",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert [p.section_path for p in parents] == [
        [],
        ["First Heading"],
        ["Second Heading"],
    ]


def test_section_chunker_uses_heading_sections():
    doc = document(
        [
            element("H1", element_type="h1", section_path=["Home"]),
            element("intro", section_path=["Home"]),
            element("H2", element_type="h2", section_path=["Home", "Guides"]),
            element("guide body", section_path=["Home", "Guides"]),
        ],
        "text/html",
        "page.html",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert [p.section_path for p in parents] == [
        ["Home"],
        ["Home", "Guides"],
    ]


def test_oversized_section_splits_into_parts_with_distinct_ids_and_locations():
    def paged(text):
        return element(
            text,
            locations=[
                SourceLocation(
                    type="pdf_bbox",
                    page=1,
                    bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
                )
            ],
        )

    first = paged("A" * 1000)
    long = paged("B" * 4000)
    others = [
        paged(f"para {i}" + "y" * 2500)
        for i in range(8)
    ]
    elements = [first, long] + others
    doc = document(elements, "application/pdf", "big.pdf")

    chunker = SectionChunker(child_max_chars=1000, parent_max_chars=5000)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)

    # 1000 + 4000 + 8 * ~2507 chars > 5000 -> always more than one part
    assert len(parents) >= 3
    assert len({p.chunk_id for p in parents}) == len(parents)
    assert all(
        len(p.text) <= 5000
        for p in parents
    ), "each part stays under the parent fallback cap"

    # Location custody: each part owns exactly its elements' locations.
    # No sharing means the combined parent locations equal the combined
    # element locations (no duplication, no loss).
    all_locations = [
        loc for el in elements
        for loc in el.locations
    ]
    parent_locations = [
        loc for p in parents
        for loc in p.locations
    ]
    assert len(parent_locations) == len(all_locations)

    child_counts = Counter(
        c.parent_chunk_id
        for c in children
    )
    assert len(child_counts) == len(parents)
    for p in parents:
        assert child_counts[p.chunk_id] >= 1


def test_section_soft_cap_splits_large_section_into_parts():
    doc = document(
        [
            element("x" * 7500),
            element("y" * 7500),
        ],
        "text/markdown",
        "big.md",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    # The default (standardized) chunker token-budgets parents: the
    # section splits into multiple parts, none above the hard token cap.
    assert len(parents) >= 2, "token budget splits an oversized section"
    for parent in parents:
        assert estimate_tokens(parent.text) <= 1800
        assert parent.metadata["part_count"] == len(parents)
        assert "\u2026[truncated]" not in parent.text
    assert len({p.chunk_id for p in parents}) == len(parents)


def test_section_tiny_tail_folds_up_without_breaching_hard_cap():
    doc = document(
        [
            element("x" * 7500),
            element("y" * 7500),
            element("c" * 400),
        ],
        "text/html",
        "long.html",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    # The tiny 'c' tail merges upward into the preceding parent (it keeps
    # the merged parent well under the 1800-token hard cap), so no
    # structurally-forced wafer parent appears.
    assert all(
        estimate_tokens(p.text) <= 1800
        for p in parents
    )
    assert not any(
        estimate_tokens(p.text) <= 200
        for p in parents
    ), "the tiny tail folds up instead of becoming a wafer parent"


def test_section_tiny_tail_merges_when_within_hard_cap():
    doc = document(
        [
            element(
                "sentence with enough words to fill a couple of "
                "windows. " * 25
            ),
            element("tail words"),
        ],
        "text/html",
        "long.html",
    )
    chunker = MarkdownChunker()
    chunks = chunker.chunk(doc)
    parents, _ = parents_children(chunks)
    # The token-budget packer folds the tiny trailing parent in when it
    # keeps the merged parent under the hard token ceiling.
    assert len(parents) == 1


def test_single_oversized_element_splits_into_token_budgeted_parents():
    doc = document(
        [element("z" * 15000)],
        "application/pdf",
        "huge.pdf",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert len(parents) >= 4
    assert all(
        estimate_tokens(p.text) <= 1800
        for p in parents
    )
    assert "\u2026[truncated]" not in "".join(
        p.text for p in parents
    )
    assert {p.metadata["part_count"] for p in parents} == {
        len(parents)
    }


def test_json_soft_cap_produces_smaller_item_boundary_parts():
    def leaf(text, path):
        return element(
            text,
            element_type="json_value",
            locations=[
                SourceLocation(
                    type="json_path",
                    json_path=path,
                )
            ],
        )

    cells = [
        leaf("x" * 1500, f"$.users[{i}].note")
        for i in range(10)
    ]
    doc = document(cells, "application/json", "big.json")
    chunker = JsonChunker(parent_soft_max_chars=3500)
    chunks = chunker.chunk(doc)
    parents, _ = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) >= 3, "oversized top-level key split into soft parts"
    assert all(
        p.metadata["part_count"] == len(parents)
        for p in parents
    )
    combined = [
        loc for parent in parents
        for loc in parent.locations
    ]
    assert len(combined) == len(cells)


def test_csv_table_parent_not_soft_split():
    header = element(
        "A | B",
        element_type="table_header",
        locations=[csv_row(1, 1)],
        metadata={"columns": ["A", "B"]},
    )
    rows = [
        element(
            f"r{i} " + "y" * 80,
            element_type="table_row",
            locations=[csv_row(i, i)],
            metadata={"columns": ["A", "B"]},
        )
        for i in range(2, 200)
    ]
    doc = document(
        [header] + rows,
        "text/csv",
        "wide.csv",
        metadata={"row_count": 198},
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert len(parents) == 1, "table stays one parent despite soft cap"
    assert "part" not in parents[0].metadata


def test_pptx_slide_parent_not_soft_split():
    def slide_el(text, slide):
        return element(
            text,
            locations=[
                SourceLocation(
                    type="slide_bbox",
                    slide=slide,
                )
            ],
            metadata={"slide_number": slide},
        )

    doc = document(
        [
            slide_el("t" * 1200, 1),
            slide_el("u" * 1200, 1),
        ],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "deck.pptx",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    # The slide is a seam: every parent belongs to slide 1. A single
    # 1200-token slide almost fills the 250-token child window 9 times,
    # which breaches the hard 8-children-per-parent cap, so the slide's
    # content splits into parts -- still one slide, never coalesced with
    # another slide.
    assert len(parents) == 2, "slide content splits on the child cap"
    assert {p.metadata["slide"] for p in parents} == {1}
    assert {p.metadata["part_count"] for p in parents} == {2}
    assert all(p.metadata["part"] in (1, 2) for p in parents)


def test_txt_paragraph_runs_grouped_into_one_parent_with_children():
    doc = document(
        [
            element("a line", locations=[text_offset(1, 1)]),
            element("b line", locations=[text_offset(2, 2)]),
            # blank line at 3 -> gap of 2 at line 4
            element("new para", locations=[text_offset(4, 4)]),
            element("more", locations=[text_offset(5, 5)]),
        ],
        "text/plain",
        "notes.txt",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert len(parents) == 1, "small paragraph runs share one parent"
    assert parents[0].metadata["paragraph_runs"] == 2
    assert parents[0].metadata["lines"] == [1, 5]
    # the short run is short enough for a single windowed child
    assert len(children) == 1
    assert "a line" in children[0].text
    assert "new para" in children[0].text
    assert_invariants(chunks)


def test_txt_paragraph_groups_split_blocks_at_run_boundaries():
    def run(elements_start, count=10, pad=90):
        return [
            element(
                f"line {i}" + "y" * pad,
                locations=[text_offset(
                    elements_start + i,
                    elements_start + i,
                )],
            )
            for i in range(count)
        ]

    # 3 runs, ~1000 chars each; group target 1500 -> 3 parent groups
    doc = document(
        run(1) + run(101) + run(201),
        "text/plain",
        "big.txt",
    )
    chunker = TxtChunker(group_target_chars=1500)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) == 3, "each run becomes its own group at this target"
    for parent in parents:
        assert parent.metadata["paragraph_runs"] == 1


def test_txt_oversized_group_splits_parts_at_run_boundaries():
    def run(elements_start, count=10, pad=90):
        return [
            element(
                f"line {i}" + "y" * pad,
                locations=[text_offset(
                    elements_start + i,
                    elements_start + i,
                )],
            )
            for i in range(count)
        ]

    # 3 runs of ~1000 chars = ~3000 total, but parent cap only 2000
    doc = document(
        run(1) + run(101) + run(201),
        "text/plain",
        "big.txt",
    )
    chunker = TxtChunker(
        group_target_chars=100000,
        parent_max_chars=2000,
    )
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    # one block (~3000 chars) oversize -> split into parts at run bounds
    assert len(parents) > 1
    for parent in parents:
        # every part starts on a run boundary (line number % 100 == 1)
        first_lines = []
        for loc in parent.locations[:1]:
            if loc.type == "text_offset" and loc.line_start is not None:
                first_lines.append(loc.line_start)
        for gt_first in first_lines:
            assert (gt_first - 1) % 100 == 0


def test_txt_no_blank_lines_is_one_run():
    doc = document(
        [
            element("l1", locations=[text_offset(1, 1)]),
            element("l2", locations=[text_offset(2, 2)]),
            element("l3", locations=[text_offset(3, 3)]),
        ],
        "text/plain",
        "notes.txt",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert len(parents) == 1


def test_markdown_code_block_is_atomic():
    doc = document(
        [
            element("Before", element_type="paragraph"),
            element("def f():\n    pass\n", element_type="code_block"),
            element("After", element_type="paragraph"),
        ],
        "text/markdown",
        "readme.md",
    )
    chunker = MarkdownChunker(child_max_chars=5)
    chunks = chunker.chunk(doc)
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    assert len(children) == 3, "code block kept as its own child"
    assert children[1].text.startswith("def f():")
    assert children[1].metadata["parent_chunk_id"] == children[0].metadata["parent_chunk_id"]
    assert_invariants(chunks)


def test_markdown_heading_only_section_folds_into_next_content():
    doc = document(
        [
            element("# API", element_type="heading", section_path=["API"]),
            element("# Endpoints", element_type="heading", section_path=["Endpoints"]),
            element("endpoint body", section_path=["Endpoints"]),
        ],
        "text/markdown",
        "README.md",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) == 1, "heading-only section produces no thin parent"
    assert parents[0].section_path == ["Endpoints"]
    assert parents[0].text.startswith("# API")
    assert "endpoint body" in parents[0].text


def test_section_chunker_folds_heading_comb_into_next_section():
    doc = document(
        [
            element("API", element_type="h1", section_path=["API"]),
            element("Endpoints", element_type="h2", section_path=["API", "Endpoints"]),
            element("endpoint body", section_path=["API", "Endpoints"]),
            element("Auth", element_type="h1", section_path=["Auth"]),
            element("auth body", section_path=["Auth"]),
        ],
        "text/html",
        "page.html",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert_invariants(chunks)
    assert [p.section_path for p in parents] == [
        ["API", "Endpoints"],
        ["Auth"],
    ]
    assert parents[0].text.startswith("API\n")


def test_section_chunker_folds_trailing_stub_into_previous_section():
    doc = document(
        [
            element("Intro", element_type="h1", section_path=["Intro"]),
            element("intro body", section_path=["Intro"]),
            element("Dangling", element_type="h2", section_path=["Intro", "Dangling"]),
        ],
        "text/html",
        "page.html",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert len(parents) == 1
    assert parents[0].section_path == ["Intro"]
    assert parents[0].text.rstrip().endswith("Dangling")


def test_section_chunker_heading_only_whole_document_gives_outline_parent():
    doc = document(
        [
            element("A", element_type="h1", section_path=["A"]),
            element("B", element_type="h2", section_path=["A", "B"]),
        ],
        "text/html",
        "page.html",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert len(parents) == 1
    assert parents[0].text.strip() == "A\n\nB"
    assert children == []


def test_csv_one_table_parent_and_row_metadata():
    header = element(
        "Name | Age",
        element_type="table_header",
        locations=[csv_row(1, 1)],
        metadata={"columns": ["Name", "Age"]},
    )
    rows = [
        element(
            f"row {i}",
            element_type="table_row",
            locations=[csv_row(i, i)],
            metadata={"columns": ["Name", "Age"]},
        )
        for i in range(2, 8)
    ]
    doc = document(
        [header] + rows,
        "text/csv",
        "data.csv",
        metadata={"row_count": 6},
    )
    chunker = CsvChunker(child_max_chars=2000)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) == 1
    assert parents[0].metadata["row_count"] == 6
    assert "schema:" in parents[0].text
    assert len(children) == 1
    assert children[0].metadata["row_start"] == 1
    assert children[0].metadata["row_end"] == 7
    assert children[0].metadata["columns"] == ["Name", "Age"]


def test_csv_many_rows_pack_into_row_groups_with_ranges():
    header = element(
        "A | B",
        element_type="table_header",
        locations=[csv_row(1, 1)],
        metadata={"columns": ["A", "B"]},
    )
    rows = [
        element(
            f"r{i}",
            element_type="table_row",
            locations=[csv_row(i, i)],
            metadata={"columns": ["A", "B"]},
        )
        for i in range(2, 30)
    ]
    doc = document([header] + rows, "text/csv", "data.csv")
    chunker = CsvChunker(child_max_chars=100)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    # rows 2..29 in packs of ~100 chars
    assert len(parents) == 1
    assert len(children) >= 2
    ranges = [(c.metadata["row_start"], c.metadata["row_end"]) for c in children]
    assert ranges == sorted(ranges)
    assert ranges[0][0] == 1
    assert ranges[-1][1] == 29


def test_xlsx_one_parent_per_sheet():
    def sheet_element(text, sheet, etype="spreadsheet_row", row="A1"):
        return element(
            text,
            element_type=etype,
            section_path=[sheet],
            locations=[
                SourceLocation(
                    type="xlsx_range",
                    sheet=sheet,
                    cell_range=row,
                )
            ],
            metadata={"columns": ["X"]},
        )

    doc = document(
        [
            sheet_element("X", "Sheet1", "table_header", "A1"),
            sheet_element("v1", "Sheet1", "spreadsheet_row", "A2"),
            sheet_element("X", "Sheet2", "table_header", "A1"),
            sheet_element("w1", "Sheet2", "spreadsheet_row", "A2"),
        ],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "wb.xlsx",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert [p.section_path for p in parents] == [["Sheet1"], ["Sheet2"]]
    assert parents[0].metadata["sheet"] == "Sheet1"
    assert children[0].metadata["sheet"] == "Sheet1"


def test_json_one_parent_per_top_level_key():
    def leaf(text, path):
        return element(
            text,
            element_type="json_value",
            locations=[
                SourceLocation(
                    type="json_path",
                    json_path=path,
                )
            ],
        )

    doc = document(
        [
            leaf("users: a1", "$.users[0].name"),
            leaf("users: a2", "$.users[0].age"),
            leaf("users: b1", "$.users[1].name"),
            leaf("meta: x", "$.meta"),
        ],
        "application/json",
        "data.json",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert_invariants(chunks)
    assert [p.section_path for p in parents] == [["users"], ["meta"]]


def test_json_oversized_array_splits_at_item_boundaries():
    def leaf(text, path):
        return element(
            text,
            element_type="json_value",
            locations=[
                SourceLocation(
                    type="json_path",
                    json_path=path,
                )
            ],
        )

    cells = [
        leaf(
            f"user {i} with some description text " + "p" * 40,
            f"$.users[{i}].name",
        )
        for i in range(50)
    ]
    doc = document(cells, "application/json", "big.json")
    chunker = JsonChunker(child_max_chars=2000, parent_max_chars=2000)
    chunks = chunker.chunk(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    assert len(parents) > 1, "oversized top-level key split into parts"
    for parent in parents:
        assert len(parent.text) <= 2000
        assert parent.metadata["part_count"] == len(parents)
    combined_locations = [
        loc
        for parent in parents
        for loc in parent.locations
    ]
    assert len(combined_locations) == len(cells)
    # each part contains only its own leaves
    children_by_parent = {}
    for child in children:
        children_by_parent.setdefault(child.parent_chunk_id, []).append(child)
    for parent, kids in children_by_parent.items():
        for k in kids:
            assert k.metadata["parent_chunk_id"] == parent


def test_pptx_short_slides_coalesce_into_one_parent():
    def slide_el(text, slide, section_path=None, etype="list_item"):
        return element(
            text,
            element_type=etype,
            section_path=section_path or [],
            locations=[
                SourceLocation(
                    type="slide_bbox",
                    slide=slide,
                )
            ],
            metadata={"slide_number": slide},
        )

    doc = document(
        [
            slide_el("Slide One", 1, section_path=["Slide One"], etype="heading"),
            slide_el("bullet a", 1),
            slide_el("bullet b", 1),
            slide_el("Slide Two", 2, section_path=["Slide Two"], etype="heading"),
            slide_el("bullet c", 2),
        ],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "deck.pptx",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert_invariants(chunks)
    # Short slides coalesce through `_parent_groups` into a banded
    # parent instead of each becoming a one-child wafer parent. The
    # coalesced parent records the slide range it covers.
    assert len(parents) == 1
    assert parents[0].metadata["slide_first"] == 1
    assert parents[0].metadata["slide_last"] == 2
    assert "Slide One" in parents[0].text
    assert "Slide Two" in parents[0].text

    # Slide content packs into windowed children (feature-scaled): each
    # short deck becomes a single child covering both slides.
    child_counts = Counter(
        c.parent_chunk_id for c in children
    )
    assert child_counts[parents[0].chunk_id] == 1
    slide_one = children[0]
    assert "Slide One" in slide_one.text
    assert "bullet a" in slide_one.text


def test_pptx_skips_empty_text_blocks_as_children():
    def slide_el(text, slide, section_path=None, etype="list_item"):
        return element(
            text,
            element_type=etype,
            section_path=section_path or [],
            locations=[
                SourceLocation(
                    type="slide_bbox",
                    slide=slide,
                )
            ],
            metadata={"slide_number": slide},
        )

    doc = document(
        [
            slide_el("Slide One", 1, section_path=["Slide One"], etype="heading"),
            element(
                "",
                element_type="image",
                locations=[SourceLocation(type="slide_bbox", slide=1)],
                metadata={"slide_number": 1},
            ),
            slide_el("bullet a", 1),
        ],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "deck.pptx",
    )
    chunks = chunk_document(doc)
    parents, children = parents_children(chunks)
    assert len(parents) == 1
    assert len(children) == 1, "empty-text image is not a child"
    assert all(c.text.strip() for c in children)
    assert children[0].text == "Slide One\n\nbullet a"


def test_pptx_untitled_slide_label():
    doc = document(
        [
            element(
                "content",
                locations=[SourceLocation(type="slide_bbox", slide=3)],
                metadata={"slide_number": 3},
            )
        ],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "deck.pptx",
    )
    chunks = chunk_document(doc)
    parents, _ = parents_children(chunks)
    assert parents[0].section_path == ["Slide 3"]


def test_children_respect_child_cap():
    texts = ["s" * 400 for _ in range(20)]
    doc = document(
        [element(t) for t in texts],
        "application/pdf",
        "cap.pdf",
    )
    chunker = SectionChunker(child_max_chars=1000)
    chunks = chunker.chunk(doc)
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    assert all(len(c.text) <= 1000 for c in children)
    assert len(children) >= 8


def test_child_metadata_is_a_self_contained_record():
    doc = document(
        [
            element(
                "H1",
                element_type="h1",
                section_path=["Home"],
                value="h1",
            ),
            element(
                "intro",
                section_path=["Home"],
                parent="h1",
                locations=[
                    SourceLocation(
                        type="pdf_bbox",
                        page=2,
                    )
                ],
            ),
        ],
        "application/pdf",
        "page.pdf",
    )
    chunks = chunk_document(doc)
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    child = children[0]
    meta = child.metadata

    assert meta["doc_id"] == "doc"
    assert meta["version_id"] == "v1"
    assert meta["user_id"] == "u1"
    assert meta["mime_type"] == "application/pdf"
    assert meta["strategy"] == "pdf"
    assert meta["chunk_type"] == "child"
    assert meta["chunk_index"] == child.chunk_index
    assert meta["parent_chunk_id"] == (
        child.parent_chunk_id
    )
    assert meta["section_path"] == ["Home"]
    assert meta["pages"] == [2]
    assert meta["parser_version"]


def test_txt_child_carries_line_span_metadata():
    doc = document(
        [
            element(
                "a line",
                locations=[text_offset(1, 1)],
            ),
            element(
                "b line",
                locations=[text_offset(2, 2)],
            ),
        ],
        "text/plain",
        "notes.txt",
    )
    chunks = chunk_document(doc)
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    meta = children[0].metadata
    assert meta["line_start"] == 1
    assert meta["line_end"] == 2
    assert meta["parser"] == "x"


def test_json_child_carries_json_paths():
    def leaf(text, path):
        return element(
            text,
            element_type="json_value",
            locations=[
                SourceLocation(
                    type="json_path",
                    json_path=path,
                )
            ],
        )

    doc = document(
        [
            leaf("a", "$.users[0].name"),
            leaf("b", "$.users[0].age"),
        ],
        "application/json",
        "data.json",
    )
    chunks = chunk_document(doc)
    children = [
        c for c in chunks
        if c.chunk_type == "child"
    ]
    meta = children[0].metadata
    assert meta["json_paths"] == [
        "$.users[0].name",
        "$.users[0].age",
    ]
    assert meta["root"] == "users"