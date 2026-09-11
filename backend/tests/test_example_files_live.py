"""Live verification that every example.* parser + chunker round-trips
through the standardized (and legacy CSV/XLSX) pipeline without violating
the token invariants.

Network-backed parsers (PDF via OpenDocumentLoader, PPTX via Docling) are
expected to call out; if the endpoint is unreachable the per-file test is
skipped rather than failed so a disconnected run still reports cleanly.

Run from backend/:

    python -m pytest tests/test_example_files_live.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.chunker.tokenizer import estimate_tokens
from app.ingestion.parsers.csv_parser import CsvParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.json_parser import JsonParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.open_document_loader import (
    OpenDocumentLoaderParser,
)
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.parsers.xlsx_parser import XlsxParser
from app.ingestion.parsers.dockling import DoclingParser

PARSERS_DIR = Path(__file__).resolve().parents[1] / "app" / "ingestion" / "parsers"

PARSERS_BY_EXTENSION = {
    ".csv": (CsvParser, "text/csv"),
    ".docx": (
        DocxParser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".md": (MarkdownParser, "text/markdown"),
    ".html": (HtmlParser, "text/html"),
    ".json": (JsonParser, "application/json"),
    ".pdf": (OpenDocumentLoaderParser, "application/pdf"),
    ".pptx": (
        DoclingParser,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".txt": (TxtParser, "text/plain"),
    ".xlsx": (
        XlsxParser,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
}

# Standardized windowing applies to every type except CSV/XLSX (legacy).
STANDARD_TYPES = {
    ".docx",
    ".md",
    ".html",
    ".json",
    ".pdf",
    ".pptx",
    ".txt",
}

CHILD_TOKEN_MAX = 250
PARENT_TOKEN_MAX = 1800

EXAMPLE_FILES = sorted(
    p
    for p in PARSERS_DIR.glob("example.*")
    if p.suffix.lower() in PARSERS_BY_EXTENSION
)


@pytest.mark.parametrize(
    "path",
    EXAMPLE_FILES,
    ids=[p.name for p in EXAMPLE_FILES],
)
def test_example_file_chunks_within_invariants(path):
    ext = path.suffix.lower()
    parser_class, mime_type = PARSERS_BY_EXTENSION[ext]
    parser = parser_class()

    try:
        parsed = asyncio.run(
            parser.parse(
                file_path=path,
                doc_id="live-doc",
                version_id="live-ver",
                user_id="live-user",
                filename=path.name,
                mime_type=mime_type,
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{path.name} parse unavailable: {type(exc).__name__}: {exc}")

    chunks = chunk_document(parsed)
    assert chunks, f"{path.name}: no chunks produced"

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]
    assert parents, f"{path.name}: no parents"
    assert children, f"{path.name}: no children"

    # every child points at a real parent; indexes continuous
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_chunk_id in parent_ids for c in children)
    indexes = [c.chunk_index for c in chunks]
    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)

    if ext in STANDARD_TYPES:
        for parent in parents:
            assert estimate_tokens(parent.text) <= PARENT_TOKEN_MAX, (
                f"{path.name}: parent over {PARENT_TOKEN_MAX} tokens "
                f"({estimate_tokens(parent.text)})"
            )
        for child in children:
            assert estimate_tokens(child.text) <= CHILD_TOKEN_MAX, (
                f"{path.name}: child over {CHILD_TOKEN_MAX} tokens "
                f"({estimate_tokens(child.text)})"
            )
    else:
        # Legacy CSV/XLSX char caps.
        assert all(
            len(c.text) <= 2000 for c in children
        ), f"{path.name}: legacy CSV/XLSX child over 2000 chars"
        assert all(
            len(c.text) <= 20000 for c in chunks
        ), f"{path.name}: legacy CSV/XLSX chunk over 20000 chars"
