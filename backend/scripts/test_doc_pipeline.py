"""
Document parser + chunker probe.

Usage (from the backend/ directory with the venv python):

    python scripts/test_doc_pipeline.py example.docx
    python scripts/test_doc_pipeline.py path/to/any.pdf --no-text
    python scripts/test_doc_pipeline.py example.csv --preview
    python scripts/test_doc_pipeline.py example.csv path/to/notes.txt

The argument is a filename; if it does not exist relative to the current
working directory it is looked up inside app/ingestion/parsers/ (the
example.* files live there).

Each file is dispatched to the parser matching its extension, parsed, and
chunked with the per-document-type chunker routed by MIME type (see
app/ingestion/chunker/). Full chunk text is printed by default;
--preview truncates to one line, --no-text suppresses the content.
"""

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make `backend/` importable when the script is run from anywhere.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.payload import build_payload  # noqa: E402
from app.ingestion.parsers.csv_parser import CsvParser  # noqa: E402
from app.ingestion.parsers.docx_parser import DocxParser  # noqa: E402
from app.ingestion.parsers.markdown_parser import MarkdownParser  # noqa: E402
from app.ingestion.parsers.html_parser import HtmlParser  # noqa: E402
from app.ingestion.parsers.json_parser import JsonParser  # noqa: E402
from app.ingestion.parsers.open_document_loader import (  # noqa: E402
    OpenDocumentLoaderParser,
)
from app.ingestion.parsers.txt_parser import TxtParser  # noqa: E402
from app.ingestion.parsers.xlsx_parser import XlsxParser  # noqa: E402
from app.ingestion.parsers.dockling import (  # noqa: E402
    DoclingParser,
)

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

PARSERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ingestion"
    / "parsers"
)

PREVIEW_CHARS = 120

H_LINE = "=" * 78
SUB_LINE = "-" * 78

FULL_TEXT = "full"
PREVIEW_TEXT = "preview"
NO_TEXT = "none"


def resolve_file(arg: str) -> tuple[Path, str]:
    """Return (path, source) where source is 'given' or 'parsers'."""
    given = Path(arg)
    if given.is_file():
        return given, "given"
    candidate = PARSERS_DIR / given.name
    if candidate.is_file():
        return candidate, "parsers/"
    raise FileNotFoundError(
        f"File not found: '{arg}' (tried cwd and {PARSERS_DIR})"
    )


def short_id(value: str, length: int = 12) -> str:
    return value[:length]


def one_line(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(
        text.split()
    )
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def format_content(
    text: str,
    text_mode: str,
) -> str | None:
    """Return how a chunk's text should be printed for a text mode."""
    if text_mode == FULL_TEXT:
        return text
    if text_mode == PREVIEW_TEXT:
        return one_line(text)
    return None


async def run_file(
    file_path: Path,
    source: str,
    text_mode: str,
) -> None:

    extension = file_path.suffix.lower()

    if extension not in PARSERS_BY_EXTENSION:
        print(f"SKIP : {source}{file_path.name} (unsupported extension)\n")
        return

    parser_class, mime_type = PARSERS_BY_EXTENSION[extension]

    parser = parser_class()

    if source == "parsers":
        display_path = str(
            file_path.relative_to(BACKEND_DIR)
        )
    else:
        display_path = str(file_path)

    print(H_LINE)
    print(f" {extension.upper()} :: {file_path.name}")
    print(f" ({display_path})")
    print(H_LINE)

    started = time.perf_counter()

    try:

        parsed = await parser.parse(
            file_path=file_path,
            doc_id="probe-doc",
            version_id="probe-ver",
            user_id="probe-user",
            filename=file_path.name,
            mime_type=mime_type,
        )

    except Exception as exc:  # noqa: BLE001 - surface any parser failure
        elapsed = time.perf_counter() - started
        print(f" parser : {parser_class.__name__}")
        print(f" mime   : {mime_type}")
        print(f" RESULT : FAIL  ({type(exc).__name__}: {exc})")
        print(f" elapsed: {elapsed:.2f}s")
        print()
        return

    element_counts = parsed.metadata.get("element_counts")
    if not isinstance(element_counts, dict):
        element_counts = dict(
            Counter(
                element.element_type
                for element in parsed.elements
            )
        )

    counts_text = ", ".join(
        f"{name}={count}"
        for name, count in sorted(
            element_counts.items()
        )
    )

    chunks = chunk_document(parsed)

    parents = [
        chunk for chunk in chunks
        if chunk.chunk_type == "parent"
    ]

    children = [
        chunk for chunk in chunks
        if chunk.chunk_type == "child"
    ]

    parent_ids = {
        parent.chunk_id
        for parent in parents
    }

    parent_by_id = {
        parent.chunk_id: parent
        for parent in parents
    }

    children_by_parent: dict[
        str, list
    ] = {}

    for child in children:
        children_by_parent.setdefault(
            child.parent_chunk_id,
            [],
        ).append(child)

    from app.core.config import settings  # noqa: E402

    cap = settings.chunk_child_max_chars

    strategy = (
        chunks[0].metadata.get("strategy")
        if chunks
        else "-"
    )

    over_cap = [
        child for child in children
        if len(child.text) > cap
    ]

    indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    indexes_continuous = (
        indexes == sorted(indexes)
        and len(set(indexes)) == len(indexes)
    )

    links_valid = (
        all(
            child.parent_chunk_id in parent_ids
            for child in children
        )
    )

    print(f" parser : {parsed.parser_name} v{parsed.parser_version}")
    print(f" mime   : {mime_type}")
    print(f" strategy: {strategy}")

    chunker_hard = settings.chunk_parent_max_chars
    chunker_soft = min(
        settings.chunk_parent_soft_max_chars,
        chunker_hard,
    )
    print(
        " parent caps:"
        f" soft {chunker_soft}"
        f" / hard {chunker_hard}"
    )
    char_count = parsed.metadata.get(
        "char_count"
    )

    if not isinstance(
        char_count,
        (int, float),
    ):

        char_count = sum(
            len(element.text)
            for element in parsed.elements
        )

    print(
        f" elements: {len(parsed.elements)}"
        f"   ({counts_text})"
    )
    print(
        f" chars  : {char_count}"
    )

    if isinstance(parsed.metadata, dict):
        for key in (
            "page_count",
            "sheet_count",
            "row_count",
            "slide_count",
            "library",
        ):
            if key in parsed.metadata:
                print(f" {key:<7}: {parsed.metadata[key]}")

    print()

    for parent_index, parent in enumerate(
        parents
    ):

        section = (
            "[" + " / ".join(parent.section_path) + "]"
            if parent.section_path
            else "[root / no section]"
        )

        parent_children = children_by_parent.get(
            parent.chunk_id,
            [],
        )

        print(SUB_LINE)
        print(
            f" PARENT-{parent_index}   {section}"
        )
        print(
            f"   id: {short_id(parent.chunk_id)}"
            f"    chars: {len(parent.text)}"
            f"    children: {len(parent_children)}"
        )

        parent_content = format_content(
            parent.text,
            text_mode,
        )

        if parent_content is not None:
            print(
                f"   text: {parent_content}"
            )
            print()

        for child_index, child in enumerate(
            parent_children
        ):

            glyph = (
                "+-"
                if child_index < len(parent_children) - 1
                else "`-"
            )

            suffix = "  <-- OVER CAP" if (
                len(child.text) > cap
            ) else ""

            indent = (
                "   |"
                if glyph == "+-"
                else "    "
            )

            print(
                f"   {glyph} CHILD-{child_index}"
                f"  {short_id(child.chunk_id)}"
                f"  {len(child.text):>6} chars"
                f"{suffix}"
            )

            payload = build_payload(child)
            payload_lines = json.dumps(
                payload,
                indent=2,
            ).splitlines()

            for line in payload_lines:
                print(f"   {indent}  {line}")
            print()

            child_content = format_content(
                child.text,
                text_mode,
            )

            if child_content is not None:
                for line in (
                    child_content.splitlines()
                    or [child_content]
                ):
                    print(f"   {indent}  {line}")
                print()

        if parent_children:
            print()

    print(SUB_LINE)

    total_child_chars = sum(
        len(child.text)
        for child in children
    )

    max_child = (
        max(
            (len(child.text) for child in children),
            default=0,
        )
    )

    print(
        f" chunks: {len(parents)} parent / {len(children)} children"
    )
    print(
        f" child chars: total {total_child_chars},"
        f" max {max_child} (cap {cap}),"
        f" over-cap {len(over_cap)}"
    )
    print(
        f" chunk indexes continuous: {indexes_continuous}"
    )
    print(
        f" child -> parent links valid: {links_valid}"
    )

    elapsed = time.perf_counter() - started

    print(f" elapsed {elapsed:.2f}s")
    print(f" STATUS : PASS")
    print()


def main() -> None:

    args = sys.argv[1:]

    text_mode = FULL_TEXT

    while "--no-text" in args:
        args.remove("--no-text")
        text_mode = NO_TEXT

    while "--preview" in args:
        args.remove("--preview")
        text_mode = PREVIEW_TEXT

    if not args:
        print(
            "usage: python scripts/test_doc_pipeline.py <file...> "
            "[--no-text] [--preview]"
        )
        sys.exit(2)

    failures = 0

    for arg in args:

        try:
            file_path, source = resolve_file(arg)
        except FileNotFoundError as exc:
            failures += 1
            print(H_LINE)
            print(f" {exc}")
            print(H_LINE)
            print()
            continue

        try:
            asyncio.run(
                run_file(
                    file_path=file_path,
                    source=source,
                    text_mode=text_mode,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(H_LINE)
            print(f" ERROR processing {file_path.name}: {exc}")
            print(H_LINE)
            print()

    print(H_LINE)
    if failures:
        print(f" DONE : {failures} file(s) failed")
    else:
        print(" DONE : all files processed")
    print(H_LINE)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()