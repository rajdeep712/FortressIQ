"""Chunking snapshot report for every example.* document.

Parses and chunks each ``app/ingestion/parsers/example.*`` file with its
per-type parser + chunker, then writes a JSON snapshot to
``backend/bench_logs/chunk_report_<timestamp>.json`` containing, per file:

  * element / parent / child counts and token stats
  * children-per-parent distribution
  * every parent's full text and every child's full text (so the actual
    content can be reviewed / diffed across runs)

PDFs go through OpenDocumentLoader and PPTX through Docling (network
calls expected; a failure is reported per-file rather than fatal, so local
files still get snapshotted).

Usage (from backend/):

    python scripts/report_chunkers.py               # write a fresh snapshot
    python scripts/report_chunkers.py --json-only    # no parent/child text payloads
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.chunker.tokenizer import (  # noqa: E402
    estimate_tokens,
)
from app.ingestion.parsers.csv_parser import CsvParser  # noqa: E402
from app.ingestion.parsers.docx_parser import DocxParser  # noqa: E402
from app.ingestion.parsers.html_parser import HtmlParser  # noqa: E402
from app.ingestion.parsers.json_parser import JsonParser  # noqa: E402
from app.ingestion.parsers.markdown_parser import MarkdownParser  # noqa: E402
from app.ingestion.parsers.open_document_loader import (  # noqa: E402
    OpenDocumentLoaderParser,
)
from app.ingestion.parsers.txt_parser import TxtParser  # noqa: E402
from app.ingestion.parsers.xlsx_parser import XlsxParser  # noqa: E402
from app.ingestion.parsers.dockling import DoclingParser  # noqa: E402

PARSERS_DIR = BACKEND_DIR / "app" / "ingestion" / "parsers"

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

LOG_DIR = BACKEND_DIR / "bench_logs"


def _section(section_path) -> list[str] | None:
    if not section_path:
        return None
    return [str(p) for p in section_path if str(p)]


def _child_stats(doc, chunks, include_text: bool) -> dict:
    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]
    by_parent: dict[str, list] = {}
    for child in children:
        by_parent.setdefault(child.parent_chunk_id, []).append(child)
    parent_counts = [
        len(by_parent.get(p.chunk_id, [])) for p in parents
    ]
    parent_tokens = [estimate_tokens(p.text) for p in parents]
    child_tokens = [estimate_tokens(c.text) for c in children]

    return {
        "parent_count": len(parents),
        "child_count": len(children),
        "children_per_parent": [
            {
                "id": p.chunk_id,
                "section": _section(p.section_path),
                "children": len(by_parent.get(p.chunk_id, [])),
                "parent_chars": len(p.text),
                "parent_tokens": estimate_tokens(p.text),
                "strategy": (p.metadata or {}).get("strategy"),
                "parent_text" if include_text else "_": (
                    p.text if include_text else None
                ),
            }
            for p in parents
        ],
        "children": [
            {
                "id": c.chunk_id,
                "parent_chunk_id": c.parent_chunk_id,
                "chars": len(c.text),
                "tokens": estimate_tokens(c.text),
                "section": _section(c.section_path),
                "child_text" if include_text else "_": (
                    c.text if include_text else None
                ),
            }
            for c in children
        ],
        "parent_token_max": max(parent_tokens) if parent_tokens else 0,
        "child_token_max": max(child_tokens) if child_tokens else 0,
        "child_token_min": min(child_tokens) if child_tokens else 0,
        "child_token_mean": (
            round(sum(child_tokens) / len(child_tokens), 1)
            if child_tokens
            else 0
        ),
        "children_per_parent_max": max(parent_counts) if parent_counts else 0,
        "children_per_parent_min": min(parent_counts) if parent_counts else 0,
        "children_per_parent_dist": {
            str(n): parent_counts.count(n)
            for n in sorted(set(parent_counts))
        },
    }


async def process_file(path: Path, include_text: bool) -> dict:
    ext = path.suffix.lower()
    parser_class, mime_type = PARSERS_BY_EXTENSION.get(ext)
    if parser_class is None:
        return {"file": str(path), "status": "SKIP", "reason": "unsupported"}
    parser = parser_class()
    try:
        parsed = await parser.parse(
            file_path=path,
            doc_id="probe-doc",
            version_id="probe-ver",
            user_id="probe-user",
            filename=path.name,
            mime_type=mime_type,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "file": str(path),
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        chunks = chunk_document(parsed)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": str(path),
            "status": "FAIL",
            "reason": f"chunking: {type(exc).__name__}: {exc}",
        }
    stats = _child_stats(parsed, chunks, include_text)
    stats.update(
        {
            "file": str(path),
            "status": "PASS",
            "mime_type": mime_type,
            "strategy": (
                chunks[0].metadata.get("strategy") if chunks else None
            ),
            "elements": len(parsed.elements),
        }
    )
    return stats


async def main() -> None:
    include_text = "--json-only" not in sys.argv
    files = sorted(
        p
        for p in PARSERS_DIR.glob("example.*")
        if p.suffix.lower() in PARSERS_BY_EXTENSION
    )
    results = []
    for path in files:
        results.append(await process_file(path, include_text))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LOG_DIR / f"chunk_report_{ts}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "files": results,
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out.resolve()}")
    for r in results:
        print(
            f"  {r.get('status'):<4} {Path(r.get('file', '')).name:<16} "
            f"parents={r.get('parent_count', '-'):<4} "
            f"children={r.get('child_count', '-'):<4} "
            f"child_tokens_max={r.get('child_token_max', '-')} "
            f"children/max_per_parent={r.get('children_per_parent_max', '-')}"
        )
        if r.get("status") == "FAIL":
            print(f"         reason: {r.get('reason')}")


if __name__ == "__main__":
    asyncio.run(main())
