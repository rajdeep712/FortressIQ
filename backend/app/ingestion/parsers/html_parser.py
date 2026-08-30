from collections import Counter
from pathlib import Path
from uuid import uuid4
import sys

# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from bs4 import BeautifulSoup

from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.text_utils import split_long_line

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_TARGET_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr")
_NOISE_TAGS = ("script", "style", "noscript", "iframe", "svg", "template")
_MAX_CELL_CHARS = 500


class HtmlParser(DocumentParser):

    name = "html"
    version = "1.1"

    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        raw = file_path.read_bytes()

        soup = BeautifulSoup(
            raw,
            "html.parser",
        )

        for noise in soup.find_all(_NOISE_TAGS):
            noise.decompose()

        title = ""
        if soup.title is not None:
            title = soup.title.get_text(" ", strip=True)

        kept = _deduplicate_nested(soup.find_all(_TARGET_TAGS))

        table_headers = _build_table_headers(kept)

        elements = []
        section = []
        order = 0

        for order, tag in enumerate(kept, start=1):

            if tag.name in _HEADING_TAGS:
                text = tag.get_text(" ", strip=True)
                if not text:
                    continue
                section = _push_heading(section, tag.name, text)
                element_type = tag.name
                section_path = list(section)
            else:
                text = tag.get_text(" ", strip=True)
                if not text:
                    continue
                element_type = tag.name
                section_path = list(section)

            dom_path = build_dom_path(tag)
            hrefs = _collect_hrefs(tag)

            if tag.name == "tr":
                pieces = _table_row_texts(tag, table_headers)
                if pieces is None:
                    continue
                row_type, row_text = pieces
                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type=row_type,
                        text=row_text,
                        order=order,
                        section_path=section_path,
                        locations=[SourceLocation(type="html_dom", dom_path=dom_path)],
                        metadata={"hrefs": hrefs} if hrefs else {},
                    )
                )
                continue

            for fragment in _text_fragments(text):
                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type=element_type,
                        text=fragment,
                        order=order,
                        section_path=section_path,
                        locations=[SourceLocation(type="html_dom", dom_path=dom_path)],
                        metadata={"hrefs": hrefs} if hrefs else {},
                    )
                )

        for index, element in enumerate(elements, start=1):
            element.order = index

        if not elements:
            body_text = soup.get_text(" ", strip=True)
            if body_text:
                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type="p",
                        text=body_text,
                        order=1,
                        locations=[SourceLocation(type="html_dom", dom_path="")],
                    )
                )

        return ParsedDocument(
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
            parser_name=self.name,
            parser_version=self.version,
            elements=elements,
            metadata={
                "title": title,
                "encoding": soup.original_encoding or "utf-8",
                "char_count": sum(len(e.text) for e in elements),
                "element_counts": dict(Counter(e.element_type for e in elements)),
            },
        )


def _text_fragments(text: str):
    for fragment, _ in split_long_line(text):
        if fragment:
            yield fragment


def _push_heading(section, tag_name, text):
    level = int(tag_name[1])
    return section[: level - 1] + [text]


def _deduplicate_nested(tags):
    matched = set(tags)
    kept = []
    for tag in tags:
        parent = tag.parent
        duplicated = False
        while parent is not None:
            if parent in matched:
                duplicated = True
                break
            parent = parent.parent
        if not duplicated:
            kept.append(tag)
    return kept


def _build_table_headers(trs):
    info = {}
    for tr in trs:
        if tr.name != "tr":
            continue
        table = tr.find_parent("table")
        if table is None:
            continue
        marker = id(table)
        if marker in info:
            continue
        header_tr = table.find("tr")
        first_th = table.find("th")
        if first_th is not None and first_th.find_parent("tr") is not None:
            header_tr = first_th.find_parent("tr")
        headers = []
        if header_tr is not None:
            headers = [
                cell.get_text(" ", strip=True)
                for cell in header_tr.find_all(["td", "th"], recursive=False)
            ]
        info[marker] = {"tr": header_tr, "headers": headers}
    return info


def _table_row_texts(tr, table_headers):
    table = tr.find_parent("table")
    if table is None:
        return None
    info = table_headers.get(id(table))
    if info is None:
        return None
    header_tr = info["tr"]
    headers = info["headers"]
    cells = [
        cell.get_text(" ", strip=True)
        for cell in tr.find_all(["td", "th"], recursive=False)
    ]
    cells = [c[:_MAX_CELL_CHARS] for c in cells]

    if tr is header_tr:
        return "table_header", " | ".join(h for h in headers if h)

    return "table_row", _format_row(headers, cells)


def _format_row(headers, cells):
    parts = []
    extra_index = 0
    for i, cell in enumerate(cells):
        header = headers[i] if i < len(headers) else None
        if header is None:
            extra_index += 1
            parts.append(f"extra_{extra_index}: {cell}")
        else:
            label = header or f"column{i + 1}"
            parts.append(f"{label}: {cell}")
    return "\n".join(parts)


def _collect_hrefs(element):
    hrefs = []
    for link in element.find_all("a", href=True):
        url = link.get("href")
        if not url:
            continue
        url = url.strip()
        if url and url not in hrefs:
            hrefs.append(url)
    return hrefs


def build_dom_path(element) -> str:

    path = []
    current = element

    while current is not None and getattr(current, "name", None):

        parent = current.parent

        if parent is None:
            path.append(f"{current.name}:nth-of-type(1)")
            break

        try:
            siblings = list(
                parent.find_all(
                    current.name,
                    recursive=False,
                )
            )
            index = siblings.index(current) + 1
        except (ValueError, AttributeError):
            index = 1

        path.append(
            f"{current.name}:nth-of-type({index})"
        )

        current = parent

    return " > ".join(
        reversed(path)
    )


if __name__ == "__main__":
    import asyncio
    html_parser = HtmlParser()
    
    document = asyncio.run(
        html_parser.parse(
            file_path=Path(__file__).parent / "example.html",
            doc_id="123",
            version_id="1",
            user_id="123",
            filename="example.html",
            mime_type="text/html",
        )
    )
    print(document)