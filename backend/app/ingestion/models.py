from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LocationType = Literal[
    "pdf_bbox",
    "page_bbox",
    "slide_bbox",
    "pptx_bbox",
    "docx_bbox",
    "csv_row",
    "xlsx_range",
    "text_offset",
    "html_dom",
    "json_path",
    "unknown",
]


@dataclass
class SourceLocation:
    """
    Format-independent pointer back to the source document.
    """
    type: LocationType

    page: int | None = None
    slide: int | None = None

    bbox: dict[str, Any] | None = None

    row_start: int | None = None
    row_end: int | None = None

    sheet: str | None = None
    cell_range: str | None = None

    line_start: int | None = None
    line_end: int | None = None

    char_start: int | None = None
    char_end: int | None = None

    json_path: str | None = None
    dom_path: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedElement:
    """
    The universal representation produced by every parser.
    """
    element_id: str

    element_type: str

    text: str

    order: int

    section_path: list[str] = field(default_factory=list)

    parent_element_id: str | None = None

    locations: list[SourceLocation] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ParsedDocument:
    """
    Universal representation of a parsed document.
    """
    doc_id: str

    version_id: str

    user_id: str

    filename: str

    mime_type: str

    parser_name: str

    parser_version: str

    elements: list[ParsedElement]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )