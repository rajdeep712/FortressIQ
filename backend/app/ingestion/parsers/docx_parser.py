from __future__ import annotations

import asyncio
import copy
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4
import sys

# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pymupdf

from app.core.config import settings
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.text_utils import (
    MAX_LINE_CHARS,
    split_long_line,
)


class DocxParser(DocumentParser):
    """
    Free DOCX parser.

    Pipeline:

        DOCX
          ↓
        LibreOffice
          ↓
        temporary PDF
          ↓
        PyMuPDF
          ↓
        ParsedDocument

    Responsibilities:
        - convert DOCX to PDF
        - extract text
        - extract bounding boxes
        - detect images
        - infer headings
        - preserve section hierarchy

    It does NOT:
        - chunk
        - embed
        - write to SQL
        - write to Qdrant
    """

    name = "libreoffice_pymupdf_docx"
    version = "1.1"

    def __init__(
        self,
        libreoffice_path: str | None = None,
        conversion_timeout: int = 120,
        keep_converted_pdf: bool = False,
    ) -> None:

        self.libreoffice_path = (
            libreoffice_path
            or settings.libreoffice_path
            or "soffice"
        )

        self.conversion_timeout = (
            conversion_timeout
        )

        self.keep_converted_pdf = (
            keep_converted_pdf
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"DOCX file not found: {path}"
            )

        if path.suffix.lower() != ".docx":

            raise ValueError(
                "DocxParser requires a .docx file."
            )

        return await self.parse_file(
            file_path=path,
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
        )

    # =========================================================
    # MAIN PARSING FLOW
    # =========================================================

    async def parse_file(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        converted_pdf: Path | None = None

        try:

            # -------------------------------------------------
            # 1. Convert DOCX → PDF (off the event loop)
            # -------------------------------------------------

            converted_pdf = await asyncio.to_thread(
                self._convert_to_pdf,
                file_path,
            )

            # -------------------------------------------------
            # 2. Extract from PDF
            # -------------------------------------------------

            return self._parse_pdf(
                pdf_path=converted_pdf,
                doc_id=doc_id,
                version_id=version_id,
                user_id=user_id,
                filename=filename,
                mime_type=mime_type,
            )

        finally:

            # -------------------------------------------------
            # 3. Cleanup
            # -------------------------------------------------

            if (
                converted_pdf is not None
                and not self.keep_converted_pdf
            ):

                converted_pdf.unlink(
                    missing_ok=True
                )

                # Remove the temporary directory
                # created by LibreOffice conversion.
                try:
                    converted_pdf.parent.rmdir()
                except OSError:
                    pass

    # =========================================================
    # LIBREOFFICE
    # =========================================================

    def _convert_to_pdf(
        self,
        docx_path: Path,
    ) -> Path:

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="rag_docx_"
            )
        )

        output_name = (
            docx_path.stem
            + ".pdf"
        )

        expected_pdf = (
            temp_dir
            / output_name
        )

        command = [
            self.libreoffice_path,

            "--headless",

            "--convert-to",
            "pdf",

            "--outdir",
            str(temp_dir),

            str(docx_path),
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.conversion_timeout,
                check=False,
            )

        except FileNotFoundError as exc:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

            raise RuntimeError(
                "LibreOffice was not found. "
                "Install LibreOffice and ensure "
                "'soffice' is available in PATH."
            ) from exc

        except subprocess.TimeoutExpired as exc:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

            raise RuntimeError(
                "DOCX → PDF conversion timed out."
            ) from exc

        if result.returncode != 0:

            stderr = (
                result.stderr.strip()
                or result.stdout.strip()
                or "unknown LibreOffice error"
            )

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

            raise RuntimeError(
                f"LibreOffice failed to convert DOCX: "
                f"{stderr}"
            )

        if not expected_pdf.exists():

            # LibreOffice may normalize the output
            # name. Find the generated PDF.
            pdf_files = list(
                temp_dir.glob("*.pdf")
            )

            if len(pdf_files) != 1:

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True,
                )

                raise RuntimeError(
                    "LibreOffice completed, but "
                    "the converted PDF could not be found."
                )

            expected_pdf = pdf_files[0]

        if expected_pdf.stat().st_size == 0:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

            raise RuntimeError(
                "LibreOffice generated an empty PDF."
            )

        return expected_pdf

    # =========================================================
    # PDF PARSING
    # =========================================================

    def _parse_pdf(
        self,
        pdf_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        document = None

        try:

            document = pymupdf.open(
                pdf_path
            )

            # -------------------------------------------------
            # Collect font statistics first.
            # These are later used for heading detection.
            # -------------------------------------------------

            font_sizes = (
                self._collect_font_sizes(
                    document
                )
            )

            body_font_size = (
                median(font_sizes)
                if font_sizes
                else 11.0
            )

            heading_stack: list[
                tuple[int, str]
            ] = []

            elements: list[
                ParsedElement
            ] = []

            order_counter = 0

            paragraph_counter = 0

            image_counter = 0

            # -------------------------------------------------
            # Iterate pages
            # -------------------------------------------------

            for page_index in range(
                len(document)
            ):

                page = document[
                    page_index
                ]

                page_number = (
                    page_index + 1
                )

                page_width = (
                    float(page.rect.width)
                )

                page_height = (
                    float(page.rect.height)
                )

                page_dict = page.get_text(
                    "dict",
                    sort=True,
                )

                blocks = page_dict.get(
                    "blocks",
                    []
                )

                if not isinstance(
                    blocks,
                    list,
                ):
                    continue

                for block_index, block in enumerate(
                    blocks
                ):

                    block_type = block.get(
                        "type"
                    )

                    # =================================================
                    # IMAGE BLOCK
                    # =================================================

                    if block_type == 1:

                        bbox = self._safe_bbox(
                            block.get(
                                "bbox"
                            )
                        )

                        if bbox is None:
                            continue

                        section_path = [
                            title
                            for _, title
                            in heading_stack
                        ]

                        element = (
                            self._build_image_element(
                                bbox=bbox,
                                page_number=page_number,
                                page_width=page_width,
                                page_height=page_height,
                                image_index=image_counter,
                                order=order_counter,
                                section_path=section_path,
                            )
                        )

                        elements.append(
                            element
                        )

                        order_counter += 1

                        image_counter += 1

                        continue

                    # =================================================
                    # TEXT BLOCK
                    # =================================================

                    if block_type != 0:
                        continue

                    lines = block.get(
                        "lines",
                        []
                    )

                    if not isinstance(
                        lines,
                        list,
                    ):
                        continue

                    line_clusters = (
                        self._split_block_into_clusters(
                            lines
                        )
                    )

                    if not line_clusters:
                        continue

                    block_bbox = self._safe_bbox(
                        block.get(
                            "bbox"
                        )
                    )

                    if block_bbox is None:

                        block_bbox = (
                            self._calculate_bbox_from_lines(
                                lines
                            )
                        )

                    for cluster_lines in line_clusters:

                        text = (
                            self._extract_block_text(
                                cluster_lines
                            )
                        )

                        if not text:
                            continue

                        bbox = (
                            self._calculate_bbox_from_lines(
                                cluster_lines
                            )
                        )

                        if bbox is None:

                            bbox = block_bbox

                        if bbox is None:
                            continue

                        cluster_spans = (
                            self._extract_spans(
                                cluster_lines
                            )
                        )

                        style = (
                            self._analyze_style(
                                cluster_spans
                            )
                        )

                        element_type = (
                            self._classify_block(
                                text=text,
                                style=style,
                                body_font_size=body_font_size,
                            )
                        )

                        # ---------------------------------------------
                        # Update section hierarchy
                        # ---------------------------------------------

                        section_path = (
                            self._update_section_path(
                                element_type=element_type,
                                text=text,
                                style=style,
                                heading_stack=heading_stack,
                            )
                        )

                        # ---------------------------------------------
                        # Source location
                        # ---------------------------------------------

                        location = (
                            SourceLocation(
                                type="docx_bbox",

                                page=page_number,

                                bbox={
                                    "x0": bbox[0],
                                    "y0": bbox[1],
                                    "x1": bbox[2],
                                    "y1": bbox[3],

                                    "coord_origin": (
                                        "TOPLEFT"
                                    ),
                                },

                                extra={
                                    "coordinate_system": (
                                        "pymupdf"
                                    ),

                                    "page_width": (
                                        page_width
                                    ),

                                    "page_height": (
                                        page_height
                                    ),

                                    "block_index": (
                                        block_index
                                    ),
                                },
                            )
                        )

                        # ---------------------------------------------
                        # Element metadata
                        # ---------------------------------------------

                        metadata = {
                            "page_number": page_number,

                            "block_index": block_index,

                            "paragraph_index": (
                                paragraph_counter
                            ),

                            "max_font_size": (
                                style[
                                    "max_font_size"
                                ]
                            ),

                            "average_font_size": (
                                style[
                                    "average_font_size"
                                ]
                            ),

                            "fonts": style[
                                "fonts"
                            ],

                            "is_bold": style[
                                "is_bold"
                            ],

                            "is_italic": style[
                                "is_italic"
                            ],

                            "is_underlined": style[
                                "is_underlined"
                            ],

                            "heading_level": style[
                                "heading_level"
                            ],

                            "line_count": len(
                                cluster_lines
                            ),

                            "span_count": len(
                                cluster_spans
                            ),

                            "page_width": (
                                page_width
                            ),

                            "page_height": (
                                page_height
                            ),
                        }

                        elements.extend(
                            self._build_text_elements(
                                text=text,
                                element_type=element_type,
                                section_path=section_path,
                                order=order_counter,
                                location=location,
                                metadata=metadata,
                            )
                        )

                        order_counter += 1

                        paragraph_counter += 1

            if not elements:

                raise ValueError(
                    "DocxParser found no extractable content."
                )

            for index, element in enumerate(
                elements,
                start=1,
            ):

                element.order = index

            metadata = (
                self._extract_document_metadata(
                    document=document
                )
            )

            metadata[
                "char_count"
            ] = sum(
                len(element.text)
                for element in elements
            )

            metadata[
                "element_counts"
            ] = dict(
                Counter(
                    element.element_type
                    for element in elements
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

                metadata=metadata,
            )

        finally:

            if document is not None:

                document.close()

    # =========================================================
    # FONT STATISTICS
    # =========================================================

    @staticmethod
    def _collect_font_sizes(
        document,
    ) -> list[float]:

        sizes: list[float] = []

        for page in document:

            data = page.get_text(
                "dict",
                sort=False,
            )

            blocks = data.get(
                "blocks",
                []
            )

            for block in blocks:

                if block.get("type") != 0:
                    continue

                for line in block.get(
                    "lines",
                    [],
                ):

                    for span in line.get(
                        "spans",
                        [],
                    ):

                        size = span.get(
                            "size"
                        )

                        if isinstance(
                            size,
                            (int, float),
                        ):

                            sizes.append(
                                float(size)
                            )

        return sizes

    # =========================================================
    # TEXT
    # =========================================================

    @staticmethod
    def _extract_block_text(
        lines: list[dict[str, Any]],
    ) -> str:

        result = []

        for line in lines:

            spans = line.get(
                "spans",
                []
            )

            line_text = ""

            for span in spans:

                text = span.get(
                    "text",
                    ""
                )

                if isinstance(
                    text,
                    str,
                ):

                    line_text += text

            line_text = (
                line_text.strip()
            )

            if line_text:

                result.append(
                    line_text
                )

        return "\n".join(
            result
        ).strip()

    @staticmethod
    def _extract_spans(
        lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        spans = []

        for line in lines:

            for span in line.get(
                "spans",
                [],
            ):

                if isinstance(
                    span,
                    dict,
                ):

                    spans.append(
                        span
                    )

        return spans

    @staticmethod
    def _split_block_into_clusters(
        lines: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:

        clusters: list[list[dict[str, Any]]] = []

        current: list[dict[str, Any]] | None = None

        ref_size = 0.0

        ref_bold = False

        for line in lines:

            if not isinstance(
                line,
                dict,
            ):
                continue

            max_size = 0.0

            bold = False

            for span in line.get(
                "spans",
                [],
            ):

                size = span.get(
                    "size"
                )

                if isinstance(
                    size,
                    (int, float),
                ):

                    max_size = max(
                        max_size,
                        float(size),
                    )

                flags = span.get(
                    "flags",
                    0,
                )

                if (
                    isinstance(
                        flags,
                        int,
                    )
                    and flags & 16
                ):

                    bold = True

            if current is None:

                current = [line]

                ref_size = max_size

                ref_bold = bold

                clusters.append(
                    current
                )

                continue

            tolerance = max(
                0.5,
                0.12 * ref_size,
            )

            same_cluster = bool(
                abs(max_size - ref_size)
                <= tolerance
                and bold == ref_bold
            )

            if same_cluster:

                current.append(
                    line
                )

            else:

                current = [line]

                ref_size = max_size

                ref_bold = bold

                clusters.append(
                    current
                )

        return [
            cluster
            for cluster in clusters
            if cluster
        ]

    # =========================================================
    # BBOX
    # =========================================================

    @staticmethod
    def _safe_bbox(
        bbox: Any,
    ) -> list[float] | None:

        if not isinstance(
            bbox,
            (list, tuple),
        ):

            return None

        if len(bbox) != 4:
            return None

        try:

            values = [
                float(value)
                for value in bbox
            ]

        except (
            TypeError,
            ValueError,
        ):

            return None

        x0, y0, x1, y1 = values

        if x1 < x0:
            x0, x1 = x1, x0

        if y1 < y0:
            y0, y1 = y1, y0

        return [
            x0,
            y0,
            x1,
            y1,
        ]

    @classmethod
    def _calculate_bbox_from_lines(
        cls,
        lines: list[dict[str, Any]],
    ) -> list[float] | None:

        boxes = []

        for line in lines:

            bbox = cls._safe_bbox(
                line.get("bbox")
            )

            if bbox is not None:
                boxes.append(bbox)

        if not boxes:
            return None

        return [
            min(
                box[0]
                for box in boxes
            ),

            min(
                box[1]
                for box in boxes
            ),

            max(
                box[2]
                for box in boxes
            ),

            max(
                box[3]
                for box in boxes
            ),
        ]

    # =========================================================
    # STYLE
    # =========================================================

    @staticmethod
    def _analyze_style(
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not spans:

            return {
                "max_font_size": 0.0,
                "average_font_size": 0.0,
                "fonts": [],
                "is_bold": False,
                "is_italic": False,
                "is_underlined": False,
                "heading_level": None,
            }

        sizes = []

        fonts = set()

        bold = False

        italic = False

        underlined = False

        for span in spans:

            size = span.get(
                "size"
            )

            if isinstance(
                size,
                (int, float),
            ):

                sizes.append(
                    float(size)
                )

            font = span.get(
                "font"
            )

            if isinstance(
                font,
                str,
            ):

                fonts.add(
                    font
                )

            flags = span.get(
                "flags",
                0
            )

            if isinstance(
                flags,
                int,
            ):

                # PyMuPDF font flags.
                # Bit 1 = italic
                # Bit 4 = bold
                if flags & 2:
                    italic = True

                if flags & 16:
                    bold = True

        if sizes:

            max_size = max(
                sizes
            )

            avg_size = (
                sum(sizes)
                / len(sizes)
            )

        else:

            max_size = 0.0

            avg_size = 0.0

        return {
            "max_font_size": max_size,

            "average_font_size": avg_size,

            "fonts": sorted(
                fonts
            ),

            "is_bold": bold,

            "is_italic": italic,

            "is_underlined": underlined,

            "heading_level": None,
        }

    # =========================================================
    # BLOCK CLASSIFICATION
    # =========================================================

    @staticmethod
    def _classify_block(
        text: str,
        style: dict[str, Any],
        body_font_size: float,
    ) -> str:

        text = text.strip()

        if not text:
            return "paragraph"

        font_size = style[
            "max_font_size"
        ]

        is_bold = style[
            "is_bold"
        ]

        # -----------------------------------------------------
        # List item
        # -----------------------------------------------------

        list_pattern = re.compile(
            r"^(?:"
            r"[•▪◦‣⁃]\s+"
            r"|[-*]\s+"
            r"|\d+[.)]\s+"
            r"|[A-Za-z][.)]\s+"
            r")"
        )

        if list_pattern.match(
            text
        ):

            return "list_item"

        # -----------------------------------------------------
        # Heading candidates
        # -----------------------------------------------------

        significantly_larger = (
            font_size
            >= body_font_size * 1.35
        )

        short_block = (
            len(text) <= 150
        )

        no_terminal_punctuation = (
            not text.endswith(
                (
                    ".",
                    ",",
                    ";",
                    ":",
                    "?",
                    "!",
                )
            )
        )

        if (
            significantly_larger
            and (
                is_bold
                or short_block
            )
        ):

            return "heading"

        if (
            is_bold
            and short_block
            and no_terminal_punctuation
            and font_size
            >= body_font_size * 1.15
        ):

            return "heading"

        return "paragraph"

    # =========================================================
    # HIERARCHY
    # =========================================================

    @staticmethod
    def _update_section_path(
        element_type: str,
        text: str,
        style: dict[str, Any],
        heading_stack: list[tuple[int, str]],
    ) -> list[str]:

        if element_type != "heading":

            return [
                title
                for _, title
                in heading_stack
            ]

        font_size = style[
            "max_font_size"
        ]

        # Infer heading depth.
        #
        # This is deliberately heuristic because we're
        # using the rendered DOCX/PDF, not the Word XML
        # semantic style information.
        if font_size >= 26:

            level = 1

        elif font_size >= 20:

            level = 2

        else:

            level = 3

        while (
            heading_stack
            and heading_stack[-1][0]
            >= level
        ):

            heading_stack.pop()

        heading_stack.append(
            (
                level,
                text.strip(),
            )
        )

        return [
            title
            for _, title
            in heading_stack
        ]

    # =========================================================
    # TEXT ELEMENT BUILDING
    # =========================================================

    @staticmethod
    def _build_text_elements(
        text: str,
        element_type: str,
        section_path: list[str],
        order: int,
        location: SourceLocation,
        metadata: dict[str, Any],
    ) -> list[ParsedElement]:

        if len(text) <= MAX_LINE_CHARS:

            return [
                ParsedElement(
                    element_id=(
                        f"docx_el_"
                        f"{uuid4().hex}"
                    ),
                    element_type=element_type,
                    text=text,
                    order=order,
                    section_path=list(
                        section_path
                    ),
                    parent_element_id=None,
                    locations=[
                        location
                    ],
                    metadata=metadata,
                )
            ]

        return DocxParser._split_long_text(
            text=text,
            element_type=element_type,
            section_path=section_path,
            order=order,
            location=location,
            metadata=metadata,
        )

    @staticmethod
    def _split_long_text(
        text: str,
        element_type: str,
        section_path: list[str],
        order: int,
        location: SourceLocation,
        metadata: dict[str, Any],
    ) -> list[ParsedElement]:

        spans = [
            (fragment, offset)
            for fragment, offset in split_long_line(
                text
            )
            if fragment
        ]

        fragment_count = len(
            spans
        )

        base_id = (
            f"docx_el_"
            f"{uuid4().hex}"
        )

        result: list[ParsedElement] = []

        for index, (fragment, _offset) in enumerate(
            spans
        ):

            fragment_meta = dict(
                metadata
            )

            fragment_meta[
                "fragment_of"
            ] = base_id

            fragment_meta[
                "fragment_index"
            ] = index

            fragment_meta[
                "fragment_count"
            ] = fragment_count

            new_location = (
                copy.copy(location)
                if index > 0
                else location
            )

            result.append(
                ParsedElement(
                    element_id=(
                        base_id
                        if index == 0
                        else f"docx_el_"
                        f"{uuid4().hex}"
                    ),
                    element_type=(
                        element_type
                        if index == 0
                        else "fragment"
                    ),
                    text=fragment,
                    order=order,
                    section_path=list(
                        section_path
                    ),
                    parent_element_id=None,
                    locations=[
                        new_location
                    ],
                    metadata=fragment_meta,
                )
            )

        return result

    # =========================================================
    # IMAGE ELEMENT
    # =========================================================

    @staticmethod
    def _build_image_element(
        bbox: list[float],
        page_number: int,
        page_width: float,
        page_height: float,
        image_index: int,
        order: int,
        section_path: list[str],
    ) -> ParsedElement:

        return ParsedElement(
            element_id=(
                f"docx_img_"
                f"{uuid4().hex}"
            ),

            element_type="image",

            text="",

            order=order,

            section_path=list(
                section_path
            ),

            parent_element_id=None,

            locations=[
                SourceLocation(
                    type="docx_bbox",

                    page=page_number,

                    bbox={
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],

                        "coord_origin": (
                            "TOPLEFT"
                        ),
                    },

                    extra={
                        "coordinate_system": (
                            "pymupdf"
                        ),

                        "page_width": (
                            page_width
                        ),

                        "page_height": (
                            page_height
                        ),

                        "image_index": (
                            image_index
                        ),
                    },
                )
            ],

            metadata={
                "page_number": page_number,

                "image_index": image_index,

                "page_width": page_width,

                "page_height": page_height,
            },
        )

    # =========================================================
    # DOCUMENT METADATA
    # =========================================================

    @staticmethod
    def _extract_document_metadata(
        document,
    ) -> dict[str, Any]:

        metadata = (
            document.metadata
            or {}
        )

        return {
            "format": "docx",

            "rendered_as": "pdf",

            "page_count": len(
                document
            ),

            "title": metadata.get(
                "title"
            ),

            "author": metadata.get(
                "author"
            ),

            "subject": metadata.get(
                "subject"
            ),

            "keywords": metadata.get(
                "keywords"
            ),

            "creator": metadata.get(
                "creator"
            ),

            "producer": metadata.get(
                "producer"
            ),

            "creation_date": metadata.get(
                "creationDate"
            ),

            "modification_date": metadata.get(
                "modDate"
            ),
        }



if __name__ == "__main__":
    import asyncio

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m app.ingestion.parsers.docx_parser <file.docx>")
            return
        file_path = Path(sys.argv[1])
        parser = DocxParser()
        document = await parser.parse(
            file_path=file_path,
            doc_id="test",
            version_id="test",
            user_id="test",
            filename=file_path.name,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        print("Parsed document:")
        print(document)
        
    asyncio.run(main())