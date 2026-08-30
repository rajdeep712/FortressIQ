from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.ingestion.parsers.base import DocumentParser


class PdfParser(DocumentParser):
    """
    Converts OpenDocumentLoader PDF JSON into
    our canonical ParsedDocument representation.

    It does NOT:
        - read the PDF
        - call OpenDocumentLoader
        - chunk
        - embed
        - store in Qdrant

    Its only job is normalization.
    """

    name = "open_document_loader_pdf"
    version = "1.0"

    def parse(
        self,
        data: dict[str, Any],
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        self._validate_response(data)

        elements: list[ParsedElement] = []

        visited_ids: set[str] = set()

        heading_stack: list[
            tuple[int, str]
        ] = []

        # --------------------------------------------------
        # Document metadata
        # --------------------------------------------------

        document_metadata = (
            self._extract_document_metadata(
                data
            )
        )

        root_nodes = data.get(
            "kids",
            []
        )

        if not isinstance(
            root_nodes,
            list,
        ):
            root_nodes = []

        # --------------------------------------------------
        # Iterative traversal
        # --------------------------------------------------

        stack: list[
            tuple[Any, str | None]
        ] = [
            (node, None)
            for node in reversed(root_nodes)
        ]

        while stack:

            node, parent_element_id = stack.pop()

            if not isinstance(
                node,
                dict,
            ):
                continue

            # ----------------------------------------------
            # Source ID
            # ----------------------------------------------

            source_id = node.get("id")

            if source_id is not None:

                source_id = str(
                    source_id
                )

                if source_id in visited_ids:
                    continue

                visited_ids.add(
                    source_id
                )

            # ----------------------------------------------
            # Basic information
            # ----------------------------------------------

            raw_type = node.get(
                "type",
                "unknown",
            )

            element_type = (
                self._normalize_type(
                    raw_type
                )
            )

            content = (
                self._extract_content(
                    node
                )
            )

            page = (
                self._extract_page(
                    node
                )
            )

            bbox = (
                self._extract_bbox(
                    node
                )
            )

            # ----------------------------------------------
            # Heading hierarchy
            # ----------------------------------------------

            current_section_path = (
                self._build_section_path(
                    node=node,
                    content=content,
                    heading_stack=heading_stack,
                )
            )

            # ----------------------------------------------
            # Determine whether this node should become
            # a ParsedElement.
            # ----------------------------------------------

            has_children = (
                self._has_children(node)
            )

            create_element = (
                bool(content.strip())
                or element_type
                in {
                    "image",
                    "caption",
                }
            )

            # Containers such as "list" and "text block"
            # should not generate duplicate text when their
            # real content is inside children.
            container_only = (
                has_children
                and element_type
                in {
                    "list",
                    "text_block",
                }
            )

            current_element_id = (
                parent_element_id
            )

            if (
                create_element
                and not container_only
            ):

                element_id = (
                    self._make_element_id(
                        source_id
                    )
                )

                locations = (
                    self._build_locations(
                        page=page,
                        bbox=bbox,
                        node=node,
                    )
                )

                element = ParsedElement(
                    element_id=element_id,

                    element_type=element_type,

                    text=content,

                    order=0,

                    section_path=(
                        current_section_path
                    ),

                    parent_element_id=(
                        parent_element_id
                    ),

                    locations=locations,

                    metadata=self._extract_metadata(
                        node
                    ),
                )

                elements.append(
                    element
                )

                current_element_id = (
                    element_id
                )

            # ----------------------------------------------
            # Nested kids and list_items
            # ----------------------------------------------

            children: list[Any] = []

            kids = node.get(
                "kids",
                [],
            )

            if isinstance(
                kids,
                list,
            ):
                children.extend(kids)

            list_items = node.get(
                "list_items",
            )

            if isinstance(
                list_items,
                list,
            ):
                children.extend(list_items)

            for child in reversed(children):

                stack.append(
                    (
                        child,
                        current_element_id,
                    )
                )

        # --------------------------------------------------
        # Clean result
        # --------------------------------------------------

        elements = [
            element
            for element in elements
            if (
                element.text.strip()
                or element.element_type
                in {
                    "image",
                    "caption",
                }
            )
        ]

        for index, element in enumerate(
            elements,
            start=1,
        ):
            element.order = index

        return ParsedDocument(
            doc_id=doc_id,

            version_id=version_id,

            user_id=user_id,

            filename=filename,

            mime_type=mime_type,

            parser_name=self.name,

            parser_version=self.version,

            elements=elements,

            metadata=document_metadata,
        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def _validate_response(
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "OpenDocumentLoader response "
                "must be a JSON object."
            )

        if "kids" not in data:

            raise ValueError(
                "OpenDocumentLoader response does not "
                "contain the expected 'kids' field."
            )

    # ======================================================
    # Document metadata
    # ======================================================

    @staticmethod
    def _extract_document_metadata(
        data: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "source_file_name": data.get(
                "file name"
            ),

            "page_count": data.get(
                "number of pages"
            ),

            "author": data.get(
                "author"
            ),

            "title": data.get(
                "title"
            ),

            "creation_date": data.get(
                "creation date"
            ),

            "modification_date": data.get(
                "modification date"
            ),
        }

    # ======================================================
    # Content
    # ======================================================

    @staticmethod
    def _extract_content(
        node: dict[str, Any],
    ) -> str:

        content = node.get(
            "content"
        )

        if content is None:
            return ""

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        return str(
            content
        ).strip()

    # ======================================================
    # Page
    # ======================================================

    @staticmethod
    def _extract_page(
        node: dict[str, Any],
    ) -> int | None:

        value = node.get(
            "page number"
        )

        if value is None:
            value = node.get(
                "page_number"
            )

        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ======================================================
    # Bounding box
    # ======================================================

    @staticmethod
    def _extract_bbox(
        node: dict[str, Any],
    ) -> list[float] | None:

        bbox = node.get(
            "bounding box"
        )

        if bbox is None:
            bbox = node.get(
                "bounding_box"
            )

        if not isinstance(
            bbox,
            (list, tuple),
        ):
            return None

        if len(bbox) != 4:
            return None

        try:

            return [
                float(value)
                for value in bbox
            ]

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ======================================================
    # Location
    # ======================================================

    @staticmethod
    def _build_locations(
        page: int | None,
        bbox: list[float] | None,
        node: dict[str, Any],
    ) -> list[SourceLocation]:

        locations = []

        if (
            page is not None
            and bbox is not None
        ):

            locations.append(
                SourceLocation(
                    type="pdf_bbox",

                    page=page,

                    bbox={
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],

                        # Preserve the original
                        # OpenDocumentLoader values.
                        "raw": bbox,
                    },

                    extra={
                        "coordinate_system": (
                            "pdf"
                        ),

                        "raw_bbox_key": (
                            "bounding box"
                        ),
                    },
                )
            )

        # --------------------------------------------------
        # Optional character span
        # --------------------------------------------------

        charspan = node.get(
            "charspan"
        )

        if charspan is None:
            charspan = node.get(
                "char_span"
            )

        if (
            isinstance(
                charspan,
                (list, tuple),
            )
            and len(charspan) == 2
        ):

            try:

                start = int(
                    charspan[0]
                )

                end = int(
                    charspan[1]
                )

                if locations:

                    locations[0].char_start = (
                        start
                    )

                    locations[0].char_end = (
                        end
                    )

                else:

                    locations.append(
                        SourceLocation(
                            type="text_offset",

                            char_start=start,

                            char_end=end,
                        )
                    )

            except (
                TypeError,
                ValueError,
            ):
                pass

        return locations

    # ======================================================
    # Type normalization
    # ======================================================

    @staticmethod
    def _normalize_type(
        raw_type: Any,
    ) -> str:

        value = str(
            raw_type
        ).strip().lower()

        mapping = {
            "heading": "heading",

            "paragraph": "paragraph",

            "p": "paragraph",

            "list": "list",

            "list item": "list_item",

            "list_item": "list_item",

            "image": "image",

            "figure": "image",

            "caption": "caption",

            "text block": "text_block",

            "text_block": "text_block",

            "table": "table",

            "table row": "table_row",

            "table_row": "table_row",

            "cell": "table_cell",
        }

        return mapping.get(
            value,
            value,
        )

    # ======================================================
    # Heading hierarchy
    # ======================================================

    @staticmethod
    def _build_section_path(
        node: dict[str, Any],
        content: str,
        heading_stack: list[
            tuple[int, str]
        ],
    ) -> list[str]:

        node_type = (
            str(
                node.get(
                    "type",
                    ""
                )
            )
            .strip()
            .lower()
        )

        if node_type != "heading":

            return [
                title
                for _, title
                in heading_stack
            ]

        level = node.get(
            "heading level"
        )

        if level is None:
            level = node.get(
                "heading_level"
            )

        try:

            level = int(level)

        except (
            TypeError,
            ValueError,
        ):

            return [
                title
                for _, title
                in heading_stack
            ]

        # Remove current/deeper levels.
        while (
            heading_stack
            and heading_stack[-1][0]
            >= level
        ):

            heading_stack.pop()

        if content:

            heading_stack.append(
                (
                    level,
                    content,
                )
            )

        return [
            title
            for _, title
            in heading_stack
        ]

    # ======================================================
    # Children
    # ======================================================

    @staticmethod
    def _has_children(
        node: dict[str, Any],
    ) -> bool:

        kids = node.get(
            "kids"
        )

        if isinstance(
            kids,
            list,
        ) and kids:

            return True

        list_items = node.get(
            "list_items"
        )

        if isinstance(
            list_items,
            list,
        ) and list_items:

            return True

        return False

    # ======================================================
    # IDs
    # ======================================================

    @staticmethod
    def _make_element_id(
        source_id: str | None,
    ) -> str:

        if source_id is not None:
            return (
                f"pdf_el_{source_id}"
            )

        return (
            "pdf_el_"
            + uuid4().hex
        )

    # ======================================================
    # Metadata
    # ======================================================

    @staticmethod
    def _extract_metadata(
        node: dict[str, Any],
    ) -> dict[str, Any]:

        metadata = {}

        useful_fields = (
            "pdfua_tag",
            "level",
            "heading level",
            "font",
            "font size",
            "text color",
            "alt_source",
            "source",
            "numbering style",
            "next list id",
            "previous list id",
        )

        for key in useful_fields:

            if key in node:

                metadata[key] = node[key]

        # Keep these explicitly because they are useful
        # when debugging provenance.
        metadata["source_type"] = node.get(
            "type"
        )

        metadata["source_id"] = node.get(
            "id"
        )

        return metadata