from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any
from uuid import uuid4

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


class PptxParser(DocumentParser):
    """
    Normalizes Docling PPTX JSON into our canonical
    ParsedDocument representation.

    This parser:
        - does NOT call Docling
        - does NOT read PPTX directly
        - does NOT chunk
        - does NOT embed

    It only converts Docling JSON -> ParsedDocument.
    """

    name = "docling_pptx"
    version = "1.1"

    # =========================================================
    # PUBLIC API
    # =========================================================

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

        document = data["document"]

        resolver = _DoclingResolver(
            document
        )

        elements: list[ParsedElement] = []

        visited: set[str] = set()

        order_counter = 0

        # -----------------------------------------------------
        # Document metadata
        # -----------------------------------------------------

        document_metadata = (
            self._extract_document_metadata(
                document
            )
        )

        # -----------------------------------------------------
        # Traverse body iteratively
        # -----------------------------------------------------

        body = document.get(
            "body",
            {}
        )

        body_children = body.get(
            "children",
            []
        )

        if not isinstance(
            body_children,
            list,
        ):

            body_children = []

        stack = [
            (
                reference,
                None,  # parent_element_id
                None,  # inherited_slide
                [],    # inherited_section_path
            )
            for reference in reversed(
                body_children
            )
        ]

        order_counter = self._drain_stack(
            stack=stack,
            resolver=resolver,
            elements=elements,
            visited=visited,
            order_counter=order_counter,
        )

        # -----------------------------------------------------
        # Safety fallback
        # -----------------------------------------------------
        #
        # If the body did not expose everything, inspect
        # slides directly.
        # -----------------------------------------------------

        if not elements:

            groups = document.get(
                "groups",
                []
            )

            if isinstance(
                groups,
                list,
            ):

                stack = []

                for group in groups:

                    if not isinstance(
                        group,
                        dict,
                    ):
                        continue

                    name = group.get(
                        "name",
                        ""
                    )

                    if self._is_slide_group(
                        name
                    ):

                        stack.append(
                            (
                                {
                                    "$ref": group.get(
                                        "self_ref"
                                    ),
                                },
                                None,
                                None,
                                [],
                            )
                        )

                order_counter = self._drain_stack(
                    stack=list(
                        reversed(stack)
                    ),
                    resolver=resolver,
                    elements=elements,
                    visited=visited,
                    order_counter=order_counter,
                )

        if not elements:

            raise ValueError(
                "Docling response contains no extractable slide content."
            )

        document_metadata[
            "char_count"
        ] = sum(
            len(element.text)
            for element in elements
        )

        document_metadata[
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
            metadata=document_metadata,
        )

    # =========================================================
    # Iterative traversal
    # =========================================================

    def _drain_stack(
        self,
        stack: list[
            tuple[
                Any,
                str | None,
                int | None,
                list[str],
            ]
        ],
        resolver: "_DoclingResolver",
        elements: list[ParsedElement],
        visited: set[str],
        order_counter: int,
    ) -> int:

        while stack:

            (
                reference,
                parent_element_id,
                inherited_slide,
                inherited_section_path,
            ) = stack.pop()

            resolved = self._resolve_node(
                reference=reference,
                resolver=resolver,
                visited=visited,
                inherited_slide=inherited_slide,
                inherited_section_path=inherited_section_path,
            )

            if resolved is None:

                continue

            (
                node,
                source_ref,
                current_slide,
                section_path,
                element_type,
                text,
            ) = resolved

            element_id = parent_element_id

            new_elements = self._build_elements(
                node=node,
                source_ref=source_ref,
                element_type=element_type,
                text=text,
                section_path=section_path,
                parent_element_id=parent_element_id,
                current_slide=current_slide,
            )

            if new_elements:

                element_id = (
                    new_elements[0].element_id
                )

                for element in new_elements:

                    order_counter += 1

                    element.order = order_counter

                    elements.append(
                        element
                    )

            children = node.get(
                "children",
                []
            )

            if isinstance(
                children,
                list,
            ):

                for child_reference in reversed(
                    children
                ):

                    stack.append(
                        (
                            child_reference,
                            element_id,
                            current_slide,
                            section_path,
                        )
                    )

        return order_counter

    def _resolve_node(
        self,
        reference: Any,
        resolver: "_DoclingResolver",
        visited: set[str],
        inherited_slide: int | None,
        inherited_section_path: list[str],
    ) -> (
        tuple[
            dict[str, Any],
            str | None,
            int | None,
            list[str],
            str,
            str,
        ]
        | None
    ):

        node = resolver.resolve(
            reference
        )

        if node is None:
            return None

        source_ref = node.get(
            "self_ref"
        )

        if source_ref:

            if source_ref in visited:
                return None

            visited.add(
                source_ref
            )

        # -----------------------------------------------------
        # Determine slide
        # -----------------------------------------------------

        current_slide = (
            self._extract_slide_number(
                node
            )
            or inherited_slide
        )

        # A group called slide-0 means slide 1.
        current_slide = (
            current_slide
            or self._slide_from_group_name(
                node.get("name")
            )
            or inherited_slide
        )

        # -----------------------------------------------------
        # Determine section
        # -----------------------------------------------------

        section_path = list(
            inherited_section_path
        )

        element_type = (
            self._normalize_element_type(
                node
            )
        )

        text = self._extract_text(
            node
        )

        # A PPTX title is a natural section boundary.
        if element_type == "title":

            if text:

                section_path = [
                    *section_path,
                    text,
                ]

        # For an ordinary element, inherit the current
        # slide title.
        elif current_slide is not None:

            section_path = list(
                inherited_section_path
            )

        return (
            node,
            source_ref,
            current_slide,
            section_path,
            element_type,
            text,
        )

    # =========================================================
    # Convert a resolved node into elements
    # =========================================================

    def _build_elements(
        self,
        node: dict[str, Any],
        source_ref: str | None,
        element_type: str,
        text: str,
        section_path: list[str],
        parent_element_id: str | None,
        current_slide: int | None,
    ) -> list[ParsedElement]:

        if not self._should_create_element(
            node=node,
            text=text,
            element_type=element_type,
        ):

            return []

        base_id = self._make_element_id(
            source_ref
        )

        locations = (
            self._build_locations(
                node=node,
                slide=current_slide,
            )
        )

        metadata = (
            self._extract_metadata(
                node
            )
        )

        # Add slide information to metadata.
        if current_slide is not None:

            metadata[
                "slide_number"
            ] = current_slide

        # -----------------------------------------------------
        # Long text -> lossless fragments
        # -----------------------------------------------------

        if len(text) <= MAX_LINE_CHARS:

            return [
                ParsedElement(
                    element_id=base_id,
                    element_type=element_type,
                    text=text,
                    order=0,
                    section_path=section_path,
                    parent_element_id=(
                        parent_element_id
                    ),
                    locations=locations,
                    metadata=metadata,
                )
            ]

        return self._split_text_fragments(
            node=node,
            text=text,
            element_type=element_type,
            section_path=section_path,
            parent_element_id=parent_element_id,
            base_id=base_id,
            locations=locations,
            metadata=metadata,
        )

    @staticmethod
    def _split_text_fragments(
        node: dict[str, Any],
        text: str,
        element_type: str,
        section_path: list[str],
        parent_element_id: str | None,
        base_id: str,
        locations: list[SourceLocation],
        metadata: dict[str, Any],
    ) -> list[ParsedElement]:

        base_span = PptxParser._first_charspan(
            node
        )

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

        result: list[ParsedElement] = []

        for index, (fragment, offset) in enumerate(
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

            fragment_locations = []

            for loc in locations:

                new_loc = copy.copy(
                    loc
                )

                if base_span is not None:

                    new_loc.char_start = (
                        base_span[0] + offset
                    )

                    new_loc.char_end = (
                        base_span[0]
                        + offset
                        + len(fragment)
                    )

                else:

                    new_loc.char_start = None
                    new_loc.char_end = None

                fragment_locations.append(
                    new_loc
                )

            result.append(
                ParsedElement(
                    element_id=(
                        base_id
                        if index == 0
                        else f"pptx_el_{uuid4().hex}"
                    ),
                    element_type=(
                        element_type
                        if index == 0
                        else "fragment"
                    ),
                    text=fragment,
                    order=0,
                    section_path=section_path,
                    parent_element_id=(
                        parent_element_id
                    ),
                    locations=fragment_locations,
                    metadata=fragment_meta,
                )
            )

        return result

    @staticmethod
    def _first_charspan(
        node: dict[str, Any],
    ) -> tuple[int, int] | None:

        provenance = node.get(
            "prov",
            []
        )

        if not isinstance(
            provenance,
            list,
        ):

            return None

        for prov in provenance:

            if not isinstance(
                prov,
                dict,
            ):
                continue

            charspan = prov.get(
                "charspan"
            )

            if not isinstance(
                charspan,
                (
                    list,
                    tuple,
                ),
            ) or len(charspan) != 2:

                continue

            try:

                start = int(
                    charspan[0]
                )

                end = int(
                    charspan[1]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            return (
                start,
                end,
            )

        return None

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _validate_response(
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Docling response must be a JSON object."
            )

        document = data.get(
            "document"
        )

        if not isinstance(
            document,
            dict,
        ):

            raise ValueError(
                "Docling response contains no valid document."
            )

    # =========================================================
    # Document metadata
    # =========================================================

    @staticmethod
    def _extract_document_metadata(
        document: dict[str, Any],
    ) -> dict[str, Any]:

        origin = document.get(
            "origin",
            {}
        )

        pages = document.get(
            "pages",
            {}
        )

        page_metadata = []

        if isinstance(
            pages,
            dict,
        ):

            for key, page in pages.items():

                if not isinstance(
                    page,
                    dict,
                ):
                    continue

                size = page.get(
                    "size",
                    {}
                )

                page_metadata.append(
                    {
                        "slide_number": page.get(
                            "page_no"
                        ),
                        "width": size.get(
                            "width"
                        )
                        if isinstance(
                            size,
                            dict,
                        )
                        else None,
                        "height": size.get(
                            "height"
                        )
                        if isinstance(
                            size,
                            dict,
                        )
                        else None,
                    }
                )

        return {
            "schema_name": document.get(
                "schema_name"
            ),
            "schema_version": document.get(
                "version"
            ),
            "document_name": document.get(
                "name"
            ),
            "origin": {
                "filename": origin.get(
                    "filename"
                ),
                "mimetype": origin.get(
                    "mimetype"
                ),
                "binary_hash": origin.get(
                    "binary_hash"
                ),
            },
            "slide_count": len(
                page_metadata
            ),
            "slides": page_metadata,
        }

    # =========================================================
    # Element type
    # =========================================================

    @staticmethod
    def _normalize_element_type(
        node: dict[str, Any],
    ) -> str:

        label = str(
            node.get(
                "label",
                "unknown",
            )
        ).strip().lower()

        name = str(
            node.get(
                "name",
                "",
            )
        ).strip().lower()

        mapping = {
            "title": "title",

            "paragraph": "paragraph",

            "list_item": "list_item",

            "list": "list",

            "picture": "image",

            "image": "image",

            "table": "table",

            "caption": "caption",

            "chapter": "slide",
        }

        if label in mapping:
            return mapping[label]

        if name.startswith(
            "slide-"
        ):
            return "slide"

        return label

    # =========================================================
    # Text
    # =========================================================

    @staticmethod
    def _extract_text(
        node: dict[str, Any],
    ) -> str:

        for key in (
            "text",
            "orig",
            "content",
        ):

            value = node.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                return value.strip()

        return ""

    # =========================================================
    # Slide number
    # =========================================================

    @staticmethod
    def _extract_slide_number(
        node: dict[str, Any],
    ) -> int | None:

        # Direct page_no is the preferred source.
        prov = node.get(
            "prov"
        )

        if isinstance(
            prov,
            list,
        ):

            for provenance in prov:

                if not isinstance(
                    provenance,
                    dict,
                ):
                    continue

                page_no = provenance.get(
                    "page_no"
                )

                if page_no is not None:

                    try:
                        return int(
                            page_no
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

        return None

    # =========================================================
    # slide-0 -> 1
    # =========================================================

    @staticmethod
    def _slide_from_group_name(
        name: Any,
    ) -> int | None:

        if not isinstance(
            name,
            str,
        ):
            return None

        match = re.match(
            r"slide-(\d+)",
            name.lower()
        )

        if not match:
            return None

        return int(
            match.group(1)
        ) + 1

    # =========================================================
    # Bounding boxes
    # =========================================================

    @staticmethod
    def _build_locations(
        node: dict[str, Any],
        slide: int | None,
    ) -> list[SourceLocation]:

        locations = []

        provenance = node.get(
            "prov",
            []
        )

        if not isinstance(
            provenance,
            list,
        ):
            return locations

        for prov in provenance:

            if not isinstance(
                prov,
                dict,
            ):
                continue

            prov_slide = (
                prov.get(
                    "page_no"
                )
                or slide
            )

            bbox = prov.get(
                "bbox"
            )

            charspan = prov.get(
                "charspan"
            )

            bbox_dict = None

            if isinstance(
                bbox,
                dict,
            ):

                try:

                    bbox_dict = {
                        "l": float(
                            bbox["l"]
                        ),
                        "t": float(
                            bbox["t"]
                        ),
                        "r": float(
                            bbox["r"]
                        ),
                        "b": float(
                            bbox["b"]
                        ),
                    }

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):

                    bbox_dict = None

                if bbox_dict is not None:

                    bbox_dict[
                        "coord_origin"
                    ] = bbox.get(
                        "coord_origin"
                    )

            if bbox_dict is not None:

                location = SourceLocation(
                    type="pptx_bbox",

                    slide=(
                        int(prov_slide)
                        if prov_slide is not None
                        else None
                    ),

                    bbox=bbox_dict,

                    extra={
                        "coordinate_system": (
                            "pptx"
                        ),

                        "coord_origin": bbox.get(
                            "coord_origin"
                        ),
                    },
                )

                # Character span
                if (
                    isinstance(
                        charspan,
                        (
                            list,
                            tuple,
                        ),
                    )
                    and len(charspan) == 2
                ):

                    try:

                        location.char_start = (
                            int(charspan[0])
                        )

                        location.char_end = (
                            int(charspan[1])
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                locations.append(
                    location
                )

            elif (
                isinstance(
                    charspan,
                    (
                        list,
                        tuple,
                    ),
                )
                and len(charspan) == 2
            ):

                try:

                    locations.append(
                        SourceLocation(
                            type="text_offset",

                            slide=(
                                int(prov_slide)
                                if prov_slide is not None
                                else None
                            ),

                            char_start=int(
                                charspan[0]
                            ),

                            char_end=int(
                                charspan[1]
                            ),
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return locations

    # =========================================================
    # Decide whether node becomes an element
    # =========================================================

    @staticmethod
    def _should_create_element(
        node: dict[str, Any],
        text: str,
        element_type: str,
    ) -> bool:

        if text:
            return True

        if element_type in {
            "image",
            "table",
        }:

            return True

        # Do not create empty containers as content.
        return False

    # =========================================================
    # Metadata
    # =========================================================

    @staticmethod
    def _extract_metadata(
        node: dict[str, Any],
    ) -> dict[str, Any]:

        metadata: dict[str, Any] = {}

        for key in (
            "label",
            "name",
            "self_ref",
            "content_layer",
            "formatting",
            "hyperlink",
            "enumerated",
            "marker",
        ):

            if key in node:

                metadata[key] = node[key]

        # Preserve all provenance references,
        # but don't duplicate the entire node.
        provenance = node.get(
            "prov"
        )

        if isinstance(
            provenance,
            list,
        ):

            metadata[
                "provenance_count"
            ] = len(provenance)

        return metadata

    # =========================================================
    # Element ID
    # =========================================================

    @staticmethod
    def _make_element_id(
        source_ref: str | None,
    ) -> str:

        if source_ref:

            safe_ref = source_ref.replace(
                "#/",
                ""
            ).replace(
                "/",
                "_"
            )

            return (
                f"pptx_el_{safe_ref}"
            )

        return (
            f"pptx_el_{uuid4().hex}"
        )

    # =========================================================
    # Slide detection
    # =========================================================

    @staticmethod
    def _is_slide_group(
        name: Any,
    ) -> bool:

        if not isinstance(
            name,
            str,
        ):
            return False

        return bool(
            re.match(
                r"slide-\d+",
                name.lower()
            )
        )


# =============================================================
# Docling reference resolver
# =============================================================


class _DoclingResolver:
    """
    Resolves Docling references such as:

        {"$ref": "#/texts/7"}

    into their actual objects.
    """

    def __init__(
        self,
        document: dict[str, Any],
    ):

        self.document = document

        self.index: dict[
            str,
            dict[str, Any]
        ] = {}

        self._build_index()

    def _build_index(
        self
    ):

        self._index_collection(
            self.document.get(
                "groups",
                []
            )
        )

        self._index_collection(
            self.document.get(
                "texts",
                []
            )
        )

        self._index_collection(
            self.document.get(
                "pictures",
                []
            )
        )

        self._index_collection(
            self.document.get(
                "tables",
                []
            )
        )

        self._index_collection(
            self.document.get(
                "key_value_items",
                []
            )
        )

        self._index_collection(
            self.document.get(
                "form_items",
                []
            )
        )

    def _index_collection(
        self,
        collection: Any,
    ):

        if not isinstance(
            collection,
            list,
        ):
            return

        for item in collection:

            if not isinstance(
                item,
                dict,
            ):
                continue

            self_ref = item.get(
                "self_ref"
            )

            if isinstance(
                self_ref,
                str,
            ):

                self.index[
                    self_ref
                ] = item

    def resolve(
        self,
        reference: Any,
    ) -> dict[str, Any] | None:

        # Already a full object.
        if isinstance(
            reference,
            dict
        ) and "self_ref" in reference:

            return reference

        if not isinstance(
            reference,
            dict,
        ):
            return None

        ref = reference.get(
            "$ref"
        )

        if not isinstance(
            ref,
            str,
        ):
            return None

        return self.index.get(
            ref
        )