import re
from uuid import uuid4

from app.core.config import settings
from app.ingestion.chunker.base import (
    Block,
    Chunk,
)
from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
)

NUMBERED_RE = re.compile(
    r"^\s*\d{1,2}[.)]\s"
)


class PdfChunker(SectionChunker):
    """
    PDF: structure-driven chunking over the OpenDocumentLoader tree.

    The parser already wires the ODL hierarchy through
    ``ParsedElement.parent_element_id`` (empty wrapper nodes such as
    ``list`` / ``text block`` emit no element, so their real contents
    hang directly off the owning element). This chunker makes the
    tree explicit:

        Major item (root-level element with kids, or a heading, or a
        numbered item, or a top-level list item)
         └── one parent chunk  (own text + descendant texts)
             └── one child chunk per descendant (no char-window packing)

    Consecutive plain root-level leaves (no kids, not a heading, not
    numbered) are merged into a single "unsectioned" run parent so
    filler paragraphs don't each become their own thin parent.

    Parents hold the whole item even when it exceeds the soft cap; the
    20k hard cap remains a truncation/split fallback only. Tiny
    adjacent children (< ``pdf_merge_tiny_child_chars``) are folded
    into their neighbour.

    Setting ``pdf_chunk_structured`` to false restores the previous
    SectionChunker (window packing) behaviour.
    """

    identifier = "pdf"

    atomic_element_types = frozenset()

    HEADING_TYPES = SectionChunker.HEADING_TYPES

    def __init__(
        self,
        *,
        structured: bool | None = None,
        sort_siblings_by_bbox: bool | None = None,
        merge_tiny_child_chars: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.structured = (
            settings.pdf_chunk_structured
            if structured is None
            else structured
        )
        self.sort_siblings_by_bbox = (
            settings.pdf_sort_siblings_by_bbox
            if sort_siblings_by_bbox is None
            else sort_siblings_by_bbox
        )
        self.merge_tiny_child_chars = (
            settings.pdf_merge_tiny_child_chars
            if merge_tiny_child_chars is None
            else merge_tiny_child_chars
        )

        # Structured parents are whole items (no soft-cap splitting);
        # the legacy Section strategy still soft-splits oversized runs.
        self.respect_parent_soft_cap = (
            not self.structured
        )

    # ---------------------------------------------------------------
    # Block grouping (tree)
    # ---------------------------------------------------------------

    def _blocks(
        self,
        document: ParsedDocument,
    ) -> list[Block]:

        if not self.structured:
            return SectionChunker._blocks(
                self,
                document,
            )

        elements = document.elements

        children_of: dict[
            str | None,
            list[ParsedElement],
        ] = {}

        for element in elements:
            children_of.setdefault(
                element.parent_element_id,
                [],
            ).append(element)

        blocks: list[Block] = []
        pending_run: list[ParsedElement] = []

        for element in elements:

            if element.parent_element_id is not None:
                continue

            kids = children_of.get(
                element.element_id,
                [],
            )

            if self._is_major(
                element,
                kids,
            ):
                if pending_run:
                    blocks.append(
                        self._make_run_block(
                            pending_run,
                        )
                    )
                    pending_run = []

                blocks.append(
                    self._make_major_block(
                        element,
                        children_of,
                    )
                )
            else:
                pending_run.append(element)

        if pending_run:
            blocks.append(
                self._make_run_block(
                    pending_run,
                )
            )

        return blocks

    def block_is_seam(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> bool:
        # Unstructured PDF keeps heading-section structure. Structured
        # PDF exposes every major item as its own content block, so
        # those blocks coalesce through `_parent_groups` into banded
        # parents (4-8 children) instead of each becoming a one-child
        # wafer parent. Item boundaries survive as child boundaries.
        # Tables are always hard seams: each table is its own parent,
        # never coalesced with neighbouring runs.
        if (
            block.key
            and block.key[0] == "table"
        ):
            return True

        if not self.structured:
            return bool(
                block.key
                and block.key != ("root",)
            )
        return False

    def _is_major(
        self,
        element: ParsedElement,
        kids: list[ParsedElement],
    ) -> bool:

        if (
            element.element_type == "table"
        ):
            return True

        if kids:
            return True

        if (
            element.element_type
            in self.HEADING_TYPES
        ):
            return True

        if (
            element.metadata.get(
                "heading level",
            ) is not None
        ):
            return True

        if NUMBERED_RE.match(
            element.text.strip(),
        ):
            return True

        if element.element_type == "list_item":
            return True

        return False

    def _make_major_block(
        self,
        root: ParsedElement,
        children_of: dict[
            str | None,
            list[ParsedElement],
        ],
    ) -> Block:

        kids = self._collect_descendants(
            root.element_id,
            children_of,
        )

        if not kids:
            kids = [root]

        if self.sort_siblings_by_bbox:
            kids = sorted(
                kids,
                key=self._reading_position,
            )

        is_table = (
            root.element_type == "table"
        )

        # Table rows are meaningful children regardless of length,
        # so do not fold short rows into their neighbours.
        if is_table:
            elements = kids
        else:
            elements = self._fold_tiny_kids(
                kids,
            )

        meta = {
            "kind": (
                "table"
                if is_table
                else "major"
            ),
            "root_element_id": (
                root.element_id
            ),
            "root_element_type": (
                root.element_type
            ),
        }

        if is_table:

            root_metadata = (
                root.metadata or {}
            )

            if (
                "number of rows"
                in root_metadata
            ):

                meta["table_rows"] = (
                    root_metadata[
                        "number of rows"
                    ]
                )

            if (
                "number of columns"
                in root_metadata
            ):

                meta["table_columns"] = (
                    root_metadata[
                        "number of columns"
                    ]
                )

        return Block(
            key=(
                ("table", root)
                if is_table
                else (
                    "major",
                    root,
                )
            ),
            elements=elements,
            meta=meta,
        )

    def _make_run_block(
        self,
        run: list[ParsedElement],
    ) -> Block:

        elements = list(run)

        if self.sort_siblings_by_bbox:
            elements = sorted(
                elements,
                key=self._reading_position,
            )

        return Block(
            key=("run",),
            elements=elements,
            meta={
                "kind": "run",
            },
        )

    # ---------------------------------------------------------------
    # Tree helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _collect_descendants(
        root_id: str,
        children_of: dict[
            str | None,
            list[ParsedElement],
        ],
    ) -> list[ParsedElement]:

        result: list[ParsedElement] = []

        stack: list[ParsedElement] = list(
            reversed(
                children_of.get(
                    root_id,
                    [],
                )
            )
        )

        while stack:

            element = stack.pop()

            result.append(element)

            sub = children_of.get(
                element.element_id,
                [],
            )

            for child in reversed(sub):
                stack.append(child)

        return result

    def _fold_tiny_kids(
        self,
        kids: list[ParsedElement],
    ) -> list[ParsedElement]:

        threshold = (
            self.merge_tiny_child_chars
        )

        if (
            threshold <= 0
            or len(kids) < 2
        ):
            return kids

        folded: list[ParsedElement] = []
        pending: list[str] = []

        for element in kids:

            text = element.text.strip()

            if not text:
                continue

            if len(text) < threshold:
                pending.append(element.text)
                continue

            if pending:
                element.text = (
                    "\n\n".join(
                        pending
                        + [element.text]
                    )
                )
                pending = []

            folded.append(element)

        if pending:
            if folded:
                folded[-1].text = (
                    folded[-1].text
                    + "\n\n"
                    + "\n\n".join(pending)
                )
            else:
                # Everything was tiny; keep the kids untouched rather
                # than swallowing the whole group.
                return kids

        return folded

    @staticmethod
    def _reading_position(
        element: ParsedElement,
    ) -> tuple[int, float, float]:

        page = 10**9
        y1 = 10**9
        x1 = 10**9

        for location in element.locations:

            if location.type != "pdf_bbox":
                continue

            if (
                location.page is not None
            ):
                page = location.page

            bbox = (
                location.bbox or {}
            )

            if isinstance(
                bbox.get("y1"),
                (int, float),
            ):
                y1 = float(
                    bbox["y1"]
                )

            if isinstance(
                bbox.get("x1"),
                (int, float),
            ):
                x1 = float(
                    bbox["x1"]
                )

            break

        return (
            page,
            y1,
            x1,
        )

    # ---------------------------------------------------------------
    # Parent text
    # ---------------------------------------------------------------

    def parent_text(
        self,
        document: ParsedDocument,
        part_elements: list[ParsedElement],
        block: Block,
    ) -> str:

        if not self.structured:
            return SectionChunker.parent_text(
                self,
                document,
                part_elements,
                block,
            )

        if (
            block.key
            and block.key[0] == "table"
        ):
            # The table element's own text already renders every row,
            # so render only the part rows (keeps split parts of a
            # very large table from duplicating the whole table).
            return "\n".join(
                element.text
                for element in part_elements
                if element.text
            )

        if (
            block.key
            and block.key[0] == "major"
        ):

            root = block.key[1]

            parts: list[str] = []

            if root.text:
                parts.append(root.text)

            parts.extend(
                element.text
                for element in part_elements
                if (
                    element.element_id
                    != root.element_id
                    and element.text
                )
            )

            if not parts:
                parts.append(
                    root.text or ""
                )

            return "\n\n".join(parts)

        return "\n\n".join(
            element.text
            for element in part_elements
        )

    # ---------------------------------------------------------------
    # Tables: a table is its own parent; its rows are the children
    # (one child chunk per row). Oversized tables band into multiple
    # parents holding at most ``children_max`` rows each.
    # ---------------------------------------------------------------

    def _split_parts(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> list[list[ParsedElement]]:

        if (
            block.key
            and block.key[0] == "table"
        ):
            return self._split_table_rows(
                document,
                block,
            )

        return super()._split_parts(
            document=document,
            block=block,
        )

    def _split_table_rows(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> list[list[ParsedElement]]:

        rows = [
            element
            for element in block.elements
            if element.text.strip()
        ]

        if not rows:
            return [[]]

        if len(rows) <= self.children_max:
            return [rows]

        return [
            rows[offset:offset + self.children_max]
            for offset in range(
                0,
                len(rows),
                self.children_max,
            )
        ]

    def _make_children(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        parent_chunk_id: str,
        section_path: list[str],
        start_index: int,
        block: Block,
    ) -> list[Chunk]:

        if (
            block.key
            and block.key[0] == "table"
        ):
            return self._make_table_row_children(
                document=document,
                elements=elements,
                parent_chunk_id=parent_chunk_id,
                section_path=section_path,
                start_index=start_index,
                block=block,
            )

        return super()._make_children(
            document=document,
            elements=elements,
            parent_chunk_id=parent_chunk_id,
            section_path=section_path,
            start_index=start_index,
            block=block,
        )

    def _make_table_row_children(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        parent_chunk_id: str,
        section_path: list[str],
        start_index: int,
        block: Block,
    ) -> list[Chunk]:

        chunks: list[Chunk] = []
        index = start_index

        for element in elements:

            if not element.text.strip():
                continue

            metadata = self._record_metadata(
                document,
                chunk_type="child",
                chunk_index=index,
                parent_chunk_id=parent_chunk_id,
                section_path=section_path,
                block=block,
                group_elements=[element],
                extra=self.child_metadata(
                    document,
                    [element],
                ),
            )

            chunks.append(
                Chunk(
                    chunk_id=(
                        "child_"
                        + uuid4().hex
                    ),

                    parent_chunk_id=(
                        parent_chunk_id
                    ),

                    chunk_type="child",

                    text=element.text,

                    chunk_index=index,

                    doc_id=document.doc_id,

                    version_id=(
                        document.version_id
                    ),

                    user_id=document.user_id,

                    section_path=list(
                        section_path
                    ),

                    locations=list(
                        element.locations
                    ),

                    metadata=metadata,
                )
            )

            index += 1

        return chunks

    # ---------------------------------------------------------------
    # Children: one child chunk per ODL kid / run member
    # ---------------------------------------------------------------

    def child_flush_indices(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
    ) -> frozenset[int]:

        if self.structured:
            return frozenset(
                range(1, len(elements))
            )

        return SectionChunker.child_flush_indices(
            self,
            document,
            elements,
        )