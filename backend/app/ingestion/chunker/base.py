from dataclasses import dataclass, field
from uuid import uuid4

##  FLOW OF THIS COMMON MACHINERY
# Block
# ↓
#_split_parts()
# ↓
#Parent Chunk
# ↓
#_make_children()
# ↓
#Child Chunks
##

from app.core.config import settings
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)


@dataclass
class Chunk:
    chunk_id: str

    parent_chunk_id: str | None

    chunk_type: str

    text: str

    chunk_index: int

    doc_id: str

    version_id: str

    user_id: str

    section_path: list[str]

    locations: list[SourceLocation] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )


class Block:
    """
    A semantic parent unit (section, slide, paragraph run, sheet,
    top-level JSON object...): a run of consecutive parsed elements.
    """

    __slots__ = ("key", "elements", "meta")

    def __init__(
        self,
        key,
        elements,
        meta: dict | None = None,
    ):
        self.key = key
        self.elements = list(elements)
        self.meta = dict(meta) if meta else {}


class BaseChunker:
    """
    Pure mechanics shared by every per-type chunker:

      * child packing (O(n), ``child_max_chars`` cap, atomic elements)
      * parent text cap + truncation marker (fallback only)
      * oversized-parent splitting into parts at preferred boundaries
      * location aggregation (a part only owns its elements' locations)
      * metadata propagation / parent-child id generation

    Per-type subclasses decide what constitutes a parent via

      * ``block_key(element)``  -- identity of the parent unit
      * ``_blocks(document)``   -- overridable custom grouping
      * ``split_boundaries``    -- preferred part-split points
      * ``parent_text``         -- parent content (may be synthetic
                                   schema/context for tabular formats)
      * ``_section_path``       -- hierarchical path shown on chunks
      * ``block_meta`` / ``child_metadata`` -- per-parent / per-child info
    """

    TRUNCATION_MARKER = "\n\u2026[truncated]"

    identifier = "base"

    atomic_element_types: frozenset[str] = frozenset()

    # Per-type chunkers opt in to splitting parents down to the soft
    # cap (sections, JSON objects) of 7500 characters. Natural single-parent types (table,
    # sheet, slide, paragraph group) leave this False and only rely on
    # the 20k hard cap as a fallback.
    respect_parent_soft_cap: bool = False

    def __init__(
        self,
        child_max_chars: int | None = None,
        parent_max_chars: int | None = None,
        parent_soft_max_chars: int | None = None,
    ):
        self.child_max_chars = (
            child_max_chars
            or settings.chunk_child_max_chars
        )
        self.parent_max_chars = (
            parent_max_chars
            or settings.chunk_parent_max_chars
        )
        self.parent_soft_max_chars = (
            parent_soft_max_chars
            or settings.chunk_parent_soft_max_chars
        )

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------
    ## ACTUAL METHOD THAT CONNECTS EVERYTHING
    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[Chunk]:

        chunks = []

        global_index = 0

        for block in self._blocks(document):
            sections = self._split_parts(
                document=document,
                block=block,
            )

            section_path = self._section_path(
                document,
                block,
            )

            base_id = f"parent_{uuid4().hex}"

            parent_count = len(sections)

            for part_index, part_elements in enumerate(
                sections,
                start=1,
            ):

                parent_id = (
                    base_id
                    if parent_count == 1
                    else f"{base_id}_p{part_index}"
                )

                parent_extras = {}

                if parent_count > 1:
                    parent_extras["part"] = part_index
                    parent_extras["part_count"] = parent_count

                parent_metadata = self._record_metadata(
                    document,
                    chunk_type="parent",
                    chunk_index=global_index,
                    parent_chunk_id=None,
                    section_path=section_path,
                    block=block,
                    group_elements=part_elements,
                    extra=parent_extras,
                )

                parent_text = self._cap_text(
                    self.parent_text(
                        document=document,
                        part_elements=part_elements,
                        block=block,
                    )
                )

                parent_locations = [
                    location
                    for element in part_elements
                    for location in element.locations
                ]

                chunks.append(
                    Chunk(
                        chunk_id=parent_id,
                        parent_chunk_id=None,
                        chunk_type="parent",
                        text=parent_text,
                        chunk_index=global_index,
                        doc_id=document.doc_id,
                        version_id=document.version_id,
                        user_id=document.user_id,
                        section_path=list(section_path),
                        locations=parent_locations,
                        metadata=parent_metadata,
                    )
                )

                global_index += 1

                child_chunks = self._make_children(
                    document=document,
                    elements=part_elements,
                    parent_chunk_id=parent_id,
                    section_path=section_path,
                    start_index=global_index,
                    block=block,
                )

                chunks.extend(child_chunks)

                global_index += len(child_chunks)

        return chunks

    # ---------------------------------------------------------------
    # Metadata record
    # ---------------------------------------------------------------

    def _record_metadata(
        self,
        document: ParsedDocument,
        *,
        chunk_type: str,
        chunk_index: int,
        parent_chunk_id: str | None,
        section_path: list[str],
        block: Block,
        group_elements: list[ParsedElement],
        extra: dict | None = None,
    ) -> dict:
        """Self-contained chunk record: identity, document context, the
        per-type block/child context and a filterable location summary.
        This dict is what persists (metadata JSON column + Qdrant
        payload) alongside the embedding."""
        meta = {
            "doc_id": document.doc_id,
            "version_id": document.version_id,
            "user_id": document.user_id,
            "filename": document.filename,
            "parser": document.parser_name,
            "parser_version": document.parser_version,
            "mime_type": document.mime_type,
            "strategy": self.identifier,
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
            "parent_chunk_id": parent_chunk_id,
            "section_path": list(section_path),
        }

        meta.update(block.meta)

        if extra:
            meta.update(extra)

        meta.update(
            self._location_summary(
                group_elements
            )
        )

        return meta

    def _location_summary(
        self,
        group_elements: list[ParsedElement],
    ) -> dict:
        """Aggregate a group's SourceLocations into compact, filterable
        keys (pages/slides/sheets/cell ranges/json & dom paths/row &
        line spans)."""
        summary = {}

        pages: list[int] = []
        slides: list[int] = []
        sheets: list[str] = []
        cell_ranges: list[str] = []
        json_paths: list[str] = []
        dom_paths: list[str] = []
        row_starts: list[int] = []
        row_ends: list[int] = []
        line_starts: list[int] = []
        line_ends: list[int] = []

        for element in group_elements:

            for location in element.locations:

                if (
                    location.page is not None
                    and location.page not in pages
                ):
                    pages.append(location.page)

                if (
                    location.slide is not None
                    and location.slide not in slides
                ):
                    slides.append(location.slide)

                if (
                    location.sheet
                    and location.sheet not in sheets
                ):
                    sheets.append(location.sheet)

                if (
                    location.cell_range
                    and location.cell_range
                    not in cell_ranges
                ):
                    cell_ranges.append(
                        location.cell_range
                    )

                if (
                    location.json_path
                    and location.json_path
                    not in json_paths
                ):
                    json_paths.append(
                        location.json_path
                    )

                if (
                    location.dom_path
                    and location.dom_path
                    not in dom_paths
                ):
                    dom_paths.append(
                        location.dom_path
                    )

                if location.row_start is not None:
                    row_starts.append(
                        location.row_start
                    )

                if location.row_end is not None:
                    row_ends.append(
                        location.row_end
                    )

                if location.line_start is not None:
                    line_starts.append(
                        location.line_start
                    )

                if location.line_end is not None:
                    line_ends.append(
                        location.line_end
                    )

        if pages:
            pages.sort()
            summary["pages"] = pages

        if slides:
            slides.sort()
            summary["slides"] = slides

        if sheets:
            summary["sheets"] = sheets

        if cell_ranges:
            summary["cell_ranges"] = cell_ranges

        if json_paths:
            summary["json_paths"] = json_paths[:50]

        if dom_paths:
            summary["dom_paths"] = dom_paths

        if row_starts:
            summary["row_start"] = min(row_starts)
            summary["row_end"] = max(row_ends)

        if line_starts:
            summary["line_start"] = min(line_starts)
            summary["line_end"] = max(line_ends)

        return summary

    # ---------------------------------------------------------------
    # Per-type hooks
    # ---------------------------------------------------------------

    def block_key(self, element: ParsedElement):
        raise NotImplementedError(
            f"{type(self).__name__} must define block_key()"
        )

    def block_meta(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        key,
    ) -> dict:
        return {}

    def split_boundaries(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
    ) -> list[int]:
        """Indices at which an oversized parent prefers to split
        (object/array-item boundaries, etc.)."""
        return []

    def parent_text(
        self,
        document: ParsedDocument,
        part_elements: list[ParsedElement],
        block: Block,
    ) -> str:
        return "\n\n".join(
            element.text
            for element in part_elements
        )

    def _section_path(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> list[str]:
        if block.elements:
            return list(
                block.elements[0].section_path
            )
        return []

    def child_metadata(
        self,
        document: ParsedDocument,
        group_elements: list[ParsedElement],
    ) -> dict:
        return {}

    def child_flush_indices(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
    ) -> frozenset[int]:
        """Indices (1..n-1) before which child packing must flush, so a
        child never spans a structural boundary (e.g. a TXT paragraph
        break)."""
        return frozenset()

    # ---------------------------------------------------------------
    # Block grouping
    # ---------------------------------------------------------------

    def _blocks(
        self,
        document: ParsedDocument,
    ) -> list[Block]:

        blocks = []

        current_key = None
        current = []

        for element in document.elements:

            key = self.block_key(element)

            if current and key != current_key:

                blocks.append(
                    Block(
                        current_key,
                        current,
                        self.block_meta(
                            document,
                            current,
                            current_key,
                        ),
                    )
                )

                current = []

            current_key = key
            current.append(element)

        if current:

            blocks.append(
                Block(
                    current_key,
                    current,
                    self.block_meta(
                        document,
                        current,
                        current_key,
                    ),
                )
            )

        return blocks

    # ---------------------------------------------------------------
    # Oversized-parent splitting (20k is a fallback, not the rule)
    # ---------------------------------------------------------------

    def _split_parts(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> list[list[ParsedElement]]:

        elements = block.elements

        lengths = [
            len(element.text)
            for element in elements
        ]

        total = (
            sum(lengths)
            + 2 * max(0, len(elements) - 1)
        )

        if self.respect_parent_soft_cap:
            threshold = min(
                self.parent_soft_max_chars,
                self.parent_max_chars,
            )
        else:
            threshold = self.parent_max_chars

        if total <= threshold:
            return [elements]

        boundaries = sorted(
            {
                index
                for index in self.split_boundaries(
                    document,
                    elements,
                )
                if 0 < index < len(elements)
            }
        )

        parts = []
        start = 0
        current_len = 0

        for index in range(len(elements)):

            if current_len:

                added = lengths[index] + 2

                if current_len + added > threshold:

                    cut = self._preferred_cut(
                        boundaries,
                        start,
                        index,
                    )

                    if cut is None:
                        cut = index

                    parts.append(
                        elements[start:cut]
                    )

                    start = cut
                    current_len = 0

            current_len += lengths[index] + (
                2
                if index > start
                else 0
            )

        if start < len(elements):
            parts.append(elements[start:])

        if self.respect_parent_soft_cap:
            self._merge_tiny_tail(
                parts,
                threshold,
            )

        return parts

    def _merge_tiny_tail(
        self,
        parts: list[list[ParsedElement]],
        threshold: int,
    ) -> None:
        """Fold a small trailing part into its predecessor so a soft-cap
        split never leaves near-empty wafer parents behind.

        A merged part may exceed the soft cap but never the hard cap
        (that stays the absolute ceiling)."""
        if len(parts) < 2:
            return

        min_tail = threshold // 2

        while len(parts) > 1:

            tail = parts[-1]

            tail_chars = (
                sum(
                    len(element.text)
                    for element in tail
                )
                + 2 * max(0, len(tail) - 1)
            )

            if tail_chars >= min_tail:
                break

            previous = parts[-2]

            previous_chars = (
                sum(
                    len(element.text)
                    for element in previous
                )
                + 2 * max(0, len(previous) - 1)
            )

            if (
                previous_chars
                + 2
                + tail_chars
                > self.parent_max_chars
            ):
                break

            previous.extend(tail)
            parts.pop()

    @staticmethod
    def _preferred_cut(
        boundaries: list[int],
        start: int,
        index: int,
    ) -> int | None:
        chosen = None
        for boundary in boundaries:
            if (
                boundary > start
                and boundary <= index
            ):
                chosen = boundary
        return chosen

    def _cap_text(self, text: str) -> str:
        if len(text) <= self.parent_max_chars:
            return text
        marker = self.TRUNCATION_MARKER
        return (
            text[: self.parent_max_chars - len(marker)]
            + marker
        )

    # ---------------------------------------------------------------
    # Child packing
    # ---------------------------------------------------------------

    def _make_children(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        parent_chunk_id: str,
        section_path: list[str],
        start_index: int,
        block: Block,
    ) -> list[Chunk]:

        chunks = []
        index = start_index

        current = []
        current_locations = []
        current_len = 0

        flush_indices = (
            self.child_flush_indices(
                document,
                elements,
            )
        )

        def flush() -> None:
            nonlocal current, current_locations, current_len, index

            if not current:
                return

            metadata = self._record_metadata(
                document,
                chunk_type="child",
                chunk_index=index,
                parent_chunk_id=parent_chunk_id,
                section_path=section_path,
                block=block,
                group_elements=current,
                extra=self.child_metadata(
                    document,
                    current,
                ),
            )

            chunks.append(
                Chunk(
                    chunk_id=f"child_{uuid4().hex}",
                    parent_chunk_id=parent_chunk_id,
                    chunk_type="child",
                    text="\n\n".join(
                        element.text
                        for element in current
                    ),
                    chunk_index=index,
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    user_id=document.user_id,
                    section_path=list(section_path),
                    locations=current_locations,
                    metadata=metadata,
                )
            )

            index += 1

            current = []
            current_locations = []
            current_len = 0

        for position, element in enumerate(
            elements
        ):

            # Empty-text blocks (an uncaptioned image, a stray blank
            # slides content box...) never become their own child
            # chunks; if it sits on a structural flush boundary the
            # boundary is still honoured so neighbours don't merge
            # across it.
            if not element.text.strip():
                if position in flush_indices:
                    flush()
                continue

            # Atomic elements keep their own child chunk (e.g. a
            # markdown code block) and are never merged with neighbors,
            # even when they exceed the child size cap.
            if (
                element.element_type
                in self.atomic_element_types
            ):
                flush()
                current = [element]
                current_locations = list(
                    element.locations
                )
                current_len = len(element.text)
                flush()
                continue

            if position in flush_indices:
                flush()

            if (
                current
                and current_len
                + len(element.text)
                + 2
                > self.child_max_chars
            ):
                flush()

            current.append(element)
            current_locations.extend(
                element.locations
            )
            current_len += len(element.text) + (
                2
                if len(current) > 1
                else 0
            )

        flush()

        return chunks

    # ---------------------------------------------------------------
    # Shared tabular metadata extraction
    # ---------------------------------------------------------------

    @staticmethod
    def _group_tabular_meta(
        group_elements: list[ParsedElement],
    ) -> dict:

        meta = {}

        row_starts = []
        row_ends = []

        sheets = []
        range_starts = []
        range_ends = []

        columns = None

        for element in group_elements:

            if columns is None and element.metadata.get(
                "columns"
            ):
                columns = element.metadata["columns"]

            for location in element.locations:

                if location.type == "csv_row":

                    if location.row_start is not None:
                        row_starts.append(
                            location.row_start
                        )

                    if location.row_end is not None:
                        row_ends.append(
                            location.row_end
                        )

                elif location.type == "xlsx_range":

                    sheet = location.sheet
                    if sheet and sheet not in sheets:
                        sheets.append(sheet)

                    if location.cell_range:
                        parts = (
                            location.cell_range.split(":")
                        )
                        range_starts.append(parts[0])
                        range_ends.append(parts[-1])

        if row_starts:
            meta["row_start"] = min(row_starts)
            meta["row_end"] = max(row_ends)

        if columns is not None:
            meta["columns"] = columns

        if sheets and range_starts:

            first = range_starts[0]
            last = range_ends[-1]

            meta["sheet"] = sheets[0]
            meta["cell_range"] = (
                first
                if first == last
                else f"{first}:{last}"
            )

        elif sheets:
            meta["sheet"] = sheets[0]

        return meta