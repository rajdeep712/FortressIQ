from dataclasses import dataclass, field, replace
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
from app.ingestion.chunker.tokenizer import (
    estimate_tokens,
)
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
    #
    # The standardized (non-tabular) token-based windowing ignores this
    # flag entirely: every parent is budgeted on tokens.
    respect_parent_soft_cap: bool = False

    def __init__(
        self,
        *,
        standard: bool | None = None,
        child_max_chars: int | None = None,
        parent_max_chars: int | None = None,
        parent_soft_max_chars: int | None = None,
        child_max_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        overlap_min_chars: int | None = None,
        overlap_max_chars: int | None = None,
    ):
        """Standard vs legacy sizing.

        ``standard`` windowing (token-based children, 200-300 char
        overlap, token-budgeted parents) is the default for every
        chunker; tabular formats (CSV/XLSX) pass ``standard=False`` to
        keep the legacy character caps. Passing any explicit character
        cap (or a sub-class specific char setting, e.g. TXT's
        ``group_target_chars`` -- see there) switches that instance to
        the legacy character behavior so tests/callers stay
        deterministic.
        """
        explicit_chars = (
            child_max_chars is not None
            or parent_max_chars is not None
            or parent_soft_max_chars is not None
        )

        self.standard = (
            settings.chunk_standard_windowing
            if standard is None
            else standard
        )

        if self.standard and not explicit_chars:
            self.chunk_standard = (
                child_max_tokens
                or settings.chunk_child_max_tokens
            )
            self.parent_token_max = (
                parent_max_tokens
                or settings.chunk_parent_max_tokens
            )
            self.overlap_min_chars = (
                overlap_min_chars
                or settings.chunk_overlap_min_chars
            )
            self.overlap_max_chars = (
                overlap_max_chars
                or settings.chunk_overlap_max_chars
            )
            self.overlap_target_chars = (
                self.overlap_min_chars
                + self.overlap_max_chars
            ) // 2
            self.children_min = (
                settings.chunk_children_min
            )
            self.children_max = (
                settings.chunk_children_max
            )

            # Character proxies only used as first-cut window guesses:
            # the real ceiling is always enforced with estimate_tokens.
            self.child_max_chars = (
                self.chunk_standard * 4
            )
            self.parent_max_chars = (
                self.parent_token_max * 4
            )

            # A parent at soft size yields ~children_max windows.
            overlap_tokens = max(
                1,
                self.overlap_target_chars // 4,
            )
            self.parent_soft_tokens = max(
                1,
                (
                    self.chunk_standard
                    * self.children_max
                    - overlap_tokens
                    * (self.children_max - 1)
                ),
            )
            # The soft end of a parent yields ~``children_max`` windows;
            # its mirror yields ~``children_min`` windows. Coalescing
            # (see ``_parent_groups``) bands every parent into this range
            # so each parent holds ``children_min``..``children_max``
            # children. Oversized lone elements are the only exceptions.
            self.parent_min_tokens = max(
                1,
                (
                    self.chunk_standard
                    * self.children_min
                    - overlap_tokens
                    * (self.children_min - 1)
                ),
            )
            # ``parent_soft_tokens`` is the theoretical cap: it assumes
            # every child fills right up to ``chunk_child_max_tokens``.
            # Windows do not; children land short of the cap at seams and
            # folded tails, so the realized new-content step is closer to
            # ``child_max - overlap - underfill``. Closing coalescing
            # and part packing at ``parent_soft_tokens`` therefore left
            # parents holding ``children_max + 1..3`` windows. Scale the
            # close cap by the realized step so a sealed parent holds at
            # most ``children_max`` children.
            underfill_margin = 20
            realized_step = max(
                1,
                (
                    self.chunk_standard
                    - overlap_tokens
                    - underfill_margin
                ),
            )
            self.parent_close_tokens = max(
                self.parent_min_tokens + 1,
                self.children_max * realized_step,
            )
            self.parent_soft_max_chars = (
                self.parent_soft_tokens * 4
            )
        else:
            self.standard = False
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
            self.chunk_standard = None
            self.parent_token_max = (
                parent_max_tokens
                or settings.chunk_parent_max_tokens
            )
            self.overlap_min_chars = (
                overlap_min_chars
                or settings.chunk_overlap_min_chars
            )
            self.overlap_max_chars = (
                overlap_max_chars
                or settings.chunk_overlap_max_chars
            )
            self.overlap_target_chars = (
                self.overlap_min_chars
                + self.overlap_max_chars
            ) // 2

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

        for block in self._parent_groups(document):
            sections = self._split_parts(
                document=document,
                block=block,
            )

            sections = [
                sub_section
                for section in sections
                for sub_section in self._cap_child_windows(section)
            ]

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

    def block_is_seam(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> bool:
        """Whether a block is a hard structural boundary.

        Seam blocks never coalesce with neighbors: each keeps its own
        parent (json top-level keys, pptx slides, pdf numbered items,
        heading sections). Content runs (txt paragraph groups, unsectioned
        body text, pdf runs) are not seams and get DP-optimized into
        banded parents by ``_parent_groups``."""
        return False

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
    # Parent grouping (coalescing)
    # ---------------------------------------------------------------

    def _combine_blocks(
        self,
        document: ParsedDocument,
        blocks: list[Block],
    ) -> Block:
        """Merge one parent group's blocks into a single parent Block.

        The base concatenates the element streams (so the default
        ``parent_text``/child windowing spans them). Per-type subclasses
        override this to aggregate labels/metadata for coalesced parents
        (e.g. a slide/item range covered by the parent)."""
        if len(blocks) == 1:
            return blocks[0]

        elements = []
        for block in blocks:
            elements.extend(block.elements)

        return Block(
            key=("coalesced",),
            elements=elements,
            meta={"spans": len(blocks)},
        )

    def _block_tokens(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> int:
        """Token size of a block's parent text (the unit that %s the
        per-parent target band)."""
        return estimate_tokens(
            self.parent_text(
                document=document,
                part_elements=block.elements,
                block=block,
            )
        )

    def _parent_groups(
        self,
        document: ParsedDocument,
    ) -> list[Block]:
        """Coalesce consecutive blocks into parent groups.

        Structural blocks (json top-level keys, pptx slides, pdf numbered
        items, heading sections -- see ``block_is_seam``) are hard
        boundaries: each keeps its own parent, never coalescing with a
        neighbor. Maximal runs of coalescable content blocks (txt
        paragraph groups, unsectioned body text, pdf unstructured runs)
        are split into parents by dynamic programming (``_dp_coalesce_run``):
        the minimum number of parents such that every non-final parent
        holds ``children_min``..``children_max`` children and stays under
        the hard parent token cap. In legacy (tabulated) mode every block
        stays its own parent.

        This is what turns small sibling content blocks into parents that
        actually hold ``children_min``..``children_max`` children instead of
        each becoming a 1-3 child wafer, while seam blocks keep the
        structure their type guarantees."""
        blocks = self._blocks(document)

        if not self.standard or len(blocks) < 2:
            return [self._combine_blocks(document, [b]) for b in blocks]

        groups: list[list[Block]] = []
        run: list[Block] = []

        for block in blocks:
            if self.block_is_seam(document, block):
                if run:
                    groups.append(run)
                    run = []
                groups.append([block])
                continue
            run.append(block)

        if run:
            groups.append(run)

        combined: list[Block] = []

        for group in groups:
            if (
                len(group) == 1
                or self.block_is_seam(document, group[0])
            ):
                combined.append(
                    self._combine_blocks(document, group)
                )
                continue

            combined.extend(
                self._dp_coalesce_run(
                    document,
                    group,
                )
            )

        return combined

    def _coalesced_block(
        self,
        document: ParsedDocument,
        blocks: list[Block],
    ) -> Block:
        return self._combine_blocks(document, blocks)

    def _group_children(
        self,
        document: ParsedDocument,
        blocks: list[Block],
    ) -> int:
        """Exact number of child windows a parent over ``blocks`` would
        yield. This is the same ``_sliding_windows`` pass
        ``_make_children_standard`` runs, so the DP's child oracle is
        ground truth rather than a token estimate."""
        combined = self._combine_blocks(document, blocks)
        elements = [
            element
            for element in combined.elements
            if element.text.strip()
        ]
        if not elements:
            return 0
        (
            stream,
            _starts,
            structural_seams,
            sentence_seams,
            atomic_spans,
        ) = self._element_stream_props(elements)
        return len(
            self._sliding_windows(
                stream,
                atomic_spans,
                structural_seams,
                sentence_seams,
            )
        )

    def _dp_coalesce_run(
        self,
        document: ParsedDocument,
        blocks: list[Block],
    ) -> list[Block]:
        """Split a coalescable run into the minimum number of parents
        such that each non-final parent holds ``children_min``..``children_max``
        children and never exceeds the hard parent token cap.

        Each candidate parent covers a contiguous slice of the run; its
        child count is measured exactly (``_group_children``) so the DP
        is driven by the real window algorithm. A slice that would hold
        more than ``children_max`` windows, or fewer than ``children_min``
        in the middle of the run, is infeasible -- a lone oversized block
        (past the token cap on its own) is always acceptable on its own,
        since ``_split_parts`` re-balances it later. The final parent may
        fall short of ``children_min`` when the whole run is too short."""
        n = len(blocks)
        if n <= 1:
            return [self._combine_blocks(document, blocks)]

        block_tokens = [
            self._block_tokens(document, block)
            for block in blocks
        ]

        prefix = [0]
        for value in block_tokens:
            prefix.append(prefix[-1] + value)

        cache: dict[tuple[str, int, int], int] = {}

        def children(start: int, end: int) -> int:
            key = ("c", start, end)
            if key not in cache:
                cache[key] = self._group_children(
                    document,
                    blocks[start:end],
                )
            return cache[key]

        INF = float("inf")
        best_groups = [INF] * (n + 1)
        best_fill = [INF] * (n + 1)
        best_split = [-1] * (n + 1)
        best_groups[0] = 0
        best_fill[0] = 0

        for start in range(n):
            if best_groups[start] == INF:
                continue

            # A slice's token sum is monotone in ``end``: stop scanning
            # once it can never fit a parent (unless a lone oversized
            # block, which keeps its own parent).
            for end in range(start + 1, n + 1):
                token_estimate = prefix[end] - prefix[start]
                if (
                    token_estimate > self.parent_token_max
                    and end != start + 1
                ):
                    break

                child_count = children(start, end)

                forced_lone = (
                    end - start == 1
                    and (
                        child_count > self.children_max
                        or token_estimate > self.parent_token_max
                    )
                )

                if (
                    child_count > self.children_max
                    and not forced_lone
                ):
                    if (
                        end > start + 1
                        and child_count > self.children_max
                    ):
                        break
                    continue

                if (
                    child_count < self.children_min
                    and end != n
                    and not forced_lone
                ):
                    continue

                deviation = (
                    abs(child_count - self.children_max)
                    if child_count <= self.children_max
                    else self.children_max
                )
                new_groups = best_groups[start] + 1
                new_fill = best_fill[start] + deviation

                if (
                    new_groups < best_groups[end]
                    or (
                        new_groups == best_groups[end]
                        and new_fill < best_fill[end]
                    )
                ):
                    best_groups[end] = new_groups
                    best_fill[end] = new_fill
                    best_split[end] = start

        result: list[Block] = []
        end = n
        while end > 0:
            start = best_split[end]
            if start < 0:
                start = end - 1
            result.append(
                self._combine_blocks(
                    document,
                    blocks[start:end],
                )
            )
            end = start

        return list(reversed(result))

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

        if self.standard:
            return self._split_parts_standard(
                document,
                block,
            )

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

    def _split_parts_standard(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> list[list[ParsedElement]]:
        """Token-budgeted parent parts (standard mode).

        Elements pack into a part until its joined text would pass the
        soft token budget (≈ ``children_max`` child windows per parent).
        A lone element at or under the *hard* parent ceiling still owns
        its own part ("structure dictates fewer children"). Oversized
        single elements are pre-split at sentence/whitespace seams into
        clones so no whole parent ever exceeds the hard token ceiling
        (atomic elements -- e.g. markdown code blocks -- are exempt and
        stay whole). Empty-text elements never become parts.
        """
        units: list[ParsedElement] = []

        for element in block.elements:
            if not element.text.strip():
                continue

            if (
                element.element_type
                in self.atomic_element_types
                or estimate_tokens(element.text)
                <= self.parent_close_tokens
            ):
                units.append(element)
                continue

            slices = self._split_oversized_text(
                element.text,
                self.parent_close_tokens,
            )
            for text_slice in slices:
                units.append(
                    replace(
                        element,
                        text=text_slice,
                    )
                )

        if not units:
            return [[]]

        def join_text(elements):
            return "\n\n".join(
                element.text
                for element in elements
            )

        joined_tokens = estimate_tokens(
            join_text(units),
        )

        if joined_tokens <= self.parent_close_tokens:
            return [units]

        # Balance oversize blocks by *projected window count*, not
        # tokens: token-based balance can't anticipate how many windows
        # a dense stream yields, leaving wafer parts of 1-3 children.
        # Windows over the whole block are computed once; the block is
        # then bisected on element boundaries so each part holds roughly
        # ``windows / n_parts`` windows (each part still at most
        # ``children_max`` -- ``_cap_child_windows`` hard-guarantees it).
        (
            stream,
            starts,
            structural_seams,
            sentence_seams,
            atomic_spans,
        ) = self._element_stream_props(units)

        windows = self._sliding_windows(
            stream,
            atomic_spans,
            structural_seams,
            sentence_seams,
        )

        n_parts = max(
            2,
            -(-len(windows) // self.children_max),
        )

        parts: list[list[ParsedElement]] = []
        remaining = units[:]
        remaining_starts = starts[:]
        cut_targets = [
            (len(windows) * (part + 1)) // n_parts
            for part in range(n_parts - 1)
        ]

        for target_window_index in cut_targets:
            cut_offset = windows[target_window_index][0]

            element_start = 0
            while (
                element_start < len(remaining)
                and remaining_starts[element_start] < cut_offset
            ):
                element_start += 1

            parts.append(remaining[:element_start])
            remaining = remaining[element_start:]
            remaining_starts = [
                s - (remaining_starts[element_start - 1] if element_start else 0)
                for s in remaining_starts[element_start:]
            ]

        if remaining:
            parts.append(remaining)

        self._merge_tiny_tail_tokens(parts)

        return parts

    def _merge_tiny_tail_tokens(
        self,
        parts: list[list[ParsedElement]],
    ) -> None:
        """Fold a small trailing part into its predecessor so soft
        token-budget splits never leave near-empty wafer parents.
        A merged part never exceeds the hard token ceiling."""
        if len(parts) < 2:
            return

        def join_text(elements):
            return "\n\n".join(
                element.text
                for element in elements
            )

        while len(parts) > 1:
            tail = parts[-1]
            tail_tokens = estimate_tokens(
                join_text(tail)
            )

            if (
                tail_tokens
                >= self.parent_close_tokens // 2
            ):
                break

            previous = parts[-2]
            merged_tokens = estimate_tokens(
                join_text(previous)
                + "\n\n"
                + join_text(tail)
            )

            if (
                merged_tokens
                > self.parent_close_tokens
            ):
                break

            previous.extend(tail)
            parts.pop()

    def _split_oversized_text(
        self,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        """Split one element's text into consecutive slices, each at
        most ``max_tokens`` on sentence/whitespace seams (a seam-less
        run falls back to a hard character boundary)."""
        if estimate_tokens(text) <= max_tokens:
            return [text]

        slices: list[str] = []
        start = 0
        length = len(text)
        # First-cut char budget: the token ceiling converts at ~4 c/t.
        char_budget = max(1, max_tokens * 4)

        while start < length:
            high = min(length, start + char_budget)
            end = self._token_slot_in_text(
                text,
                start,
                high,
                max_tokens,
            )
            slices.append(text[start:end])
            start = end

        return slices

    def _token_slot_in_text(
        self,
        text: str,
        start: int,
        high: int,
        max_tokens: int,
    ) -> int:
        """Largest position <= ``high`` (and >= ``start``) whose slice
        fits ``max_tokens``, drifted down to the nearest seam."""
        if (
            high == start
            or estimate_tokens(text[start:high])
            <= max_tokens
        ):
            return high

        lo, hi = start + 1, high
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (
                estimate_tokens(text[start:mid])
                <= max_tokens
            ):
                lo = mid
            else:
                hi = mid - 1

        if lo <= start:
            return min(len(text), start + 1)

        # Drift up to one overlap window down from the token boundary
        # to the nearest sentence/word seam (seam-less runs -- e.g.
        # repeated filler -- keep the hard token boundary).
        position = lo
        floor = max(start, lo - self.overlap_max_chars)
        while (
            position > floor
            and not self._is_seam_at(
                text,
                position,
            )
        ):
            position -= 1
        if self._is_seam_at(text, position):
            return position
        return lo

    @staticmethod
    def _is_seam_at(text: str, position: int) -> bool:
        """True when ``text[:position]`` ends on a sentence or word
        boundary (position is the first char of the next unit)."""
        if position <= 0 or position >= len(text):
            return True
        previous = text[position - 1]
        following = text[position]

        if previous in " \t":
            return True

        if (
            previous in ".!?;"
            and following in " \n"
        ):
            return True

        if previous == "\n":
            return True

        return False

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
        if not self.standard:
            if len(text) <= self.parent_max_chars:
                return text
            marker = self.TRUNCATION_MARKER
            return (
                text[: self.parent_max_chars - len(marker)]
                + marker
            )

        marker = self.TRUNCATION_MARKER
        full_tokens = estimate_tokens(text)

        if full_tokens <= self.parent_token_max:
            return text

        budget = self.parent_token_max - (
            estimate_tokens(marker)
        )

        cut = self._token_slot_in_text(
            text,
            0,
            len(text),
            budget,
        )
        if cut >= len(text):
            return text
        return text[:cut] + marker

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

        if self.standard:
            return self._make_children_standard(
                document=document,
                elements=elements,
                parent_chunk_id=parent_chunk_id,
                section_path=section_path,
                start_index=start_index,
                block=block,
            )

        return self._make_children_legacy(
            document=document,
            elements=elements,
            parent_chunk_id=parent_chunk_id,
            section_path=section_path,
            start_index=start_index,
            block=block,
        )

    def _element_stream_props(
        self,
        elements: list[ParsedElement],
    ) -> tuple[
        str,
        list[int],
        set[int],
        set[int],
        list[tuple[int, int]],
    ]:
        """Join default window's element texts into a stream, mapping
        each element back to its char offset (so windows can be routed
        to owning elements) and pre-computing structural/sentence seams
        and atomic spans."""
        stream = "\n\n".join(
            element.text
            for element in elements
        )

        starts: list[int] = []
        cursor = 0
        for index, element in enumerate(elements):
            if index > 0:
                cursor += 2  # "\n\n" separator
            starts.append(cursor)
            cursor += len(element.text)

        structural_seams = set(starts[1:])
        sentence_seams = self._sentence_boundaries(stream)

        atomic_spans: list[tuple[int, int]] = []
        for element, start in zip(elements, starts):
            if (
                element.element_type
                in self.atomic_element_types
            ):
                atomic_spans.append(
                    (
                        start,
                        start + len(element.text),
                    )
                )

        return (
            stream,
            starts,
            structural_seams,
            sentence_seams,
            atomic_spans,
        )

    def _cap_child_windows(
        self,
        elements: list[ParsedElement],
    ) -> list[list[ParsedElement]]:
        """Split a parent part when it would hold more than
        ``children_max`` windows, so every parent in standard mode ends
        with ``children_min``..``children_max`` children.

        ``_sliding_windows`` is the ground truth here: token-only
        estimates cannot anticipate how many windows a dense stream
        actually yields, so the projected window count is measured. A
        part breaching the cap is bisected on an element boundary near
        the middle window, then each half is re-checked recursively."""
        if not self.standard or len(elements) < 2:
            return [elements]

        (
            stream,
            starts,
            structural_seams,
            sentence_seams,
            atomic_spans,
        ) = self._element_stream_props(elements)

        windows = self._sliding_windows(
            stream,
            atomic_spans,
            structural_seams,
            sentence_seams,
        )

        if len(windows) <= self.children_max:
            return [elements]

        split_offset = windows[len(windows) // 2][0]
        split_index = 0
        for element_index, element_start in enumerate(starts):
            if element_start >= split_offset:
                split_index = element_index
                break
            split_index = element_index + 1

        split_index = max(
            1,
            min(len(elements), split_index),
        )

        if (
            split_index >= len(elements)
            or split_index == 0
        ):
            return [elements]

        left = self._cap_child_windows(elements[:split_index])
        right = self._cap_child_windows(elements[split_index:])
        return left + right

    def _make_children_standard(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        parent_chunk_id: str,
        section_path: list[str],
        start_index: int,
        block: Block,
    ) -> list[Chunk]:
        """Token-aware sliding-window children (standard mode).

        The parent's element body is joined into one stream. Windows
        slide over it: every child is at most ``chunk_standard``
        tokens; consecutive children overlap by ``overlap_min_chars``
        .. ``overlap_max_chars``, aligned to element/paragraph seams
        and sentence boundaries. Atomic elements (e.g. markdown code
        blocks) always keep their own window. A window owns the
        locations (and ``child_metadata``) of every element it
        overlaps, so a spanning window still points at all its source
        regions.
        """
        elements = [
            element
            for element in elements
            if element.text.strip()
        ]

        if not elements:
            return []

        (
            stream,
            starts,
            structural_seams,
            sentence_seams,
            atomic_spans,
        ) = self._element_stream_props(elements)

        windows = self._sliding_windows(
            stream,
            atomic_spans,
            structural_seams,
            sentence_seams,
        )

        chunks: list[Chunk] = []
        index = start_index

        for start, end in windows:
            owning = self._owners_for_window(
                elements,
                starts,
                start,
                end,
            )

            metadata = self._record_metadata(
                document,
                chunk_type="child",
                chunk_index=index,
                parent_chunk_id=parent_chunk_id,
                section_path=section_path,
                block=block,
                group_elements=owning,
                extra=self.child_metadata(
                    document,
                    owning,
                ),
            )

            chunks.append(
                Chunk(
                    chunk_id=f"child_{uuid4().hex}",
                    parent_chunk_id=parent_chunk_id,
                    chunk_type="child",
                    text=stream[start:end],
                    chunk_index=index,
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    user_id=document.user_id,
                    section_path=list(section_path),
                    locations=[loc for element in owning for loc in element.locations],
                    metadata=metadata,
                )
            )

            index += 1

        return chunks

    def _sliding_windows(
        self,
        stream: str,
        atomic_spans: list[tuple[int, int]],
        structural_seams: set[int],
        sentence_seams: set[int],
    ) -> list[tuple[int, int]]:
        """Windows over ``stream``: token-capped, seam-aligned, with a
        200-300 char overlap between consecutive windows. Atomic spans
        always occupy their own window and are never straddled."""
        windows: list[tuple[int, int]] = []
        start = 0
        length = len(stream)

        while start < length:
            atom = self._atomic_at(atomic_spans, start)
            if atom is not None:
                if (
                    estimate_tokens(
                        stream[atom[0]:atom[1]]
                    )
                    <= self.chunk_standard
                ):
                    windows.append(atom)
                else:
                    windows.extend(
                        self._split_atomic_windows(
                            stream,
                            atom,
                        )
                    )
                start = atom[1]
                continue

            next_atom_start = (
                self._next_atom_start(
                    atomic_spans,
                    start,
                )
            )

            end = self._token_slot_char(
                stream,
                start,
                self.child_max_chars,
                self.chunk_standard,
            )

            if (
                next_atom_start is not None
                and start < next_atom_start < end
            ):
                end = next_atom_start

            if end <= start:
                end = min(length, start + 1)

            refined = self._refine_end(
                stream,
                start,
                end,
                structural_seams,
                sentence_seams,
            )
            if refined > start:
                end = refined

            # A tail smaller than the maximum overlap can't seed another
            # window with a full 200-300 char overlap: fold it into this
            # window and stop -- but only while it keeps the window
            # within the child token cap. The same applies when this
            # window itself is at most ``overlap_max`` chars long: any
            # next window would start inside it (overlap swallows all
            # the new content), so the remaining tail is folded in.
            if (
                (
                    length - end < self.overlap_max_chars
                    or end - start <= self.overlap_max_chars
                )
                and estimate_tokens(stream[start:length])
                <= self.chunk_standard
            ):
                end = length

            windows.append((start, end))

            if end >= length:
                break

            next_start = self._overlap_start(
                end,
                structural_seams,
                sentence_seams,
            )
            if next_start <= start:
                # No usable seam ahead: overlap would swallow the whole
                # next window or rewind into the current one. Jump
                # straight to ``end`` (zero overlap) instead of creeping
                # one char at a time -- creeping turns dense, seam-poor
                # text into hundreds of redundant windows.
                next_start = end
            start = next_start

        return windows

    @staticmethod
    def _atomic_at(
        spans: list[tuple[int, int]],
        position: int,
    ) -> tuple[int, int] | None:
        for start, end in spans:
            if start <= position < end:
                return start, end
            if position < start:
                break
        return None

    def _split_atomic_windows(
        self,
        stream: str,
        atom: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Split an atomic span that alone exceeds the child token cap
        into several token-capped windows inside its own span. Atomic
        blocks keep their own children (never merged with neighbours or
        straddled), but a single oversized block is split so no child
        exceeds the token cap."""
        atom_start, atom_end = atom
        windows: list[tuple[int, int]] = []
        start = atom_start
        while start < atom_end:
            end = self._token_slot_char(
                stream,
                start,
                self.child_max_chars,
                self.chunk_standard,
            )
            if end <= start:
                end = min(atom_end, start + 1)
            windows.append((start, end))
            start = end
        return windows

    @staticmethod
    def _next_atom_start(
        spans: list[tuple[int, int]],
        position: int,
    ) -> int | None:
        for start, _end in spans:
            if start > position:
                return start
        return None

    def _token_slot_char(
        self,
        stream: str,
        start: int,
        max_chars: int,
        max_tokens: int,
    ) -> int:
        """Largest end (char index) for a window starting at ``start``
        whose text fits ``max_tokens``, capped at ``start +
        max_chars`` (and the stream length)."""
        length = len(stream)
        high = min(
            length,
            start + max(1, max_chars),
        )

        if (
            high == start
            or estimate_tokens(stream[start:high])
            <= max_tokens
        ):
            return high

        lo, hi = start + 1, high
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (
                estimate_tokens(stream[start:mid])
                <= max_tokens
            ):
                lo = mid
            else:
                hi = mid - 1

        return max(start + 1, lo)

    def _refine_end(
        self,
        stream: str,
        start: int,
        end: int,
        structural_seams: set[int],
        sentence_seams: set[int],
    ) -> int:
        """Settle a window end on the seam closest to ``end`` flattened
        toward the token cap so children fill densely (toward 250
        tokens), only stepping back to a seam within ``overlap_min``
        chars of the cap. Element/paragraph seams win over bare sentence
        ends; if none is found the token cap end is kept."""
        low = max(start, end - self.overlap_min_chars)
        high = end
        high = min(high, len(stream))

        if high > low:
            for position in range(high, low - 1, -1):
                if (
                    position in structural_seams
                    and position > start
                ):
                    return position
            for position in range(high, low - 1, -1):
                if (
                    position in sentence_seams
                    and position > start
                ):
                    return position

        return end

    def _overlap_start(
        self,
        end: int,
        structural_seams: set[int],
        sentence_seams: set[int],
    ) -> int:
        """Start of the next window: ``end`` minus 200-300 chars,
        settled on a sentence/element seam so the overlap region is a
        coherent stretch of text."""
        low = max(0, end - self.overlap_max_chars)
        high = end - self.overlap_min_chars
        if high < low:
            high = low

        if high >= low:
            for position in range(high, low - 1, -1):
                if position in sentence_seams:
                    return position
            for position in range(high, low - 1, -1):
                if position in structural_seams:
                    return position

        return max(0, end - self.overlap_target_chars)

    @staticmethod
    def _sentence_boundaries(stream: str) -> set[int]:
        """Char positions that end a sentence/paragraph: right after a
        newline, or after a sentence terminator followed by space or a
        newline."""
        boundaries: set[int] = set()
        length = len(stream)
        for index in range(1, length):
            previous = stream[index - 1]
            following = stream[index]
            if previous == "\n":
                boundaries.add(index)
            elif (
                previous in ".!?;"
                and following in " \n"
            ):
                boundaries.add(index)
        return boundaries

    @staticmethod
    def _owners_for_window(
        elements: list[ParsedElement],
        starts: list[int],
        start: int,
        end: int,
    ) -> list[ParsedElement]:
        owners: list[ParsedElement] = []
        for element, element_start in zip(
            elements,
            starts,
        ):
            element_end = element_start + len(
                element.text
            )
            if (
                element_start < end
                and element_end > start
            ):
                owners.append(element)
        return owners

    def _make_children_legacy(
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