from app.ingestion.chunker.base import (
    BaseChunker,
    Block,
)
from app.ingestion.models import ParsedDocument


class TxtChunker(BaseChunker):
    """
    TXT: little structure, so parents come from logical paragraph
    groups rather than an arbitrary whole document.

    The parser emits one element per non-blank line (with character
    offsets). A blank line (a gap >= 2 physical line numbers between
    consecutive elements) ends a paragraph run.

    Consecutive paragraph runs are grouped into one parent (a paragraph
    group), starting a new group only when it would grow past
    ``group_target_chars``. Paragraph runs are atomic for children too:
    a child chunk never spans a blank-line paragraph break, so a group
    parent typically owns several paragraph children. The 20k parent cap
    remains just a fallback for oversized groups.
    """

    identifier = "txt"

    GROUP_TARGET_CHARS = 8000

    def __init__(
        self,
        child_max_chars: int | None = None,
        parent_max_chars: int | None = None,
        group_target_chars: int | None = None,
    ):
        super().__init__(
            child_max_chars=child_max_chars,
            parent_max_chars=parent_max_chars,
        )
        self.group_target_chars = (
            group_target_chars
            or self.GROUP_TARGET_CHARS
        )

    def _blocks(
        self,
        document: ParsedDocument,
    ) -> list[Block]:

        blocks = []
        run = []
        block = []
        run_number = 0
        block_chars = 0
        previous_last_line = None

        def block_chars_of(elements) -> int:
            if not elements:
                return 0
            return (
                sum(
                    len(element.text)
                    for element in elements
                )
                + 2 * (len(elements) - 1)
            )

        def flush_block() -> None:
            nonlocal block, block_chars

            if block:

                key = ("paragraphs", run_number, len(block))

                blocks.append(
                    Block(
                        key=key,
                        elements=block,
                        meta=self.block_meta(
                            document,
                            block,
                            key,
                        ),
                    )
                )

            block = []
            block_chars = 0

        def add_run() -> None:
            nonlocal run, block, block_chars

            if not run:
                return

            run_chars = block_chars_of(run)

            if (
                block
                and block_chars + 2 + run_chars
                > self.group_target_chars
            ):
                flush_block()

            block.extend(run)

            block_chars += run_chars + (
                2 if block_chars else 0
            )

            run = []

        for element in document.elements:

            first_line, last_line = self._line_bounds(
                element
            )

            if first_line is None:
                first_line = previous_last_line

            if (
                run
                and previous_last_line is not None
                and first_line is not None
                and first_line - previous_last_line >= 2
            ):
                add_run()
                run_number += 1

            run.append(element)

            if last_line is not None or first_line is not None:
                previous_last_line = (
                    last_line
                    if last_line is not None
                    else first_line
                )

        add_run()

        flush_block()

        return blocks

    def block_meta(
        self,
        document,
        elements,
        key,
    ) -> dict:

        first_line = None
        last_line = None

        for element in elements:
            element_first, element_last = (
                self._line_bounds(element)
            )
            if element_first is not None:
                first_line = (
                    element_first
                    if first_line is None
                    else min(first_line, element_first)
                )
            if element_last is not None:
                last_line = (
                    element_last
                    if last_line is None
                    else max(last_line, element_last)
                )

        meta = {
            "paragraph_runs": self._run_count(
                elements
            )
        }

        if (
            first_line is not None
            and last_line is not None
        ):
            meta["lines"] = [
                first_line,
                last_line,
            ]

        return meta

    def split_boundaries(
        self,
        document,
        elements,
    ) -> list[int]:
        return self._run_start_indices(elements)

    def child_flush_indices(
        self,
        document,
        elements,
    ) -> frozenset[int]:
        return frozenset(
            self._run_start_indices(elements)
        )

    def _run_count(self, elements) -> int:
        return 1 + len(
            self._run_start_indices(elements)
        )

    def _run_start_indices(self, elements) -> list[int]:
        """Indices of elements that begin a new paragraph run."""
        starts = []
        previous_last_line = None

        for index, element in enumerate(elements):

            first_line, last_line = self._line_bounds(
                element
            )

            if (
                index > 0
                and previous_last_line is not None
                and first_line is not None
                and first_line - previous_last_line >= 2
            ):
                starts.append(index)

            if last_line is not None or first_line is not None:
                previous_last_line = (
                    last_line
                    if last_line is not None
                    else first_line
                )

        return starts

    @staticmethod
    def _line_bounds(element):
        first = None
        last = None
        for location in element.locations:
            if location.type != "text_offset":
                continue
            if location.line_start is not None:
                first = (
                    location.line_start
                    if first is None
                    else min(first, location.line_start)
                )
            if location.line_end is not None:
                last = (
                    location.line_end
                    if last is None
                    else max(last, location.line_end)
                )
        return first, last