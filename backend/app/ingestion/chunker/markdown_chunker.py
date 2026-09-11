from app.ingestion.chunker.base import (
    BaseChunker,
    Block,
)
from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)
from app.ingestion.models import (
    ParsedDocument,
)


class MarkdownChunker(SectionChunker):
    """
    Markdown: use the heading hierarchy directly (# -> sections).
    Code blocks are atomic: each keeps its own child chunk and is never
    merged with neighboring elements or split at the child cap.

    Heading sections are content, not hard structure: `_parent_groups`
    DP-coalesces them into banded parents (4-8 children) so short
    sections don't each become a one-child wafer parent. Section
    boundaries survive as child boundaries.
    """

    identifier = "markdown"

    atomic_element_types = frozenset(
        {"code_block"}
    )

    def block_is_seam(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> bool:
        return False