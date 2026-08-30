from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)


class MarkdownChunker(SectionChunker):
    """
    Markdown: use the heading hierarchy directly (# -> sections).
    Code blocks are atomic: each keeps its own child chunk and is never
    merged with neighboring elements or split at the child cap.
    """

    identifier = "markdown"

    atomic_element_types = frozenset(
        {"code_block"}
    )