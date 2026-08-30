from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)


class HtmlChunker(SectionChunker):
    """
    HTML: use the DOM/heading hierarchy (h1-h6 -> sections) instead of
    character-only chunking. Paragraphs, list items and table rows are
    packed as children within each section.
    """

    identifier = "html"