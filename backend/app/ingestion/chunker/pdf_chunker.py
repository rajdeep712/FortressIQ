from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)


class PdfChunker(SectionChunker):
    """
    PDF: heading section -> parent; children preserve page/bbox locations.
    """

    identifier = "pdf"