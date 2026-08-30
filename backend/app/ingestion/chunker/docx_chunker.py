from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)


class DocxChunker(SectionChunker):
    """
    DOCX: heading -> parent sections; children preserve rendered
    page/bbox provenance.
    """

    identifier = "docx"