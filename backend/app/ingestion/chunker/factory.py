from app.ingestion.chunker.base import (
    BaseChunker,
)
from app.ingestion.chunker.csv_chunker import (
    CsvChunker,
)
from app.ingestion.chunker.docx_chunker import (
    DocxChunker,
)
from app.ingestion.chunker.html_chunker import (
    HtmlChunker,
)
from app.ingestion.chunker.json_chunker import (
    JsonChunker,
)
from app.ingestion.chunker.markdown_chunker import (
    MarkdownChunker,
)
from app.ingestion.chunker.pdf_chunker import (
    PdfChunker,
)
from app.ingestion.chunker.pptx_chunker import (
    PptxChunker,
)
from app.ingestion.chunker.section_chunker import (
    SectionChunker,
)
from app.ingestion.chunker.txt_chunker import (
    TxtChunker,
)
from app.ingestion.chunker.xlsx_chunker import (
    XlsxChunker,
)
from app.ingestion.models import ParsedDocument


def get_chunker_for(
    document: ParsedDocument,
) -> BaseChunker:
    return get_chunker(
        mime_type=document.mime_type,
        filename=document.filename,
    )


def get_chunker(
    mime_type: str | None = None,
    filename: str | None = None,
) -> BaseChunker:

    mime = (
        mime_type or ""
    ).lower().split(";")[0].strip()

    suffix = (filename or "").lower()

    if (
        "wordprocessingml.document" in mime
        or suffix.endswith(".docx")
    ):
        return DocxChunker()

    if (
        "presentationml.presentation" in mime
        or suffix.endswith(".pptx")
    ):
        return PptxChunker()

    if (
        "spreadsheetml.sheet" in mime
        or suffix.endswith(".xlsx")
        or suffix.endswith(".xlsm")
    ):
        return XlsxChunker()

    if (
        mime == "text/csv"
        or suffix.endswith(".csv")
    ):
        return CsvChunker()

    if (
        "json" in mime
        or any(
            suffix.endswith(ext)
            for ext in (".json", ".jsonl", ".ndjson")
        )
    ):
        return JsonChunker()

    if (
        mime == "text/plain"
        or suffix.endswith(".txt")
    ):
        return TxtChunker()

    if (
        "markdown" in mime
        or any(
            suffix.endswith(ext)
            for ext in (".md", ".markdown")
        )
    ):
        return MarkdownChunker()

    if (
        "html" in mime
        or any(
            suffix.endswith(ext)
            for ext in (".html", ".htm")
        )
    ):
        return HtmlChunker()

    if (
        "pdf" in mime
        or suffix.endswith(".pdf")
    ):
        return PdfChunker()

    return SectionChunker()


def chunk_document(
    document: ParsedDocument,
) -> list:
    return get_chunker_for(document).chunk(
        document
    )