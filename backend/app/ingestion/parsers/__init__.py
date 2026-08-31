from app.ingestion.parsers.csv_parser import CsvParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.json_parser import JsonParser
from app.ingestion.parsers.open_document_loader import (
    OpenDocumentLoaderParser,
)
from app.ingestion.parsers.dockling import DoclingParser
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.parsers.xlsx_parser import XlsxParser


PARSERS_BY_EXTENSION = {
    ".csv": CsvParser,
    ".docx": DocxParser,
    ".md": MarkdownParser,
    ".html": HtmlParser,
    ".json": JsonParser,
    ".pdf": OpenDocumentLoaderParser,
    ".pptx": DoclingParser,
    ".txt": TxtParser,
    ".xlsx": XlsxParser,
}


def get_parser(extension: str):
    """Return the parser instance for a file extension."""
    parser_class = PARSERS_BY_EXTENSION.get(
        extension.lower()
    )
    if parser_class is None:
        raise ValueError(
            f"Unsupported document extension: {extension}"
        )
    return parser_class()