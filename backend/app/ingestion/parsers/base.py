from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.models import ParsedDocument


class DocumentParser(ABC):
    """
    Interface for document parsers.
    All parsers must implement this interface.
    """
    name: str = "unknown"
    version: str = "1.0"

    @abstractmethod
    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:
        raise NotImplementedError