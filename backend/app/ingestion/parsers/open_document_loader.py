from pathlib import Path

import httpx
import sys
# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.core.config import settings
from app.ingestion.models import ParsedDocument
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pdf_parser import PdfParser


class OpenDocumentLoaderParser(DocumentParser):

    name = "open_document_loader"
    version = "1.1"

    def __init__(
        self,
        pdf_parser: PdfParser | None = None,
    ):
        self.pdf_parser = pdf_parser or PdfParser()

    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        endpoint = settings.odl_api_endpoint

        if not endpoint:
            raise ValueError(
                "OpenDocumentLoader endpoint is not configured. "
                "Set ODL_API_ENDPOINT in .env."
            )

        file_bytes = file_path.read_bytes()

        timeout = httpx.Timeout(
            settings.odl_timeout_seconds
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(
                endpoint,
                files={
                    "file": (
                        filename,
                        file_bytes,
                        mime_type,
                    ),
                },
            )

        if response.is_error:

            detail = self._extract_error_detail(
                response
            )

            raise ValueError(
                "OpenDocumentLoader request failed "
                f"(HTTP {response.status_code}): {detail}"
            )

        data = self._decode_json(response)

        return self.pdf_parser.parse(
            data=data,
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
        )

    @staticmethod
    def _extract_error_detail(
        response: httpx.Response,
    ) -> str:

        try:
            body = response.json()
        except (ValueError, TypeError):
            text = response.text.strip()
            return text if text else "no detail provided"

        if isinstance(body, dict):
            for key in ("detail", "message", "error"):
                if body.get(key):
                    return str(body[key])

        return str(body)

    @staticmethod
    def _decode_json(
        response: httpx.Response,
    ) -> dict:

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                "OpenDocumentLoader returned a non-JSON response "
                f"(HTTP {response.status_code})."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "OpenDocumentLoader response must be a JSON object."
            )

        return data


if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await OpenDocumentLoaderParser().parse(
            file_path=Path(__file__).parent / "example.pdf",
            doc_id="doc123",
            version_id="v1",
            user_id="user123",
            filename="example.pdf",
            mime_type="application/pdf",
        )
        print(result)

    asyncio.run(main())