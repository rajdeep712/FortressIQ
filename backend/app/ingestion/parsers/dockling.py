from pathlib import Path
import json
import sys
# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import httpx

from app.core.config import settings
from app.ingestion.models import ParsedDocument
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pptx_parser import PptxParser


class DoclingParser(DocumentParser):
    """
    HTTP transport for the self-hosted Dockling API.

    This parser:
        - does NOT normalize the Dockling response
        - does NOT read PPTX directly
        - does NOT chunk
        - does NOT embed

    It only calls the Dockling endpoint (securing the request with the
    x-api-key header) and hands the JSON response to PptxParser, which
    performs the actual conversion to ParsedDocument.
    """

    name = "docling"
    version = "1.2"

    def __init__(
        self,
        pptx_parser: PptxParser | None = None,
    ):
        self.pptx_parser = pptx_parser or PptxParser()

    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        endpoint = settings.docling_api_endpoint

        if not endpoint:
            raise ValueError(
                "Docling endpoint is not configured. "
                "Set DOCLING_API_ENDPOINT in .env."
            )

        api_key = settings.docling_api_key

        if not api_key:
            raise ValueError(
                "Docling API key is not configured. "
                "Set DOCLING_API_KEY in .env."
            )

        file_bytes = file_path.read_bytes()

        timeout = httpx.Timeout(
            settings.docling_timeout_seconds
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(
                endpoint,
                headers={
                    "x-api-key": api_key,
                },
                data={
                    "from_formats": "pptx",
                    "to_formats": "json",
                    "abort_on_error": "false",
                },
                files={
                    "files": (
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
                "Docling request failed "
                f"(HTTP {response.status_code}): {detail}"
            )

        data = self._decode_json(response)

        document = self._extract_document(
            data
        )

        return self.pptx_parser.parse(
            data=document,
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
        )

    @staticmethod
    def _extract_document(
        data: dict,
    ) -> dict:
        """
        Locate the Docling document within the server response.

        docling-serve wraps the document as:

            {"document": {"json_content": <docling-json>, ...}, "task": ...}

        so we unwrap to the object PptxParser expects.

        Custom wrappers that already return the docling JSON document
        (with a "document" key) are handled as a fallback.
        """

        wrapper = data.get(
            "document",
        )

        if isinstance(
            wrapper,
            dict,
        ):

            inner = wrapper.get(
                "json_content",
            )

            if isinstance(
                inner,
                dict,
            ):

                return {
                    "document": inner,
                }

            if isinstance(
                inner,
                str,
            ):

                try:

                    decoded = json.loads(
                        inner
                    )

                except TypeError:
                    decoded = None

                except json.JSONDecodeError:
                    decoded = None

                if isinstance(
                    decoded,
                    dict,
                ):

                    return {
                        "document": decoded,
                    }

            if "document" in data or (
                "body" in wrapper
                or "texts" in wrapper
                or "groups" in wrapper
            ):

                return {
                    "document": wrapper,
                }

        raise ValueError(
            "Docling response contains no recognizable "
            "document JSON."
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
                "Docling returned a non-JSON response "
                f"(HTTP {response.status_code})."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Docling response must be a JSON object."
            )

        return data

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m app.ingestion.parsers.dockling <file.pptx>")
            return
        file_path = Path(sys.argv[1])
        parser = DoclingParser()
        result = await parser.parse(
            file_path=file_path,
            doc_id="test",
            version_id="test",
            user_id="test",
            filename=file_path.name,
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
        print(result)

    asyncio.run(main())