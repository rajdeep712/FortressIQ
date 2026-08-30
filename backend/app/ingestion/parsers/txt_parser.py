import sys
from pathlib import Path
from uuid import uuid4

# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.text_utils import (
    decode_text,
    iter_lines_with_offsets,
    split_long_line,
)


class TxtParser(DocumentParser):
    """
    Parser for .txt files.

    Produces one `paragraph` element per non-blank line, with accurate
    character offsets for citation. Very long lines are split into
    sub-elements so they do not exceed the chunker's size limit.
    """

    name = "txt"
    version = "1.1"

    async def parse(
        self,
        file_path: Path,
        doc_id: str,
        version_id: str,
        user_id: str,
        filename: str,
        mime_type: str,
    ) -> ParsedDocument:

        text = decode_text(file_path)

        elements = []
        long_lines_split = 0
        order = 0

        for (
            line_number,
            line,
            char_start,
            char_end,
        ) in iter_lines_with_offsets(text):

            if not line.strip():
                continue

            fragments = split_long_line(line)

            if len(fragments) > 1:
                long_lines_split += 1

            for fragment, frag_offset in fragments:
                order += 1

                frag_start = char_start + frag_offset
                frag_end = frag_start + len(fragment)

                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type="paragraph",
                        text=fragment,
                        order=order,
                        locations=[
                            SourceLocation(
                                type="text_offset",
                                line_start=line_number,
                                line_end=line_number,
                                char_start=frag_start,
                                char_end=frag_end,
                            )
                        ],
                    )
                )

        return ParsedDocument(
            doc_id=doc_id,
            version_id=version_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_type,
            parser_name=self.name,
            parser_version=self.version,
            elements=elements,
            metadata={
                "line_count": len(text.splitlines()),
                "char_count": len(text),
                "long_lines_split": long_lines_split,
            },
        )


if __name__ == "__main__":
    import asyncio

    txt_parser = TxtParser()
    example_path = Path(__file__).parent / "example.txt"

    result = asyncio.run(
        txt_parser.parse(
            file_path=example_path,
            doc_id="123",
            version_id="123",
            user_id="123",
            filename="example.txt",
            mime_type="text/plain",
        )
    )
    print(result)
