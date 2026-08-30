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
    parse_heading,
    parse_fence,
)


class MarkdownParser(DocumentParser):
    """
    Parser for Markdown (.md) documents.

    Produces `heading`, `paragraph` and `code_block` elements with
    accurate character offsets. ATX headings follow CommonMark-ish
    rules (1-6 leading #, optional space, optional closing hashes),
    and fenced code blocks (``` / ~~~) are tracked so that "# comments"
    inside code are not mistaken for headings.
    """

    name = "markdown"
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
        order = 0

        current_section: list[str] = []
        heading_count = 0
        paragraph_count = 0
        code_block_count = 0

        # Fenced code state: marker char ("`" or "~") or None.
        fence_marker = None
        fence_lines: list[tuple[int, str, int, int]] = []

        def flush_fence():
            nonlocal fence_marker, fence_lines, code_block_count, order
            if fence_marker is None or not fence_lines:
                fence_marker = None
                fence_lines = []
                return

            start_line = fence_lines[0][0]
            end_line = fence_lines[-1][0]
            start_char = fence_lines[0][2]
            end_char = fence_lines[-1][3]

            order += 1
            elements.append(
                ParsedElement(
                    element_id=str(uuid4()),
                    element_type="code_block",
                    text="\n".join(
                        ln for _, ln, _, _ in fence_lines
                    ),
                    order=order,
                    section_path=list(current_section),
                    locations=[
                        SourceLocation(
                            type="text_offset",
                            line_start=start_line,
                            line_end=end_line,
                            char_start=start_char,
                            char_end=end_char,
                        )
                    ],
                )
            )
            code_block_count += 1
            fence_marker = None
            fence_lines = []

        for (
            line_number,
            line,
            char_start,
            char_end,
        ) in iter_lines_with_offsets(text):

            # Inside a fenced code block: accumulate, ignore heading rules.
            if fence_marker is not None:
                fence_lines.append(
                    (line_number, line, char_start, char_end)
                )

                # A closing fence uses the same marker family.
                closing = parse_fence(line)
                if closing and closing[0] == fence_marker:
                    flush_fence()
                elif closing is None and not line.strip():
                    # blank lines are fine inside code; keep accumulating
                    pass
                continue

            if not line.strip():
                continue

            # Opening fence?
            opening = parse_fence(line)
            if opening:
                fence_marker = opening[0]
                fence_lines = [
                    (line_number, line, char_start, char_end)
                ]
                # A single-line fence (open == close on same line) is rare;
                # treat as open and let a later line close it.
                continue

            # Heading?
            heading = parse_heading(line)
            if heading is not None:
                level, title = heading

                # Keep section path aligned with the heading level,
                # clamping skipped levels gracefully.
                if level - 1 < len(current_section):
                    current_section = current_section[: level - 1]
                else:
                    current_section = current_section[: len(current_section)]
                current_section.append(title)

                order += 1
                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type="heading",
                        text=title,
                        order=order,
                        section_path=list(current_section),
                        locations=[
                            SourceLocation(
                                type="text_offset",
                                line_start=line_number,
                                line_end=line_number,
                                char_start=char_start,
                                char_end=char_end,
                            )
                        ],
                    )
                )
                heading_count += 1
                continue

            # Paragraph.
            order += 1
            elements.append(
                ParsedElement(
                    element_id=str(uuid4()),
                    element_type="paragraph",
                    text=line,
                    order=order,
                    section_path=list(current_section),
                    locations=[
                        SourceLocation(
                            type="text_offset",
                            line_start=line_number,
                            line_end=line_number,
                            char_start=char_start,
                            char_end=char_end,
                        )
                    ],
                )
            )
            paragraph_count += 1

        # File may end while a fence is still open.
        flush_fence()

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
                "heading_count": heading_count,
                "paragraph_count": paragraph_count,
                "code_block_count": code_block_count,
                "line_count": len(text.splitlines()),
                "char_count": len(text),
            },
        )


if __name__ == "__main__":
    import asyncio

    markdown_parser = MarkdownParser()
    example_path = Path(__file__).parent / "example.md"

    result = asyncio.run(
        markdown_parser.parse(
            file_path=example_path,
            doc_id="123",
            version_id="123",
            user_id="123",
            filename="example.md",
            mime_type="text/markdown",
        )
    )
    print(result)
