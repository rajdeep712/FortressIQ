import sys
from pathlib import Path
from uuid import uuid4

# Add backend directory to sys.path when running this file directly
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from openpyxl import load_workbook

from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.csv_parser import (
    _make_unique_headers,
)
from app.ingestion.parsers.text_utils import (
    MAX_LINE_CHARS,
    split_long_line,
)


def _cell_text(value) -> str:
    """Normalize a single cell value to searchable text."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    return str(value).strip()


def _split_value(value: str) -> list[str]:
    """Split one cell value losslessly into <=MAX_LINE_CHARS chunks."""
    pieces = []

    for fragment, _ in split_long_line(value):
        while len(fragment) > MAX_LINE_CHARS:
            pieces.append(
                fragment[:MAX_LINE_CHARS]
            )
            fragment = fragment[MAX_LINE_CHARS:]
        if fragment:
            pieces.append(fragment)

    return pieces


def _split_text(parts: list[str]) -> list[str]:
    """
    Split long cell parts losslessly and re-pack them into lines
    no longer than MAX_LINE_CHARS.

    The column label ("Name: ...") stays attached to every chunk of
    its value, so no cell content is ever truncated and each fragment
    remains self-describing.
    """
    fragments = []

    for part in parts:

        label, separator, value = part.partition(
            ": "
        )

        if not separator:
            label = None
            value = part

        if label is not None and label:
            budget = (
                MAX_LINE_CHARS
                - len(label)
                - 2
            )
        else:
            budget = MAX_LINE_CHARS

        for chunk in _split_value(value):

            if budget <= 0:
                fragments.append(chunk)
                continue

            remaining = chunk

            while len(remaining) > budget:
                fragments.append(
                    f"{label}: {remaining[:budget]}"
                )
                remaining = remaining[budget:]

            fragments.append(
                f"{label}: {remaining}"
            )

    lines = []
    current = ""

    for fragment in fragments:

        if not current:
            current = fragment

        elif (
            len(current) + 3 + len(fragment)
            <= MAX_LINE_CHARS
        ):
            current = (
                current + " | " + fragment
            )

        else:
            lines.append(current)
            current = fragment

    if current:
        lines.append(current)

    return lines


class XlsxParser(DocumentParser):

    name = "xlsx"
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

        elements = []
        order = 0
        row_count = 0
        char_count = 0

        workbook = None
        sheet_names: list[str] = []

        try:

            try:
                workbook = load_workbook(
                    filename=file_path,
                    read_only=True,
                    data_only=True,
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid Excel file: {exc}"
                ) from exc

            for worksheet in workbook.worksheets:

                section_path = [worksheet.title]

                sheet_names.append(
                    worksheet.title
                )

                sheet_headers: list[str] = []
                seen_header = False

                for row in worksheet.iter_rows():

                    values = [
                        cell.value
                        for cell in row
                    ]

                    texts = [
                        _cell_text(value)
                        for value in values
                    ]

                    nonblank = [
                        text
                        for text in texts
                        if text
                    ]

                    if not nonblank:
                        continue

                    filled_cells = [
                        cell
                        for cell in row
                        if cell.value is not None
                    ]

                    first_cell = filled_cells[0]
                    last_cell = filled_cells[-1]

                    cell_range = (
                        f"{first_cell.coordinate}:"
                        f"{last_cell.coordinate}"
                    )

                    location = SourceLocation(
                        type="xlsx_range",
                        sheet=worksheet.title,
                        cell_range=cell_range,
                    )

                    # --------------------------------------
                    # First non-blank row = the header row.
                    # --------------------------------------

                    if not seen_header:

                        seen_header = True

                        order += 1

                        headers = _make_unique_headers(
                            [
                                _cell_text(cell.value)
                                for cell in filled_cells
                            ]
                        )

                        sheet_headers = headers

                        header_text = " | ".join(
                            header
                            for header in headers
                            if header
                        )

                        char_count += len(
                            header_text
                        )

                        elements.append(
                            ParsedElement(
                                element_id=str(uuid4()),
                                element_type="table_header",
                                text=header_text,
                                order=order,
                                section_path=section_path,
                                locations=[location],
                                metadata={
                                    "columns": headers,
                                },
                            )
                        )

                        continue

                    # --------------------------------------
                    # Data row -> "header: value" parts.
                    # --------------------------------------

                    parts = []

                    padded = texts[:] + [""] * (
                        len(sheet_headers) - len(texts)
                    )

                    for header, text in zip(
                        sheet_headers,
                        padded,
                    ):
                        if text:
                            parts.append(
                                f"{header}: {text}"
                            )

                    for index, text in enumerate(
                        texts[len(sheet_headers):],
                        start=1,
                    ):
                        if text:
                            parts.append(
                                f"extra_{index}: {text}"
                            )

                    if not parts:
                        continue

                    for fragment in _split_text(parts):

                        order += 1
                        char_count += len(fragment)

                        elements.append(
                            ParsedElement(
                                element_id=str(uuid4()),
                                element_type="spreadsheet_row",
                                text=fragment,
                                order=order,
                                section_path=section_path,
                                locations=[location],
                                metadata={
                                    "columns": sheet_headers,
                                },
                            )
                        )

                    row_count += 1

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                f"Failed to read Excel file: {exc}"
            ) from exc

        finally:
            if workbook is not None:
                try:
                    workbook.close()
                except Exception:
                    pass

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
                "sheet_count": len(sheet_names),
                "sheet_names": sheet_names,
                "row_count": row_count,
                "char_count": char_count,
            },
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await XlsxParser().parse(
            file_path=Path(__file__).parent / "example.xlsx",
            doc_id="doc123",
            version_id="v1",
            user_id="user123",
            filename="example.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        print(result)

    asyncio.run(main())