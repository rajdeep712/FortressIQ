import csv
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

# Cell values longer than this are truncated so a single long cell
# does not create an over-limit chunk (chunker child_max_chars=2000).
MAX_CELL_CHARS = 500

# Only the first SAMPLE_SIZE characters are sniff-tested for a delimiter.
SAMPLE_SIZE = 8192

# Candidate delimiters for auto-detection, in preference order.
SNIFF_DELIMITERS = ",;\t|"


def _make_unique_headers(raw_header) -> list[str]:
    """
    Disambiguate duplicate column names so later occurrences are not
    silently dropped (e.g. ["name", "name"] -> ["name", "name_2"]).
    """
    seen: dict[str, int] = {}
    result: list[str] = []

    for name in raw_header:
        name = (name or "").strip()
        if not name:
            name = "column"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1

        result.append(name)

    return result


def _sniff_dialect(sample: str):
    """
    Detect the CSV dialect (delimiter/quotechar/quoting) from a sample.
    Returns None on failure, so the caller can fall back to comma.
    """
    if not sample:
        return None

    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=SNIFF_DELIMITERS,
        )
    except csv.Error:
        return None


def _clip(value: str) -> str:
    """Trim and truncate a single cell value."""
    value = (value or "").strip()
    if len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "..."
    return value


class CsvParser(DocumentParser):

    name = "csv"
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
        row_count = 0
        char_count = 0

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as file:

            # -----------------------------------------------------------------
            # Dialect detection (delimiter / quoting) with comma fallback.
            # -----------------------------------------------------------------

            sample = file.read(SAMPLE_SIZE)
            file.seek(0)

            dialect = _sniff_dialect(sample)

            if dialect is not None:
                reader = csv.reader(file, dialect=dialect)
                delimiter = dialect.delimiter
            else:
                reader = csv.reader(file, delimiter=",")
                delimiter = ","

            try:
                raw_header = next(reader, None) or []
            except csv.Error as exc:
                raise ValueError(
                    f"Invalid CSV header row: {exc}"
                )

            headers = _make_unique_headers(raw_header)

            # -----------------------------------------------------------------
            # Header element so column names are searchable/embeddable, even
            # when the file is header-only.
            # -----------------------------------------------------------------

            if headers:
                header_text = " | ".join(headers)
                char_count += len(header_text)

                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type="table_header",
                        text=header_text,
                        order=1,
                        locations=[
                            SourceLocation(
                                type="csv_row",
                                row_start=1,
                                row_end=1,
                            )
                        ],
                        metadata={
                            "columns": headers,
                        },
                    )
                )

            # -----------------------------------------------------------------
            # Data rows.
            # -----------------------------------------------------------------

            for row_number, row in enumerate(
                reader,
                start=2,
            ):

                try:
                    cells = list(row)
                except csv.Error as exc:
                    raise ValueError(
                        f"Malformed CSV row {row_number}: {exc}"
                    )

                # Rows shorter than the header get blank padding; rows longer
                # keep their surplus cells as "extra_N".
                padded = cells[:] + [""] * (
                    len(headers) - len(cells)
                )

                parts = [
                    f"{header}: {_clip(value)}"
                    for header, value in zip(
                        headers,
                        padded,
                    )
                    if (value or "").strip()
                ]

                extras = cells[len(headers):]
                for index, extra in enumerate(
                    extras,
                    start=1,
                ):
                    clipped = _clip(extra)
                    if clipped:
                        parts.append(
                            f"extra_{index}: {clipped}"
                        )

                # Skip fully-blank rows.
                if not parts:
                    continue

                text = " | ".join(parts)
                char_count += len(text)
                row_count += 1

                elements.append(
                    ParsedElement(
                        element_id=str(uuid4()),
                        element_type="table_row",
                        text=text,
                        order=row_number,
                        locations=[
                            SourceLocation(
                                type="csv_row",
                                row_start=row_number,
                                row_end=row_number,
                            )
                        ],
                        metadata={
                            "columns": headers,
                        },
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
                "header": headers,
                "delimiter": delimiter,
                "encoding": "utf-8",
                "row_count": row_count,
                "char_count": char_count,
            },
        )


if __name__ == "__main__":
    import asyncio

    csv_parser = CsvParser()
    example_path = Path(__file__).parent / "example.csv"

    result = asyncio.run(
        csv_parser.parse(
            file_path=example_path,
            doc_id="123",
            version_id="123",
            user_id="123",
            filename="example.csv",
            mime_type="text/csv",
        )
    )

    print(result)