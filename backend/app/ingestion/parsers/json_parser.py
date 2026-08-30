import json
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
    split_long_line,
)


def _reject_constant(token: str):
    """
    Reject NaN / Infinity / -Infinity, which Python's json module
    accepts by default but are invalid per the JSON spec.
    """
    raise ValueError(f"Non-standard JSON constant: {token}")


def iter_json_leaves(root):
    """
    Iteratively traverse a parsed JSON tree in document (pre-)order.

    Yields (path, value, label, depth) where:
      - path: JSON-path-ish pointer (e.g. "$.users[0].name")
      - value: the leaf value
      - label: the immediate dict key for object leaves, None otherwise
      - depth: nesting depth of the leaf

    Iterative (explicit stack) so deeply nested documents cannot blow
    the Python recursion limit.
    """
    stack = [(root, "$", None, 1)]

    while stack:
        node, path, label, depth = stack.pop()

        if isinstance(node, dict):
            for key in reversed(list(node)):
                stack.append(
                    (node[key], f"{path}.{key}", key, depth + 1)
                )

        elif isinstance(node, list):
            for index in reversed(range(len(node))):
                stack.append(
                    (node[index], f"{path}[{index}]", None, depth + 1)
                )

        else:
            yield path, node, label, depth


def _leaf_text(value, label: str | None) -> list[str]:
    """
    Produce the element text for a JSON leaf value.

    Returns a list so long string values can be split into multiple
    fragments (each still under the chunker's size limit).
    """
    if isinstance(value, str):
        fragments = [
            fragment
            for fragment, _ in split_long_line(value)
        ]
    else:
        # json.dumps normalizes true/false/null and non-string scalars
        # to spec-correct representations.
        fragments = [
            json.dumps(value, ensure_ascii=False)
        ]

    if label:
        fragments = [
            f"{label}: {fragment}"
            for fragment in fragments
        ]

    return fragments


class JsonParser(DocumentParser):

    name = "json"
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

        raw = decode_text(file_path)

        elements = []
        value_count = 0
        char_count = 0
        max_depth = 0
        document_format = "json"

        if not raw.strip():
            return ParsedDocument(
                doc_id=doc_id,
                version_id=version_id,
                user_id=user_id,
                filename=filename,
                mime_type=mime_type,
                parser_name=self.name,
                parser_version=self.version,
                elements=[],
                metadata={
                    "format": "json",
                    "value_count": 0,
                    "char_count": 0,
                    "max_depth": 0,
                    "line_count": 0,
                },
            )

        def consume_leaves(
            root,
            path_prefix: str = "",
        ) -> None:
            """Build elements from a parsed JSON tree."""
            nonlocal value_count, char_count, max_depth

            for path, value, label, depth in iter_json_leaves(root):
                max_depth = max(max_depth, depth)

                fragments = _leaf_text(value, label)

                for fragment in fragments:
                    value_count += 1
                    char_count += len(fragment)

                    elements.append(
                        ParsedElement(
                            element_id=str(uuid4()),
                            element_type="json_value",
                            text=fragment,
                            order=value_count,
                            locations=[
                                SourceLocation(
                                    type="json_path",
                                    json_path=(
                                        f"{path_prefix}{path}"
                                    ),
                                )
                            ],
                        )
                    )

        try:
            data = json.loads(
                raw,
                parse_constant=_reject_constant,
            )
            consume_leaves(data)

        except ValueError as exc:
            # Whole-document parse failed -- try JSON Lines fallback.
            line_count = 0
            parsed_lines = 0
            jsonl_success = True

            for line_number, line in enumerate(
                raw.splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue

                line_count += 1

                try:
                    line_data = json.loads(
                        line,
                        parse_constant=_reject_constant,
                    )
                except ValueError as line_exc:
                    column = getattr(line_exc, "colno", None)
                    location = (
                        f"(line {line_number}, column {column})"
                        if column is not None
                        else f"(line {line_number})"
                    )
                    raise ValueError(
                        "Invalid JSON document: "
                        f"{location}: {line_exc}"
                    )

                consume_leaves(
                    line_data,
                    path_prefix=f"L{line_number}.",
                )
                parsed_lines += 1

            document_format = "jsonl"
            line_count = parsed_lines

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
                "format": document_format,
                "value_count": value_count,
                "char_count": char_count,
                "max_depth": max_depth,
                "line_count": (
                    len(raw.splitlines())
                    if document_format == "json"
                    else line_count
                ),
            },
        )


if __name__ == "__main__":
    import asyncio

    parser = JsonParser()
    example_path = Path(__file__).parent / "example.json"

    result = asyncio.run(
        parser.parse(
            file_path=example_path,
            doc_id="123",
            version_id="123",
            user_id="123",
            filename="example.json",
            mime_type="application/json",
        )
    )

    print(result)