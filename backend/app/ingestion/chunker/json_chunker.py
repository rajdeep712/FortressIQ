import re

from app.ingestion.chunker.base import (
    BaseChunker,
    Block,
)
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
)


def _top_context(path: str | None) -> str:
    """
    Top-level segment of a json_path: "$.users[0].name" -> "users",
    JSONL "L3.users[0].name" -> "L3", scalar root "$" -> "root".
    """
    if not path:
        return "root"

    stripped = path.lstrip("$.")

    if not stripped:
        return "root"

    match = re.match(
        r"[A-Za-z_][A-Za-z0-9_]*",
        stripped,
    )

    if match:
        return match.group(0)

    return "root"


def _container_path(path: str | None) -> str | None:
    """
    Nearest object/array-item container of a leaf path, e.g.
    "$.users[0].name" -> "$.users[0]"  and  "$.users[0]" -> "$.users[0]".
    Array-item and object-item boundaries fall where this changes.
    """
    if not path:
        return None
    return re.sub(r"\.[^.\[\]]+$", "", path)


class JsonChunker(BaseChunker):
    """
    JSON: parent per logical object/array.

         Root/Object Parent      (top-level key, or one JSONL line)
          ├── nested child groups
          └── ...

    Small JSON objects keep one parent per top-level key. Oversized
    structures (e.g. a ``users`` array of 100k items) split into
    multiple parents at object/array-item boundaries -- never one
    gigantic parent. Children preserve their JSON paths.
    """

    identifier = "json"

    respect_parent_soft_cap = True

    def block_key(self, element: ParsedElement):
        return (
            "root",
            _top_context(self._path_of(element)),
        )

    def block_is_seam(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> bool:
        # One parent per top-level JSON key: keys are structure and never
        # coalesce with neighbors (oversized keys still split into parts).
        return True

    def block_meta(
        self,
        document,
        elements,
        key,
    ) -> dict:
        root = key[1] if len(key) > 1 else "root"
        return {
            "root": root,
            "value_count": len(elements),
        }

    def split_boundaries(
        self,
        document,
        elements,
    ) -> list[int]:

        boundaries = []
        previous = None

        for index, element in enumerate(elements):

            container = _container_path(
                self._path_of(element)
            )

            if (
                previous is not None
                and container != previous
            ):
                boundaries.append(index)

            previous = container

        return boundaries

    def _section_path(self, document, block):
        root = block.meta.get("root")
        return [root] if root else []

    def parent_text(
        self,
        document,
        part_elements,
        block,
    ) -> str:

        root = block.meta.get("root") or "root"

        lines = [f"root: {root}"]

        sample = "\n".join(
            element.text
            for element in part_elements[:8]
        )

        if sample:
            lines.append("")
            lines.append(sample)

        return "\n".join(lines)

    @staticmethod
    def _path_of(element) -> str | None:
        for location in element.locations:
            if (
                location.type == "json_path"
                and location.json_path
            ):
                return location.json_path
        return None