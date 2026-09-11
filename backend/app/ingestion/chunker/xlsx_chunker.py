from app.ingestion.chunker.base import BaseChunker
from app.ingestion.models import ParsedElement


class XlsxChunker(BaseChunker):
    """
    XLSX: one sheet parent per sheet.

        Workbook
         ├── Sheet Parent
         │     ├── row group
         │     └── row group
         └── another Sheet Parent

    Parent text is sheet name + schema + a small row sample. Children
    are packed row/cell-range groups; each child carries ``sheet``,
    ``cell_range`` and ``columns`` metadata.

    (Future work: one sheet may hold several logical tables -- detect
    separate table/region parents then instead of sheet -> children.)
    """

    identifier = "xlsx"

    def __init__(
        self,
        child_max_chars: int | None = None,
        parent_max_chars: int | None = None,
        parent_soft_max_chars: int | None = None,
    ):
        # Tabular formats keep the legacy character-based sizing: row
        # groups, no window overlap, no token cap.
        super().__init__(
            standard=False,
            child_max_chars=child_max_chars,
            parent_max_chars=parent_max_chars,
            parent_soft_max_chars=parent_soft_max_chars,
        )

    def block_key(self, element: ParsedElement):
        return ("sheet",) + tuple(
            element.section_path
        )

    def _section_path(self, document, block):
        if block.key:
            return list(block.key)[1:]
        return []

    def block_meta(
        self,
        document,
        elements,
        key,
    ) -> dict:

        meta = {}

        path = list(key)
        if len(path) > 1 and path[1]:
            meta["sheet"] = path[1]

        columns = None

        for element in elements:
            if columns is None and element.metadata.get(
                "columns"
            ):
                columns = element.metadata["columns"]

        if columns is not None:
            meta["columns"] = columns

        return meta

    def parent_text(
        self,
        document,
        part_elements,
        block,
    ) -> str:

        lines = []

        sheet = block.meta.get("sheet")
        if sheet:
            lines.append(f"sheet: {sheet}")

        columns = block.meta.get("columns") or []
        if columns:
            lines.append(
                "schema: " + " | ".join(columns)
            )

        sample = [
            element
            for element in part_elements
            if element.element_type != "table_header"
        ][:5]

        if sample:
            if lines:
                lines.append("")
            lines.extend(
                element.text
                for element in sample
            )

        return "\n".join(lines) or "(empty sheet)"

    def child_metadata(
        self,
        document,
        group_elements,
    ) -> dict:
        return self._group_tabular_meta(
            group_elements
        )