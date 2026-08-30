from app.ingestion.chunker.base import BaseChunker
from app.ingestion.models import ParsedElement


class CsvChunker(BaseChunker):
    """
    CSV: one table parent per file.

        Table Parent          (schema + column names + dataset context,
         ├── rows 1-N           NOT all rows)
         ├── rows N+1-M
         └── ...

    Parent text is the schema plus a small sample of rows. Children are
    packed row groups; each child carries ``columns`` / ``row_start`` /
    ``row_end`` metadata derived from its own rows.
    """

    identifier = "csv"

    TABLE_KEY = ("table",)

    def block_key(self, element: ParsedElement):
        return self.TABLE_KEY

    def block_meta(
        self,
        document,
        elements,
        key,
    ) -> dict:

        meta = {}

        columns = None

        for element in elements:
            if columns is None and element.metadata.get(
                "columns"
            ):
                columns = element.metadata["columns"]

        if columns is not None:
            meta["columns"] = columns

        row_count = document.metadata.get("row_count")
        if row_count is not None:
            meta["row_count"] = row_count

        return meta

    def parent_text(
        self,
        document,
        part_elements,
        block,
    ) -> str:

        lines = []

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

        return "\n".join(lines) or "(empty table)"

    def child_metadata(
        self,
        document,
        group_elements,
    ) -> dict:
        return self._group_tabular_meta(
            group_elements
        )