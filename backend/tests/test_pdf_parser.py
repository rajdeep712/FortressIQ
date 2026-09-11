"""Tests for PdfParser normalization of opendataloader_pdf JSON.

The hosted OpenDocumentLoader service (opendataloader_pdf) emits list
entries under the ``"list items"`` key (with a space). Older payloads used
``"list_items"`` (underscore). PdfParser must handle both so list text is
not silently dropped.
"""

from app.ingestion.parsers.pdf_parser import PdfParser


def _make_odl_document(root_kids: list[dict]) -> dict:
    return {
        "file name": "example.pdf",
        "number of pages": 1,
        "author": None,
        "title": "Example",
        "creation date": "2024-01-01",
        "modification date": "2024-01-01",
        "kids": root_kids,
    }


def _parse(data: dict):
    return PdfParser().parse(
        data=data,
        doc_id="doc1",
        version_id="v1",
        user_id="user1",
        filename="example.pdf",
        mime_type="application/pdf",
    )


class TestListItemsKey:

    def _list_node(self, list_key: str) -> dict:
        return {
            "id": 10,
            "type": "list",
            "page number": 1,
            "bounding box": [0.0, 0.0, 100.0, 20.0],
            list_key: [
                {
                    "id": 11,
                    "type": "list item",
                    "content": "First item",
                    "page number": 1,
                    "bounding box": [0.0, 0.0, 100.0, 10.0],
                },
                {
                    "id": 12,
                    "type": "list item",
                    "content": "Second item",
                    "page number": 1,
                    "bounding box": [0.0, 0.0, 100.0, 10.0],
                },
            ],
        }

    def test_list_text_extracted_from_space_key(self):
        doc = _make_odl_document([
            {
                "id": 1,
                "type": "heading",
                "heading level": 1,
                "content": "Section",
                "page number": 1,
                "bounding box": [0.0, 0.0, 100.0, 15.0],
            },
            self._list_node("list items"),
        ])

        result = _parse(doc)
        texts = [el.text for el in result.elements]

        assert texts == ["Section", "First item", "Second item"]

    def test_list_text_extracted_from_underscore_key(self):
        doc = _make_odl_document([
            self._list_node("list_items"),
        ])

        result = _parse(doc)
        texts = [el.text for el in result.elements]

        assert texts == ["First item", "Second item"]

    def test_list_container_emits_no_phantom_element(self):
        doc = _make_odl_document([
            self._list_node("list items"),
        ])

        result = _parse(doc)

        assert all(el.text.strip() for el in result.elements)
        assert [el.element_type for el in result.elements] == [
            "list_item",
            "list_item",
        ]

    def test_nested_list_items_traversed(self):
        doc = _make_odl_document([
            {
                "id": 10,
                "type": "list",
                "page number": 1,
                "list items": [
                    {
                        "id": 11,
                        "type": "list item",
                        "content": "Parent list item",
                        "page number": 1,
                        "kids": [
                            {
                                "id": 12,
                                "type": "list item",
                                "content": "Nested list item",
                                "page number": 1,
                            },
                        ],
                    },
                ],
            },
        ])

        result = _parse(doc)
        texts = [el.text for el in result.elements]

        assert texts == ["Parent list item", "Nested list item"]


class TestBaseFields:

    def test_headings_and_paragraphs_preserved(self):
        doc = _make_odl_document([
            {
                "id": 1,
                "type": "heading",
                "heading level": 1,
                "content": "Title",
                "page number": 1,
                "bounding box": [0.0, 0.0, 100.0, 15.0],
            },
            {
                "id": 2,
                "type": "paragraph",
                "content": "Body text",
                "page number": 1,
                "bounding box": [0.0, 0.0, 100.0, 15.0],
            },
        ])

        result = _parse(doc)

        assert [el.text for el in result.elements] == [
            "Title",
            "Body text",
        ]

    def test_type_and_section_path_fields(self):
        doc = _make_odl_document([
            {
                "id": 1,
                "type": "heading",
                "heading level": 1,
                "content": "Title",
                "page number": 1,
                "bounding box": [0.0, 0.0, 100.0, 15.0],
            },
            {
                "id": 2,
                "type": "paragraph",
                "content": "Body",
                "page number": 1,
                "bounding box": [0.0, 0.0, 100.0, 15.0],
            },
        ])

        result = _parse(doc)

        title, body = result.elements

        assert title.element_type == "heading"
        assert title.section_path == ["Title"]
        assert title.locations[0].page == 1
        assert body.element_type == "paragraph"
        assert body.section_path == ["Title"]
        assert body.metadata["source_type"] == "paragraph"

    def test_missing_kids_field_raises(self):
        import pytest

        with pytest.raises(ValueError):
            _parse({"file name": "x.pdf"})


class TestTableElements:

    def _table_node(self, rows_key="rows", cells_key="cells") -> dict:
        return {
            "id": 50,
            "type": "table",
            "number of rows": 2,
            "number of columns": 2,
            "page number": 3,
            "bounding box": [0.0, 100.0, 400.0, 200.0],
            rows_key: [
                {
                    "id": 51,
                    "type": "table row",
                    "row number": 1,
                    "page number": 3,
                    cells_key: [
                        {
                            "id": 511,
                            "type": "table cell",
                            "column number": 1,
                            "bounding box": [0.0, 100.0, 200.0, 110.0],
                            "kids": [
                                {
                                    "id": 5111,
                                    "type": "paragraph",
                                    "content": "Survived",
                                    "page number": 3,
                                    "bounding box": [1.0, 100.0, 150.0, 110.0],
                                },
                            ],
                        },
                        {
                            "id": 512,
                            "type": "table cell",
                            "column number": 2,
                            "bounding box": [200.0, 100.0, 400.0, 110.0],
                            "kids": [
                                {
                                    "id": 5121,
                                    "type": "paragraph",
                                    "content": "Survival status",
                                    "page number": 3,
                                    "bounding box": [210.0, 100.0, 350.0, 110.0],
                                },
                            ],
                        },
                    ],
                },
                {
                    "id": 52,
                    "type": "table row",
                    "row number": 2,
                    "page number": 3,
                    cells_key: [
                        {
                            "id": 521,
                            "type": "table cell",
                            "column number": 1,
                            "page number": 3,
                            "kids": [
                                {
                                    "id": 5211,
                                    "type": "paragraph",
                                    "content": "0",
                                    "page number": 3,
                                },
                            ],
                        },
                        {
                            "id": 522,
                            "type": "table cell",
                            "column number": 2,
                            "page number": 3,
                            "kids": [
                                {
                                    "id": 5221,
                                    "type": "paragraph",
                                    "content": "Did not survive",
                                    "page number": 3,
                                },
                            ],
                        },
                    ],
                },
            ],
        }

    def test_table_and_rows_emitted_with_proper_links(self):
        doc = _make_odl_document([
            {
                "id": 1,
                "type": "heading",
                "heading level": 1,
                "content": "Stats",
                "page number": 3,
                "bounding box": [0.0, 50.0, 200.0, 60.0],
            },
            self._table_node(),
            {
                "id": 60,
                "type": "paragraph",
                "content": "After the table.",
                "page number": 3,
                "bounding box": [0.0, 210.0, 200.0, 220.0],
            },
        ])

        result = _parse(doc)

        assert [el.element_type for el in result.elements] == [
            "heading",
            "table",
            "table_row",
            "table_row",
            "paragraph",
        ]

        table, row1, row2 = result.elements[1:4]

        assert table.parent_element_id is None
        assert table.text == (
            "Survived | Survival status\n"
            "0 | Did not survive"
        )
        assert [row.text for row in (row1, row2)] == [
            "Survived | Survival status",
            "0 | Did not survive",
        ]
        assert row1.parent_element_id == table.element_id
        assert row2.parent_element_id == table.element_id

        # Section path inherited from the surrounding heading.
        assert table.section_path == ["Stats"]

    def test_cell_paragraphs_not_duplicated_as_flat_elements(self):
        doc = _make_odl_document([self._table_node()])

        result = _parse(doc)

        assert [el.element_type for el in result.elements] == [
            "table",
            "table_row",
            "table_row",
        ]
        assert all(el.text.strip() for el in result.elements)

    def test_row_locations_aggregated_from_cells(self):
        doc = _make_odl_document([self._table_node()])

        result = _parse(doc)

        row1 = result.elements[1]

        pages = [
            location.page
            for location in row1.locations
            if location.type == "pdf_bbox"
        ]

        assert pages == [3, 3]
        assert row1.locations[0].bbox["x1"] == 1.0
        assert row1.locations[1].bbox["x1"] == 210.0

    def test_table_metadata_dims_and_row_numbers(self):
        doc = _make_odl_document([self._table_node()])

        result = _parse(doc)

        table, row1, row2 = result.elements

        assert table.metadata["number of rows"] == 2
        assert table.metadata["number of columns"] == 2
        assert row1.metadata["row number"] == 1
        assert row2.metadata["row number"] == 2

    def test_falls_back_to_rows_and_cells_under_kids(self):
        node = self._table_node(
            rows_key="kids",
            cells_key="kids",
        )

        for row in node["kids"]:
            row["type"] = "table row"
            for cell in row["kids"]:
                cell["type"] = "table cell"

        doc = _make_odl_document([node])

        result = _parse(doc)

        assert [el.element_type for el in result.elements] == [
            "table",
            "table_row",
            "table_row",
        ]
        assert result.elements[0].text == (
            "Survived | Survival status\n"
            "0 | Did not survive"
        )