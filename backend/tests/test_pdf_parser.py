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