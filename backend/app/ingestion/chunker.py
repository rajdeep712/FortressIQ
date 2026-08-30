from dataclasses import dataclass, field
from uuid import uuid4

from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
    SourceLocation,
)


@dataclass
class Chunk:

    chunk_id: str

    parent_chunk_id: str | None

    chunk_type: str

    text: str

    chunk_index: int

    doc_id: str

    version_id: str

    user_id: str

    section_path: list[str]

    locations: list[SourceLocation] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )


class HierarchicalChunker:

    def __init__(
        self,
        child_max_chars: int = 2000,
    ):
        self.child_max_chars = child_max_chars

    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[Chunk]:

        groups = self._group_by_section(
            document.elements
        )

        chunks = []

        global_index = 0

        for section_path, elements in groups:

            parent_text = "\n\n".join(
                element.text
                for element in elements
            )

            parent_chunk_id = (
                f"parent_{uuid4().hex}"
            )

            parent_locations = []

            for element in elements:
                parent_locations.extend(
                    element.locations
                )

            chunks.append(
                Chunk(
                    chunk_id=parent_chunk_id,
                    parent_chunk_id=None,
                    chunk_type="parent",
                    text=parent_text,
                    chunk_index=global_index,
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    user_id=document.user_id,
                    section_path=section_path,
                    locations=parent_locations,
                    metadata={
                        "parser": document.parser_name,
                        "filename": document.filename,
                    },
                )
            )

            global_index += 1

            child_chunks = self._make_children(
                elements=elements,
                parent_chunk_id=parent_chunk_id,
                document=document,
                start_index=global_index,
                section_path=section_path,
            )

            chunks.extend(child_chunks)

            global_index += len(
                child_chunks
            )

        return chunks

    def _group_by_section(
        self,
        elements: list[ParsedElement],
    ):

        groups = []

        current_path = None
        current_elements = []

        for element in elements:

            section_path = tuple(
                element.section_path
            )

            if (
                current_path is not None
                and section_path != current_path
            ):

                groups.append(
                    (
                        list(current_path),
                        current_elements,
                    )
                )

                current_elements = []

            current_path = section_path

            current_elements.append(
                element
            )

        if current_elements:

            groups.append(
                (
                    list(current_path or []),
                    current_elements,
                )
            )

        return groups

    def _make_children(
        self,
        elements,
        parent_chunk_id,
        document,
        start_index,
        section_path,
    ):

        chunks = []

        current_text = []
        current_locations = []

        index = start_index

        for element in elements:

            prospective = "\n\n".join(
                current_text + [element.text]
            )

            if (
                current_text
                and len(prospective)
                > self.child_max_chars
            ):

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"child_{uuid4().hex}"
                        ),
                        parent_chunk_id=parent_chunk_id,
                        chunk_type="child",
                        text="\n\n".join(
                            current_text
                        ),
                        chunk_index=index,
                        doc_id=document.doc_id,
                        version_id=document.version_id,
                        user_id=document.user_id,
                        section_path=section_path,
                        locations=current_locations,
                        metadata={
                            "parser": document.parser_name,
                            "parent_chunk_id": parent_chunk_id,
                        },
                    )
                )

                index += 1

                current_text = []
                current_locations = []

            current_text.append(
                element.text
            )

            current_locations.extend(
                element.locations
            )

        if current_text:

            chunks.append(
                Chunk(
                    chunk_id=f"child_{uuid4().hex}",
                    parent_chunk_id=parent_chunk_id,
                    chunk_type="child",
                    text="\n\n".join(
                        current_text
                    ),
                    chunk_index=index,
                    doc_id=document.doc_id,
                    version_id=document.version_id,
                    user_id=document.user_id,
                    section_path=section_path,
                    locations=current_locations,
                    metadata={
                        "parser": document.parser_name,
                        "parent_chunk_id": parent_chunk_id,
                    },
                )
            )

        return chunks