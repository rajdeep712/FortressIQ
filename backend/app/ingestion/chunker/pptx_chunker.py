from app.ingestion.chunker.base import BaseChunker
from app.ingestion.models import ParsedElement


class PptxChunker(BaseChunker):
    """
    PPTX: one parent per slide -- the slide is the semantic boundary,
    not the generic 20k rule.

         PPTX
          ├── Slide 1 Parent
          │     ├── title
          │     ├── bullet
          │     └── textbox
          └── Slide 2 Parent
                  └── ...

    Titled and untitled body elements are unified into the same slide
    parent (the parser hands body elements an empty section_path, so
    naive section grouping would split a slide apart). A slide that is
    enormous falls back to parent-part splitting.

    Children: one child per content block (title, each bullet, each
    textbox) rather than size-packing a whole slide into a single
    child, so a slide parent normally holds several granular children.
    Empty-text blocks (e.g. images with no caption) are skipped.
    """

    identifier = "pptx"

    def child_flush_indices(
        self,
        document,
        elements,
    ) -> frozenset[int]:
        return frozenset(range(1, len(elements)))

    def block_key(self, element: ParsedElement):
        slide = element.metadata.get("slide_number")

        if slide is None:
            for location in element.locations:
                if location.slide is not None:
                    slide = location.slide
                    break

        if slide is not None:
            return ("slide", slide)

        if element.section_path:
            return ("slide", tuple(element.section_path))

        return ("slide", None)

    def block_meta(
        self,
        document,
        elements,
        key,
    ) -> dict:

        slide = key[1] if len(key) > 1 else None

        title = None
        for element in elements:
            if element.section_path:
                title = element.section_path[0]
                break

        meta = {}

        if slide is not None:
            meta["slide"] = slide
        elif title is None:
            meta["slide"] = "unknown"

        if title:
            meta["slide_title"] = title

        return meta

    def _section_path(self, document, block):

        title = block.meta.get("slide_title")
        if title:
            return [title]

        slide = block.meta.get("slide")
        if slide and slide != "unknown":
            return [f"Slide {slide}"]

        return ["Slide"]

    def split_boundaries(
        self,
        document,
        elements,
    ) -> list[int]:
        return []