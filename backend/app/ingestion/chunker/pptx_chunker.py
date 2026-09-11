from app.ingestion.chunker.base import (
    BaseChunker,
    Block,
)
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
)


class PptxChunker(BaseChunker):
    """
    PPTX: one parent per slide-titled content run, DP-coalesced so
    short slides share a banded parent instead of each becoming a
    one-child wafer parent. Slide boundaries survive as child
    boundaries (a child never spans a slide break).

         PPTX
          ├── Parent  (covers slides/groups of blocks)
          │     └── windowed children
          └── ...

    Titled and untitled body elements are unified into the same slide
    block (the parser hands body elements an empty section_path, so
    naive section grouping would split a slide apart). A slide block
    that is enormous falls back to part splitting.

    Children: one child per content block (title, each bullet, each
    textbox) rather than size-packing a whole slide into a single
    child, so a parent normally holds several granular children.
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

    def block_is_seam(
        self,
        document: ParsedDocument,
        block: Block,
    ) -> bool:
        # Slide blocks are content, not hard structure: `_parent_groups`
        # DP-coalesces them into banded parents (4-8 children) so a
        # short deck doesn't fragment into one-child slide wafers. Slide
        # boundaries survive as child boundaries.
        return False

    def _combine_blocks(
        self,
        document: ParsedDocument,
        blocks: list[Block],
    ) -> Block:
        if len(blocks) == 1:
            return blocks[0]

        elements = []
        slides = []
        titles = []
        for block in blocks:
            elements.extend(block.elements)
            slide = block.meta.get("slide")
            if slide is not None and slide not in ("unknown",):
                slides.append(slide)
            title = block.meta.get("slide_title")
            if title and title not in titles:
                titles.append(title)

        meta = {"spans": len(blocks)}
        if slides:
            meta["slides"] = slides
            meta["slide_first"] = min(slides)
            meta["slide_last"] = max(slides)
        if titles:
            meta["slide_title"] = " / ".join(titles)

        return Block(
            key=("coalesced", "slide"),
            elements=elements,
            meta=meta,
        )

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

        slides = block.meta.get("slides")
        if slides:
            return [f"Slides {slides[0]}-{slides[-1]}"]

        return ["Slide"]

    def split_boundaries(
        self,
        document,
        elements,
    ) -> list[int]:
        return []