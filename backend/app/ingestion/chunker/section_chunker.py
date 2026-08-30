from app.ingestion.chunker.base import (
    BaseChunker,
    Block,
)
from app.ingestion.models import (
    ParsedDocument,
    ParsedElement,
)


class SectionChunker(BaseChunker):
    """
    Unstructured strategy for documents that carry heading structure:

        Document
         ├── Root / Unsectioned Parent   (content before the first
         │       └── children              heading, or no headings at all)
         ├── Section Parent
         │       └── children
         └── Section Parent
                 └── children

    A parent unit is a run of consecutive elements sharing the same
    ``section_path``. Empty ``section_path`` elements (pre-heading or
    heading-less content) form the Root/Unsectioned parent. The 20k
    parent cap is only a fallback for exceptionally long sections.

    Robustness rules:

      * Headings are structure, not content. A heading is only emitted
        as a child when its section actually has body content below it.
      * A heading-only section (a heading immediately followed by
        another heading) is a dead-end "stub": it produces no parent
        and no child of its own. Its heading text is folded into the
        NEXT section's parent as prefix context, so thin
        (heading-echo) parents never appear. Stubs at the end of the
        document are appended to the preceding section instead.
    """

    identifier = "section"

    respect_parent_soft_cap = True

    HEADING_TYPES = frozenset(
        {
            "heading",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }
    )

    def block_key(self, element: ParsedElement):
        return tuple(element.section_path)

    def _blocks(
        self,
        document: ParsedDocument,
    ) -> list[Block]:

        raw = super()._blocks(document)

        blocks = []
        pending_prefix = []

        for block in raw:

            if self._is_stub(block):

                pending_prefix.extend(
                    element.text
                    for element in block.elements
                    if element.text
                )

                continue

            if pending_prefix:
                block.meta["prefix"] = (
                    "\n\n".join(pending_prefix)
                )
                pending_prefix = []

            blocks.append(block)

        if pending_prefix:

            trailing = "\n\n".join(
                pending_prefix
            )

            if blocks:

                existing = blocks[-1].meta.get(
                    "tail"
                )

                if existing:
                    blocks[-1].meta["tail"] = (
                        existing
                        + "\n\n"
                        + trailing
                    )
                else:
                    blocks[-1].meta["tail"] = (
                        trailing
                    )

            else:
                blocks.append(
                    Block(
                        key=("root",),
                        elements=[],
                        meta={"prefix": trailing},
                    )
                )

        return blocks

    def parent_text(
        self,
        document: ParsedDocument,
        part_elements: list[ParsedElement],
        block: Block,
    ) -> str:

        prefix = block.meta.get("prefix")

        tail = block.meta.get("tail")

        body = "\n\n".join(
            element.text
            for element in part_elements
        )

        if prefix:
            body = prefix + "\n\n" + body

        if tail:
            body = body + "\n\n" + tail

        return body

    def split_boundaries(
        self,
        document,
        elements,
    ) -> list[int]:
        return []

    def _is_stub(self, block: Block) -> bool:
        if not block.elements:
            return False
        return all(
            element.element_type
            in self.HEADING_TYPES
            for element in block.elements
        )