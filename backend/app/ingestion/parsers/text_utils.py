import re
from pathlib import Path


# Lines longer than this are split into sub-elements on whitespace
# boundaries so a single monolithic line does not exceed the chunker's
# child_max_chars limit (2000) and blow up a single vector chunk.
MAX_LINE_CHARS = 2000


# Characters that splitlines(keepends=True) treats as line terminators.
_LINE_TERMINATORS = (
    "\r",
    "\n",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)

_LINE_TERMINATOR_CHARS = "".join(_LINE_TERMINATORS)


def decode_text(file_path: Path) -> str:
    """
    Read a text-based file as UTF-8, stripping a leading BOM and
    replacing invalid bytes instead of raising.
    """
    raw = file_path.read_bytes()
    return raw.decode("utf-8-sig", errors="replace")


def _strip_terminator(line_with_end: str) -> str:
    """
    Remove trailing line terminators (e.g. CRLF -> strip both \r and \n)
    from a line produced by splitlines(keepends=True) so char offsets
    and element text stay accurate.
    """
    return line_with_end.rstrip(_LINE_TERMINATOR_CHARS)


def iter_lines_with_offsets(
    text: str,
) -> list[tuple[int, str, int, int]]:
    """
    Return (line_number, line_text, char_start, char_end) for every
    physical line in `text`.

    A single O(n) pass with a running offset is used so char offsets
    always point at the true position of the line in the document
    (unlike str.find, which returns the first matching occurrence and
    breaks when lines repeat).
    """
    results = []

    offset = 0
    for index, line_with_end in enumerate(
        text.splitlines(keepends=True),
        start=1,
    ):
        line = _strip_terminator(line_with_end)
        char_start = offset
        char_end = offset + len(line)
        results.append((index, line, char_start, char_end))
        offset += len(line_with_end)

    return results


def split_long_line(
    line: str,
) -> list[tuple[str, int]]:
    """
    Split a very long line into sub-fragments no longer than
    MAX_LINE_CHARS, breaking on whitespace where possible.

    Returns a list of (text, offset_within_line) tuples so the caller
    can compute accurate absolute character offsets (rather than relying
    on str.find, which returns the first occurrence and breaks when a
    fragment repeats).
    """
    if len(line) <= MAX_LINE_CHARS:
        return [(line, 0)]

    # Absolute spans of each whitespace-separated word in the line.
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for word in line.split(" "):
        start = line.find(word, cursor)
        spans.append((word, start, start + len(word)))
        cursor = start + len(word)

    fragments: list[tuple[str, int]] = []

    def flush(words):
        if not words:
            return
        text = " ".join(w for w, _, _ in words)
        start = words[0][1]
        fragments.append((text, start))

    current: list[tuple[str, int, int]] = []
    current_len = 0

    for word, start, end in spans:
        # A single word exceeding the limit is hard-cut on its own.
        if end - start > MAX_LINE_CHARS:
            flush(current)
            current = []
            current_len = 0

            pos = start
            remaining = word
            while len(remaining) > MAX_LINE_CHARS:
                frag = remaining[:MAX_LINE_CHARS]
                fragments.append((frag, pos))
                pos += len(frag)
                remaining = remaining[MAX_LINE_CHARS:]
            if remaining:
                current = [(remaining, pos, pos + len(remaining))]
                current_len = len(remaining)
            continue

        add = len(word) + (1 if current else 0)
        if current and current_len + add > MAX_LINE_CHARS:
            flush(current)
            current = [(word, start, end)]
            current_len = len(word)
        else:
            current.append((word, start, end))
            current_len += add

    flush(current)
    return fragments


HEADING_RE = re.compile(
    r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$"
)

CLOSING_HASH_RE = re.compile(r"[ \t]+#+[ \t]*$")

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def parse_heading(line: str):
    """
    Parse an ATX heading line.

    Returns (level: int, title: str) if the line is a heading,
    or None when it is not (or is an empty heading that should be
    skipped).
    """
    match = HEADING_RE.match(line)
    if not match:
        return None

    level = len(match.group(1))
    title = match.group(2) or ""

    # Strip trailing closing-sequence hashes (e.g. "Title ##").
    title = CLOSING_HASH_RE.sub("", title).rstrip()

    # A heading with no content is not useful as a section.
    if not title:
        return None

    return level, title


def parse_fence(line: str):
    """
    If `line` opens or closes a fenced code block, return the fence
    marker string (e.g. "```" or "~~~"); otherwise return None.

    A fence is opened by 3+ backticks or tildes at the start of the
    line; the same sequence closes an open fence.
    """
    match = FENCE_RE.match(line)
    if not match:
        return None
    return match.group(1)[0] * 3
