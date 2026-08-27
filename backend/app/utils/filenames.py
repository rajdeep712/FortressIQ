import re
import unicodedata
from pathlib import Path

# Filesystem/S3-safe maximum. Most object stores and filesystems
# reject names longer than 255 bytes for the final path segment.
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """
    Convert a user-supplied filename into a safe, filesystem- and
    S3-friendly name.

    Hardening steps:
      1. Strip any path components (prevents ../ traversal and absolute paths)
      2. Normalize Unicode to ASCII (NFKD) so we stay in a safe charset
      3. Remove control chars, NULL bytes, path separators and the
         Windows-forbidden characters * : ? " < > |
      4. Split stem / extension and sanitize each independently so the
         extension is always preserved
      5. Collapse every run of unsafe characters into a single "_"
      6. Trim leading / trailing dots and underscores
      7. Fall back to "uploaded_file" when the result is empty
      8. Truncate to MAX_FILENAME_LENGTH while keeping the extension

    Example:
        "../../my report!!.pdf"  ->  "my_report.pdf"
    """
    if not filename:
        return "uploaded_file"

    # 1. Drop directory parts (../ , absolute paths, / and \ separators)
    raw = Path(filename).name
    if not raw:
        raw = "uploaded_file"

    # 2. Unicode -> ASCII (é -> e, ç -> c, ...)
    raw = unicodedata.normalize("NFKD", raw)
    raw = raw.encode("ascii", "ignore").decode("ascii")

    # 3. Strip control chars + separators + Windows-forbidden chars
    raw = re.sub(r"[\x00-\x1f\x7f\\/:*?\"<>|]", "", raw)

    # Defensive: drop any residual path-like prefix after stripping above
    raw = Path(raw).name
    if not raw:
        return "uploaded_file"

    # 4. Separate extension, sanitize stem and extension independently
    stem, dot, ext = raw.rpartition(".")

    if dot:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_")

        ext = re.sub(r"[^A-Za-z0-9_-]+", "_", ext)
        ext = re.sub(r"_+", "_", ext).strip("_")

        if not stem:
            stem = "uploaded_file"

        name = f"{stem}.{ext}" if ext else stem
    else:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "uploaded_file"

    # 8. Enforce maximum length, preserving the extension
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot:
            budget = MAX_FILENAME_LENGTH - len(dot) - len(ext)
            name = (stem[:budget] + dot + ext) if budget > 0 else ext[:MAX_FILENAME_LENGTH]
        else:
            name = name[:MAX_FILENAME_LENGTH]

    return name


if __name__ == '__main__':
    print(sanitize_filename('../weird/../file.txt'))