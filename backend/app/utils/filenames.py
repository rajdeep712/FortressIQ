import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Convert an uploaded filename into a safe filename.

    Example:

        "../../my report!!.pdf"

    becomes something like:

        "my_report.pdf"
    """

    filename = Path(filename).name

    # Replace anything other than:
    # letters, numbers, dot, underscore and hyphen
    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    # Prevent hidden/empty filenames
    filename = filename.strip(".")

    if not filename:
        filename = "uploaded_file"

    return filename