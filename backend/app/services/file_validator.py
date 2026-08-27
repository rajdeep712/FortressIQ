import os
import sys
import sysconfig
from pathlib import Path

# python-magic-bin bundles libmagic.dll inside the magic/ package
# directory. python-magic's loader searches PATH and CWD but not
# the package dir, so we inject it into PATH before importing magic.
if sys.platform == "win32":
    _magic_bin_dir = (
        Path(sysconfig.get_path("purelib")) / "magic" / "libmagic"
    )
    if _magic_bin_dir.exists():
        os.environ["PATH"] = (
            str(_magic_bin_dir) + os.pathsep + os.environ.get("PATH", "")
        )

import magic
import zipfile

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".xlsx",
    ".csv",
    ".json",
    ".pptx",
    ".html",
}

FILE_SIGNATURES = {
    ".pdf": [
        b"%PDF-",
    ],

    # DOCX / PPTX / XLSX are ZIP containers.
    ".docx": [
        b"PK\x03\x04",
    ],

    ".pptx": [
        b"PK\x03\x04",
    ],

    ".xlsx": [
        b"PK\x03\x04",
    ],
}


EXPECTED_MIME_TYPES = {
    ".pdf": {
        "application/pdf",
    },

    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    },

    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/zip",
    },

    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
    },

    ".txt": {
        "text/plain",
    },

    ".md": {
        "text/plain",
        "text/markdown",
    },

    ".csv": {
        "text/csv",
        "text/plain",
    },

    ".json": {
        "application/json",
        "text/plain",
    },

    ".html": {
        "text/html",
        "text/plain",
    },
}

OFFICE_CONTENT_TYPES = {
    ".docx": "wordprocessingml.document",
    ".pptx": "presentationml.presentation",
    ".xlsx": "spreadsheetml.sheet",
}

MAX_FILE_SIZE_MB = 50


def get_extension(filename: str) -> str:
    """
    Extract the lowercase extension from a filename.
    """
    return Path(filename).suffix.lower()


def is_supported_extension(filename: str) -> bool:
    """
    Check whether the file extension is supported.
    """
    extension = get_extension(filename)

    return extension in SUPPORTED_EXTENSIONS


def check_file_signature(
    file_path: Path,
    extension: str,
) -> bool:
    """
    Perform a basic magic-byte/signature check.
    """

    signatures = FILE_SIGNATURES.get(extension)

    # Text-based formats don't have one universal
    # magic signature, so they are handled separately.
    if not signatures:
        return True

    with file_path.open("rb") as file:
        header = file.read(16)

    return any(
        header.startswith(signature)
        for signature in signatures
    )


def detect_mime_type(file_path: Path) -> str:
    """
    Detect MIME type from actual file contents.
    """
    mime = magic.Magic(mime=True)
    return mime.from_file(str(file_path))


def check_mime_type(
    detected_mime: str,
    extension: str,
) -> bool:
    """
    Check if the detected MIME type matches
    what we expect for this file type.
    """
    allowed_mimes = EXPECTED_MIME_TYPES.get(extension, set())

    return detected_mime in allowed_mimes


def validate_office_package(
    file_path: Path,
    extension: str,
) -> bool:
    """
    Validate DOCX/PPTX/XLSX as Office Open XML packages.
    """

    if extension not in OFFICE_CONTENT_TYPES:
        return True

    try:
        with zipfile.ZipFile(file_path, "r") as archive:

            if archive.testzip() is not None:
                return False

            names = archive.namelist()

            # Basic OOXML package validation
            if "[Content_Types].xml" not in names:
                return False

            expected_fragment = OFFICE_CONTENT_TYPES[extension]

            content_types = archive.read(
                "[Content_Types].xml"
            ).decode(
                "utf-8",
                errors="ignore",
            )

            if expected_fragment not in content_types:
                return False

            return True

    except (zipfile.BadZipFile, OSError):
        return False


MAX_ZIP_FILES = 5000
MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500 MB


def check_zip_limits(file_path: Path) -> bool:
    """
    Check zip file for:
    • file count <= 5000
    • total uncompressed size <= 500MB
    """

    try:
        with zipfile.ZipFile(file_path, "r") as archive:

            infos = archive.infolist()

            if len(infos) > MAX_ZIP_FILES:
                return False

            total_uncompressed = sum(
                info.file_size
                for info in infos
            )

            if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
                return False

            return True

    except zipfile.BadZipFile:
        return False