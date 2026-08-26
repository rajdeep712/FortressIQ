import hashlib
from pathlib import Path


CHUNK_SIZE = 1024 * 1024  # 1 MB


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 hash without loading
    the entire file into memory.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            sha256.update(chunk)

    return sha256.hexdigest()