import sys
from pathlib import Path

# Ensure the backend/ directory is importable so `import app...` works
# regardless of the current working directory or pytest's import mode
# (the default "prepend" mode auto-adds this, but "importlib" mode
# does not, which otherwise breaks `import app` in IDE runners).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest

from app.utils.filenames import sanitize_filename


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../my report!!.pdf", "my_report.pdf"),
        ("../weird/../file.txt", "file.txt"),
        ("my report!!.pdf", "my_report.pdf"),
        ("!!!.pdf", "uploaded_file.pdf"),
        (".gitignore", "uploaded_file.gitignore"),
        ("résumé café.docx", "resume_cafe.docx"),
        ("..\\..\\x.pdf", "x.pdf"),
        ("my..report..pdf", "my_report.pdf"),
        ("  leading spaces .pdf", "leading_spaces.pdf"),
        ("a" * 300 + ".pdf", "a" * 251 + ".pdf"),  # 255 - len(".pdf")
        ("", "uploaded_file"),
        (".", "uploaded_file"),
        ("con.pdf", "con.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_preserves_extension_on_long_stem():
    name = sanitize_filename("x" * 300 + ".verylongext")
    assert name.endswith(".verylongext")
    assert len(name) <= 255
