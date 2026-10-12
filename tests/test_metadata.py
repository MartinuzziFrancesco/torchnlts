import pathlib
import re
from importlib.metadata import version

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_PATTERNS = {
    "pyproject.toml": r"""^version\s*=\s*["']([^"']+)["']""",
    "CITATION.cff": r"""^version:\s*["']?([^"'\s]+)["']?\s*$""",
    "CITATION.bib": r"^\s*version\s*=\s*\{([^}]+)\}",
}


@pytest.mark.parametrize("filename", VERSION_PATTERNS)
def test_version_matches_installed_package(filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    match = re.search(VERSION_PATTERNS[filename], text, flags=re.MULTILINE)
    assert match is not None, f"no version line in {filename}"
    assert match.group(1) == version("torchnlts")
