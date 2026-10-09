import ast
import importlib
import pathlib

import pytest

import torchnlts

ROOT = pathlib.Path(torchnlts.__file__).parent
SUBPACKAGES = [
    "causality", "chaos", "dimension", "embedding", "entropy", "fluctuations",
    "lyapunov", "neighbors", "recurrence", "sciml", "surrogate",
]  # fmt: skip


def _defined(subpackage):
    """Map public name -> defining module name, found by parsing the source."""
    found = {}
    for path in sorted((ROOT / subpackage).glob("[!_]*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                found[node.name] = f"torchnlts.{subpackage}.{path.stem}"
    return found


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_all_matches_defined_names(subpackage):
    package = importlib.import_module(f"torchnlts.{subpackage}")
    defined = _defined(subpackage)
    assert sorted(package.__all__) == sorted(defined)
    assert len(set(package.__all__)) == len(package.__all__)
    for name, module in defined.items():
        assert getattr(package, name) is getattr(importlib.import_module(module), name)


def test_top_level():
    assert isinstance(torchnlts.__version__, str)
    for name in [*SUBPACKAGES, "convert"]:
        assert getattr(torchnlts, name).__name__ == f"torchnlts.{name}"
    assert torchnlts.systems.__name__ == "torchnlts.systems"
    with pytest.raises(AttributeError):
        torchnlts.nonexistent
