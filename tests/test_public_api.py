import ast
import importlib
import inspect
import pathlib
import subprocess
import sys

import pytest

import torchnlts

ROOT = pathlib.Path(torchnlts.__file__).parent
SUBPACKAGES = [
    "causality", "chaos", "dimension", "embedding", "entropy", "fluctuations",
    "lyapunov", "matrixprofile", "neighbors", "recurrence", "sciml", "surrogate",
]  # fmt: skip
SINGLE_FILE_MODULES = ["convert", "scaling"]


def _public_names(path):
    """Public function and class names defined at the top level of ``path``, found by parsing the source."""
    return [
        node.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]


def _defined(subpackage):
    """Map public name -> defining module name."""
    found = {}
    for path in sorted((ROOT / subpackage).glob("[!_]*.py")):
        for name in _public_names(path):
            found[name] = f"torchnlts.{subpackage}.{path.stem}"
    return found


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_all_matches_defined_names(subpackage):
    package = importlib.import_module(f"torchnlts.{subpackage}")
    defined = _defined(subpackage)
    assert sorted(package.__all__) == sorted(defined)
    assert len(set(package.__all__)) == len(package.__all__)
    for name, module in defined.items():
        assert getattr(package, name) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module_name", SINGLE_FILE_MODULES)
def test_single_file_module_all_matches_defined_names(module_name):
    module = importlib.import_module(f"torchnlts.{module_name}")
    defined = _public_names(pathlib.Path(module.__file__))
    assert sorted(module.__all__) == sorted(defined)
    assert len(set(module.__all__)) == len(module.__all__)


def _public_callables():
    for module_name in [*SUBPACKAGES, *SINGLE_FILE_MODULES]:
        module = importlib.import_module(f"torchnlts.{module_name}")
        for name in module.__all__:
            yield pytest.param(getattr(module, name), id=f"{module_name}.{name}")


@pytest.mark.parametrize("public", list(_public_callables()))
def test_optional_parameters_are_keyword_only(public):
    allowed = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
    positional_optional = [
        parameter.name
        for parameter in inspect.signature(public).parameters.values()
        if parameter.default is not inspect.Parameter.empty and parameter.kind not in allowed
    ]
    assert not positional_optional


def test_top_level():
    assert isinstance(torchnlts.__version__, str)
    modules = [*SUBPACKAGES, *SINGLE_FILE_MODULES]
    assert sorted(torchnlts.__all__) == sorted(modules)
    for name in modules:
        assert getattr(torchnlts, name).__name__ == f"torchnlts.{name}"
    with pytest.raises(AttributeError):
        torchnlts.nonexistent


def test_top_level_attributes_exist_without_other_imports():
    code = "import sys, torchnlts; sys.exit(not all(hasattr(torchnlts, name) for name in torchnlts.__all__))"
    subprocess.run([sys.executable, "-c", code], check=True)
