"""Nonlinear time series analysis in PyTorch."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from torchnlts import (
    causality,
    chaos,
    convert,
    dimension,
    embedding,
    entropy,
    fluctuations,
    lyapunov,
    neighbors,
    recurrence,
    sciml,
    surrogate,
)

try:
    __version__ = version("torchnlts")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "causality",
    "chaos",
    "convert",
    "dimension",
    "embedding",
    "entropy",
    "fluctuations",
    "lyapunov",
    "neighbors",
    "recurrence",
    "sciml",
    "surrogate",
    "systems",
]


def __getattr__(name):
    # systems imports dysts (~0.7 s), so it is loaded on first access.
    if name == "systems":
        return import_module("torchnlts.systems")
    raise AttributeError(f"module 'torchnlts' has no attribute {name!r}")
