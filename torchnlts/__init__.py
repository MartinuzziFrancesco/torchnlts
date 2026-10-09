"""Nonlinear time series analysis in PyTorch."""

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
]
