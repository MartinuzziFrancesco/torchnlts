"""Fractal and attractor dimension estimators."""

from torchnlts.dimension.correlation import (
    correlation_sum,
    soft_correlation_sum,
    correlation_dimension,
)
from torchnlts.dimension.generalized import (
    box_occupation_probabilities,
    box_counting_entropy,
    generalized_dimension,
)
from torchnlts.dimension.higuchi import higuchi_length, higuchi_dimension
from torchnlts.dimension.katz import katz_dimension
from torchnlts.dimension.petrosian import petrosian_dimension
from torchnlts.dimension.variogram import power_variogram, variogram_dimension

__all__ = [
    "correlation_sum",
    "soft_correlation_sum",
    "correlation_dimension",
    "box_occupation_probabilities",
    "box_counting_entropy",
    "generalized_dimension",
    "higuchi_length",
    "higuchi_dimension",
    "katz_dimension",
    "petrosian_dimension",
    "power_variogram",
    "variogram_dimension",
]
