"""Surrogate data generation and tests."""

from torchnlts.surrogate.aaft import aaft_surrogate
from torchnlts.surrogate.hypothesis import SurrogateTestResult, surrogate_test
from torchnlts.surrogate.iaaft import iaaft_surrogate
from torchnlts.surrogate.phase import phase_randomized_surrogate
from torchnlts.surrogate.shuffle import random_shuffle_surrogate, random_shift_surrogate

__all__ = [
    "aaft_surrogate",
    "SurrogateTestResult",
    "surrogate_test",
    "iaaft_surrogate",
    "phase_randomized_surrogate",
    "random_shuffle_surrogate",
    "random_shift_surrogate",
]
