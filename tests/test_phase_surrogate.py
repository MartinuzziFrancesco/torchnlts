"""Tests for torchnlts.surrogate.phase.

Ground truth:
  1. Paper: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) -- phase randomization preserves the Fourier
     magnitude. For an odd-length signal every non-DC bin round-trips exactly; for an even-length signal the Nyquist
     bin, which Hermitian symmetry requires to be real, does not.
  2. Julia: TimeseriesSurrogates.jl's ``RandomFourier``, run via subprocess on the same even-length input, shows the
     same pattern (interior bins exact, Nyquist bin not). The two RNG streams differ, so surrogate values are not
     compared, only each surrogate's own magnitude-preservation pattern.
"""

import math
from unittest import mock

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.surrogate.phase import phase_randomized_surrogate


def test_phase_randomized_surrogate_preserves_all_nondc_bins_for_odd_length():
    torch.manual_seed(0)
    x = torch.randn(15, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x)
    original_magnitude = torch.fft.rfft(x).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate).abs()
    assert torch.allclose(original_magnitude[1:], surrogate_magnitude[1:], atol=1e-10, rtol=1e-10)


def test_phase_randomized_surrogate_nyquist_bin_broken_for_even_length():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x)
    original_magnitude = torch.fft.rfft(x).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate).abs()
    # Interior bins are preserved exactly; the Nyquist bin is not (documented quirk, the gap is large, not noise).
    assert torch.allclose(original_magnitude[1:-1], surrogate_magnitude[1:-1], atol=1e-10, rtol=1e-10)
    assert not torch.isclose(original_magnitude[-1], surrogate_magnitude[-1], atol=0.05)


def test_phase_randomized_surrogate_nyquist_magnitude_is_scaled_by_abs_cosine_of_its_phase():
    # irfft drops the imaginary part of the Nyquist bin, so its magnitude becomes |X| |cos(phase)|; the mean is kept.
    x = torch.randn(16, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    surrogate = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(1))
    phase = 2 * math.pi * torch.rand(1, 9, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    nyquist = torch.fft.rfft(x).abs()[-1]
    assert torch.isclose(torch.fft.rfft(surrogate).abs()[-1], nyquist * torch.cos(phase[0, -1, 0]).abs(), rtol=1e-9)
    assert torch.isclose(surrogate.mean(), x.mean(), atol=1e-12, rtol=0)


def test_phase_randomized_surrogate_matches_timeseries_surrogates_jl_pattern():
    # Same literal input to both languages' generators; each output is checked against its own spectrum (see module
    # docstring).
    trajectory = [0.3, -1.2, 0.7, 2.1, -0.4, 1.0, -0.9, 0.5]
    julia_code = f"""
    using TimeseriesSurrogates, Random, FFTW
    x = {trajectory}
    Random.seed!(1)
    s = surrogenerator(x, RandomFourier())()
    println(join(abs.(rfft(x)), " "))
    println(join(abs.(rfft(s)), " "))
    """
    stdout = run_julia(julia_code)
    julia_original_line, julia_surrogate_line = stdout.strip().splitlines()
    julia_original_magnitude = torch.tensor([float(v) for v in julia_original_line.split()], dtype=torch.float64)
    julia_surrogate_magnitude = torch.tensor([float(v) for v in julia_surrogate_line.split()], dtype=torch.float64)
    assert julia_original_magnitude.shape == (5,)
    assert torch.allclose(julia_original_magnitude[1:-1], julia_surrogate_magnitude[1:-1], atol=1e-10, rtol=1e-10)
    assert not torch.isclose(julia_original_magnitude[-1], julia_surrogate_magnitude[-1], atol=0.05)

    x = torch.tensor(trajectory, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(1))
    original_magnitude = torch.fft.rfft(x).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate).abs()
    assert torch.allclose(original_magnitude, julia_original_magnitude, atol=1e-10, rtol=0)
    assert torch.allclose(original_magnitude[1:-1], surrogate_magnitude[1:-1], atol=1e-10, rtol=1e-10)
    assert not torch.isclose(original_magnitude[-1], surrogate_magnitude[-1], atol=0.05)


def test_phase_randomized_surrogate_mean_approximately_preserved():
    torch.manual_seed(2)
    x = torch.randn(200, dtype=torch.float64) + 5.0
    surrogate = phase_randomized_surrogate(x)
    # rtol=0: the default rtol=1e-5 would loosen this to ~5e-5 around a mean of 5.
    assert torch.isclose(surrogate.mean(), x.mean(), atol=1e-8, rtol=0)


def test_phase_randomized_surrogate_zero_phase_matches_hand_derived_formula():
    # Mock torch.rand to 0 so every phase is exactly 0; the expected surrogate is then irfft(magnitude) + mean.
    x = torch.tensor([1.0, 2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    mean = x.mean()
    magnitude = torch.fft.rfft(x - mean).abs()
    expected = torch.fft.irfft(magnitude.to(torch.complex128), n=x.shape[0]) + mean

    with mock.patch("torch.rand", side_effect=lambda shape, **kwargs: torch.zeros(shape, dtype=torch.float64)):
        result = phase_randomized_surrogate(x)
    assert torch.allclose(result, expected, atol=1e-12, rtol=0)


def test_phase_randomized_surrogate_differs_from_original():
    torch.manual_seed(3)
    x = torch.randn(50, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x)
    assert not torch.allclose(surrogate, x)


def test_phase_randomized_surrogate_reproducible_with_generator():
    x = torch.randn(30, dtype=torch.float64)
    first = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(42))
    second = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(42))
    third = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(43))
    assert torch.equal(first, second)
    assert not torch.allclose(first, third)


def test_phase_randomized_surrogate_batched_shape_and_per_row_invariant():
    torch.manual_seed(4)
    trajectory = torch.randn(3, 20, 2, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(trajectory)
    assert surrogate.shape == trajectory.shape
    original_magnitude = torch.fft.rfft(trajectory, dim=-2).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate, dim=-2).abs()
    # 20 is even, so exclude the last (Nyquist) bin along the time axis.
    assert torch.allclose(original_magnitude[:, 1:-1, :], surrogate_magnitude[:, 1:-1, :], atol=1e-10, rtol=1e-10)


def test_phase_randomized_surrogate_dtype_preserved_float32():
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float32)
    surrogate = phase_randomized_surrogate(trajectory)
    assert surrogate.dtype == torch.float32


def test_phase_randomized_surrogate_rejects_bad_ndim():
    with pytest.raises(ValueError):
        phase_randomized_surrogate(torch.zeros(2, 2, 2, 2))
