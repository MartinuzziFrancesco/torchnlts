"""Tests for torchnlts.surrogate.phase.

Ground truth:
  1. Paper: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) -- phase randomization preserves the Fourier
     magnitude. Hermitian symmetry forces the DC and (even length) Nyquist phases to 0 or pi, so every bin round-trips.
  2. Julia: TimeseriesSurrogates.jl's ``RandomFourier``, run via subprocess on the same even-length input, preserves
     the interior bins; it draws a uniform Nyquist phase and so does not preserve that bin, unlike this package. The
     two RNG streams differ, so surrogate values are not compared, only each surrogate's own magnitude pattern.
"""

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


def test_phase_randomized_surrogate_preserves_nyquist_bin_for_even_length():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x)
    assert torch.allclose(torch.fft.rfft(x).abs()[1:], torch.fft.rfft(surrogate).abs()[1:], atol=1e-10, rtol=1e-10)


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
    x = torch.tensor(trajectory, dtype=torch.float64)
    surrogate = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(1))
    original_magnitude = torch.fft.rfft(x).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate).abs()
    assert torch.allclose(original_magnitude, julia_original_magnitude, atol=1e-10, rtol=0)
    assert torch.allclose(original_magnitude[1:], surrogate_magnitude[1:], atol=1e-10, rtol=1e-10)


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
    assert torch.allclose(original_magnitude[:, 1:, :], surrogate_magnitude[:, 1:, :], atol=1e-10, rtol=1e-10)


def test_phase_randomized_surrogate_dtype_preserved_float32():
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float32)
    surrogate = phase_randomized_surrogate(trajectory)
    assert surrogate.dtype == torch.float32


def test_phase_randomized_surrogate_rejects_bad_ndim():
    with pytest.raises(ValueError):
        phase_randomized_surrogate(torch.zeros(2, 2, 2, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 64, 65, 1000])
def test_phase_randomized_surrogate_preserves_every_fourier_magnitude_and_the_mean(length, dtype):
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    x = torch.randn(length, dtype=dtype, generator=torch.Generator().manual_seed(length)) + 3.0
    for seed in range(5):
        surrogate = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(seed))
        assert surrogate.dtype == dtype
        assert torch.isfinite(surrogate).all()
        original_magnitude = torch.fft.rfft(x - x.mean()).abs()
        surrogate_magnitude = torch.fft.rfft(surrogate - surrogate.mean()).abs()
        assert torch.allclose(
            original_magnitude, surrogate_magnitude, atol=tolerance * (1 + original_magnitude.max()), rtol=0
        )
        assert torch.isclose(surrogate.mean(), x.mean(), atol=tolerance * 10, rtol=0)
        if length % 2 == 0:
            assert torch.isclose(surrogate.var(unbiased=False), x.var(unbiased=False), atol=tolerance * 10, rtol=1e-5)


def test_phase_randomized_surrogate_draws_both_nyquist_signs():
    x = torch.randn(16, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    generator = torch.Generator().manual_seed(0)
    signs = {
        torch.sign(torch.fft.rfft(phase_randomized_surrogate(x, generator=generator) - x.mean())[-1].real).item()
        for _ in range(50)
    }
    assert signs == {-1.0, 1.0}


def test_phase_randomized_surrogate_consumes_one_rfft_shaped_draw():
    x = torch.randn(2, 16, 3, dtype=torch.float64)
    used = torch.Generator().manual_seed(5)
    phase_randomized_surrogate(x, generator=used)
    reference = torch.Generator().manual_seed(5)
    torch.rand(2, 9, 3, generator=reference, dtype=torch.float64)
    assert torch.equal(torch.rand(4, generator=used), torch.rand(4, generator=reference))
    assert not torch.equal(phase_randomized_surrogate(x, generator=used), phase_randomized_surrogate(x, generator=used))


def test_phase_randomized_surrogate_nyquist_signs_are_independent_per_channel():
    x = torch.randn(64, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    surrogate = phase_randomized_surrogate(x, generator=torch.Generator().manual_seed(1))
    nyquist = torch.fft.rfft(surrogate - x.mean(dim=0), dim=0)[-1].real
    assert torch.allclose(nyquist.abs(), torch.fft.rfft(x - x.mean(dim=0), dim=0)[-1].abs(), atol=1e-10)
    assert len(set(torch.sign(nyquist).tolist())) == 2
