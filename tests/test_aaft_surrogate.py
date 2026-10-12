"""Tests for torchnlts.surrogate.aaft.

Ground truth:
  1. Paper: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) -- AAFT preserves the amplitude distribution exactly
     (sorted values are equal) in three steps: rank Gaussian noise to match the data, phase-randomize that series,
     rank-restore the data to the result. With mocked randomness, the Gaussian intermediate gives a different answer
     than phase-randomizing the data directly, so the first step matters for non-Gaussian data.
  2. Julia: TimeseriesSurrogates.jl's ``AAFT`` skips the Gaussian rank-matching step. Run via subprocess with ``AAFT``
     and ``RandomFourier`` seeded identically, rank-restoring the ``RandomFourier`` surrogate to the data reproduces
     the ``AAFT`` output exactly. This implementation follows the paper instead.
"""

from unittest import mock

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.surrogate.aaft import aaft_surrogate
from torchnlts.surrogate.phase import phase_randomized_surrogate


def test_aaft_surrogate_preserves_amplitude_distribution_exactly():
    torch.manual_seed(0)
    x = torch.randn(30, dtype=torch.float64) ** 2
    surrogate = aaft_surrogate(x)
    assert torch.equal(torch.sort(x).values, torch.sort(surrogate).values)


def test_aaft_surrogate_uses_gaussianized_series_not_raw_data():
    # Mocked randomness; for skewed x the Gaussian-rank intermediate gives a different rank order than the Julia shortcut.
    x = torch.tensor([0.01, 0.02, 0.03, 0.04, 20.0, 0.05, 0.06, 0.07], dtype=torch.float64)
    fixed_noise = torch.tensor([-1.5, 0.3, -0.2, 1.1, 0.0, -0.8, 0.9, 2.0], dtype=torch.float64)

    def fake_randn(shape, generator=None, dtype=None, device=None):
        return fixed_noise.reshape(shape)

    def zero_phase(shape, **kwargs):
        return torch.zeros(shape, dtype=torch.float64)

    with mock.patch("torch.randn", side_effect=fake_randn), mock.patch("torch.rand", side_effect=zero_phase):
        result = aaft_surrogate(x)

    with mock.patch("torch.rand", side_effect=zero_phase):
        ft_direct = phase_randomized_surrogate(x)
    sorted_x = torch.sort(x).values
    rank_direct = torch.argsort(torch.argsort(ft_direct, stable=True), stable=True)
    julia_style_shortcut = torch.gather(sorted_x, 0, rank_direct)

    assert not torch.equal(result, julia_style_shortcut)
    assert torch.equal(torch.sort(result).values, sorted_x)


def test_aaft_surrogate_matches_independent_reimplementation_with_zero_phase():
    # Positive oracle: independent plain-torch reimplementation of the three-step recipe, compared exactly.
    x = torch.tensor([5.0, 1.0, 20.0, 0.5, 3.0], dtype=torch.float64)  # odd length: no Nyquist bin
    fixed_noise = torch.tensor([0.2, -1.0, 0.5, 2.0, -0.3], dtype=torch.float64)

    def fake_randn(shape, generator=None, dtype=None, device=None):
        return fixed_noise.reshape(shape)

    def zero_phase(shape, **kwargs):
        return torch.zeros(shape, dtype=torch.float64)

    with mock.patch("torch.randn", side_effect=fake_randn), mock.patch("torch.rand", side_effect=zero_phase):
        result = aaft_surrogate(x)

    rank_x = torch.argsort(torch.argsort(x, stable=True), stable=True)
    sorted_noise = torch.sort(fixed_noise, stable=True).values
    gaussianized = sorted_noise[rank_x]
    mean = gaussianized.mean()
    magnitude = torch.fft.rfft(gaussianized - mean).abs()
    ft_surrogate = torch.fft.irfft(magnitude.to(torch.complex128), n=x.shape[0]) + mean
    rank_ft = torch.argsort(torch.argsort(ft_surrogate, stable=True), stable=True)
    sorted_x = torch.sort(x, stable=True).values
    expected = sorted_x[rank_ft]

    assert torch.allclose(result, expected, atol=1e-12, rtol=0)


def test_aaft_surrogate_handles_ties_and_constant_input():
    # Ties (including the all-tied constant case) must not crash the double-argsort rank trick or the final gather.
    x = torch.tensor([1.0, 2.0, 2.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    surrogate = aaft_surrogate(x)
    assert torch.equal(torch.sort(x).values, torch.sort(surrogate).values)

    constant = torch.full((10,), 3.0, dtype=torch.float64)
    surrogate_constant = aaft_surrogate(constant)
    assert torch.equal(surrogate_constant, constant)


def test_aaft_surrogate_matches_timeseries_surrogates_jl_shortcut():
    # Julia: AAFT(x) == rank-restore(x, RandomFourier(x)), i.e. no Gaussian-rank step (see module docstring).
    trajectory = [0.01, 0.02, 0.03, 0.04, 20.0, 0.05, 0.06, 0.07]
    julia_code = f"""
    using TimeseriesSurrogates, Random, FFTW
    x = {trajectory}
    Random.seed!(1)
    s_aaft = surrogenerator(x, AAFT())()
    Random.seed!(1)
    s_ft = surrogenerator(x, RandomFourier())()
    sorted_x = sort(x)
    ix = sortperm(s_ft)
    s_ft_rank_restored = similar(x)
    s_ft_rank_restored[ix] .= sorted_x
    println(s_aaft == s_ft_rank_restored)
    """
    stdout = run_julia(julia_code)
    assert stdout.strip() == "true"


def test_aaft_surrogate_differs_from_original():
    torch.manual_seed(1)
    x = torch.randn(50, dtype=torch.float64)
    surrogate = aaft_surrogate(x)
    assert not torch.allclose(surrogate, x)


def test_aaft_surrogate_reproducible_with_generator():
    x = torch.randn(30, dtype=torch.float64)
    first = aaft_surrogate(x, generator=torch.Generator().manual_seed(42))
    second = aaft_surrogate(x, generator=torch.Generator().manual_seed(42))
    third = aaft_surrogate(x, generator=torch.Generator().manual_seed(43))
    assert torch.equal(first, second)
    assert not torch.allclose(first, third)


def test_aaft_surrogate_batched_shape_and_per_row_amplitude_preservation():
    torch.manual_seed(2)
    trajectory = torch.randn(3, 25, 2, dtype=torch.float64)
    surrogate = aaft_surrogate(trajectory)
    assert surrogate.shape == trajectory.shape
    assert torch.equal(
        torch.sort(trajectory, dim=-2).values,
        torch.sort(surrogate, dim=-2).values,
    )


def test_aaft_surrogate_unbatched_multivariate_shape():
    torch.manual_seed(4)
    trajectory = torch.randn(25, 2, dtype=torch.float64)  # [T, D], no batch dim
    surrogate = aaft_surrogate(trajectory)
    assert surrogate.shape == trajectory.shape
    assert torch.equal(
        torch.sort(trajectory, dim=-2).values,
        torch.sort(surrogate, dim=-2).values,
    )


def test_aaft_surrogate_dtype_preserved_float32():
    torch.manual_seed(3)
    trajectory = torch.randn(20, dtype=torch.float32)
    surrogate = aaft_surrogate(trajectory)
    assert surrogate.dtype == torch.float32


def test_aaft_surrogate_rejects_bad_ndim():
    with pytest.raises(ValueError):
        aaft_surrogate(torch.zeros(2, 2, 2, 2))
