"""Tests for torchnlts.surrogate.iaaft.

Ground truth:
  1. Paper: Schreiber & Schmitz (1996) -- IAAFT preserves the amplitude distribution exactly (sorted values are equal,
     for any iteration count and seed) and the power spectrum approximately, with a relative spectral error that
     generally decreases over the first iterations. A mocked initial permutation is compared with an independent torch
     implementation of the iteration.
  2. Julia: TimeseriesSurrogates.jl's ``IAAFT``, run via subprocess on the same input, shows the same seed-dependent
     spread of converged spectral error as this implementation, not a fixed, always-tiny error.
"""

from unittest import mock

import numpy as np
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.surrogate.iaaft import iaaft_surrogate

_TEST_TRAJECTORY = [0.01, 0.02, 0.03, 0.04, 20.0, 0.05, 0.06, 0.07, 0.15, 0.9]


def _relative_spectral_error(original: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    original_magnitude = torch.fft.rfft(original).abs()
    surrogate_magnitude = torch.fft.rfft(surrogate).abs()
    return (original_magnitude - surrogate_magnitude).norm() / original_magnitude.norm()


def test_iaaft_surrogate_preserves_amplitude_distribution_exactly():
    torch.manual_seed(0)
    x = torch.randn(30, dtype=torch.float64) ** 2
    surrogate = iaaft_surrogate(x, max_iterations=20)
    assert torch.equal(torch.sort(x).values, torch.sort(surrogate).values)


def test_iaaft_surrogate_amplitude_preserved_regardless_of_iteration_count():
    torch.manual_seed(1)
    x = torch.randn(20, dtype=torch.float64) ** 2
    for max_iterations in (1, 5, 50):
        surrogate = iaaft_surrogate(x, max_iterations=max_iterations, generator=torch.Generator().manual_seed(7))
        assert torch.equal(torch.sort(x).values, torch.sort(surrogate).values)


def test_iaaft_surrogate_matches_independent_reimplementation():
    # Positive oracle: mocked initial permutation; 1 and 3 alternating-projection rounds reimplemented in torch.
    x = torch.tensor([5.0, 1.0, 20.0, 0.5, 3.0], dtype=torch.float64)  # odd length: no Nyquist bin
    fixed_keys = torch.tensor([0.3, 0.9, 0.1, 0.7, 0.5], dtype=torch.float64)

    def fake_rand(shape, generator=None, dtype=None, device=None):
        return fixed_keys.reshape(shape)

    with mock.patch("torch.rand", side_effect=fake_rand):
        result_1 = iaaft_surrogate(x, max_iterations=1)
        result_3 = iaaft_surrogate(x, max_iterations=3)

    initial_permutation = torch.argsort(fixed_keys)
    target_magnitude = torch.fft.rfft(x).abs()
    sorted_x = torch.sort(x, stable=True).values

    def one_round(s):
        phase = torch.angle(torch.fft.rfft(s))
        adjusted = target_magnitude * torch.exp(1j * phase)
        s = torch.fft.irfft(adjusted, n=x.shape[0])
        rank = torch.argsort(torch.argsort(s, stable=True), stable=True)
        return sorted_x[rank]

    surrogate = x[initial_permutation]
    expected_1 = one_round(surrogate)
    s = surrogate
    for _ in range(3):
        s = one_round(s)
    expected_3 = s

    assert torch.allclose(result_1, expected_1, atol=1e-12, rtol=0)
    assert torch.allclose(result_3, expected_3, atol=1e-12, rtol=0)


def test_iaaft_surrogate_relative_spectral_error_decreases_with_more_iterations():
    # Longer, slower-converging series: the 10-point _TEST_TRAJECTORY converges in one iteration for most seeds.
    torch.manual_seed(0)
    x = torch.randn(200, dtype=torch.float64) ** 2
    error_after_1 = _relative_spectral_error(
        x, iaaft_surrogate(x, max_iterations=1, generator=torch.Generator().manual_seed(0))
    )
    error_after_50 = _relative_spectral_error(
        x, iaaft_surrogate(x, max_iterations=50, generator=torch.Generator().manual_seed(0))
    )
    assert error_after_50 < error_after_1


def test_iaaft_surrogate_matches_timeseries_surrogates_jl_seed_spread():
    # TimeseriesSurrogates.jl (see module docstring) shows the same seed-dependent convergence spread on this input;
    # the spread is what is checked.
    julia_code = f"""
    using TimeseriesSurrogates, Random, FFTW
    x = {_TEST_TRAJECTORY}
    orig_mag = abs.(rfft(x))
    for seed in 0:9
        Random.seed!(seed)
        s = surrogenerator(x, IAAFT())()
        surr_mag = abs.(rfft(s))
        rel_err = sqrt(sum((orig_mag .- surr_mag).^2)) / sqrt(sum(orig_mag.^2))
        println(rel_err)
    end
    """
    stdout = run_julia(julia_code)
    julia_errors = torch.tensor([float(v) for v in stdout.strip().splitlines()], dtype=torch.float64)
    assert julia_errors.shape == (10,)
    # Every seed converges (generous bound, observed worst ~3e-3) and the errors differ across seeds.
    assert julia_errors.max() < 1e-2
    assert julia_errors.max() > 10 * julia_errors.min()

    x = torch.tensor(_TEST_TRAJECTORY, dtype=torch.float64)
    python_errors = torch.stack(
        [
            _relative_spectral_error(x, iaaft_surrogate(x, generator=torch.Generator().manual_seed(seed)))
            for seed in range(10)
        ]
    )
    assert python_errors.max() < 1e-2
    assert python_errors.max() > 10 * python_errors.min()


def test_iaaft_surrogate_differs_from_original():
    torch.manual_seed(2)
    x = torch.randn(50, dtype=torch.float64)
    surrogate = iaaft_surrogate(x)
    assert not torch.allclose(surrogate, x)


def test_iaaft_surrogate_reproducible_with_generator():
    x = torch.randn(30, dtype=torch.float64)
    first = iaaft_surrogate(x, generator=torch.Generator().manual_seed(42))
    second = iaaft_surrogate(x, generator=torch.Generator().manual_seed(42))
    third = iaaft_surrogate(x, generator=torch.Generator().manual_seed(43))
    assert torch.equal(first, second)
    assert not torch.allclose(first, third)


def test_iaaft_surrogate_batched_shape_and_per_row_amplitude_preservation():
    torch.manual_seed(3)
    trajectory = torch.randn(3, 25, 2, dtype=torch.float64) ** 2
    surrogate = iaaft_surrogate(trajectory, max_iterations=20)
    assert surrogate.shape == trajectory.shape
    assert torch.equal(
        torch.sort(trajectory, dim=-2).values,
        torch.sort(surrogate, dim=-2).values,
    )


def test_iaaft_surrogate_unbatched_multivariate_shape():
    torch.manual_seed(4)
    trajectory = torch.randn(25, 2, dtype=torch.float64) ** 2
    surrogate = iaaft_surrogate(trajectory, max_iterations=20)
    assert surrogate.shape == trajectory.shape
    assert torch.equal(
        torch.sort(trajectory, dim=-2).values,
        torch.sort(surrogate, dim=-2).values,
    )


def test_iaaft_surrogate_handles_ties_and_constant_input():
    x = torch.tensor([1.0, 2.0, 2.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    surrogate = iaaft_surrogate(x, max_iterations=20)
    assert torch.equal(torch.sort(x).values, torch.sort(surrogate).values)

    constant = torch.full((10,), 3.0, dtype=torch.float64)
    surrogate_constant = iaaft_surrogate(constant, max_iterations=20)
    assert torch.equal(surrogate_constant, constant)


def test_iaaft_surrogate_dtype_preserved_float32():
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float32)
    surrogate = iaaft_surrogate(trajectory, max_iterations=10)
    assert surrogate.dtype == torch.float32


def test_iaaft_surrogate_rejects_bad_ndim():
    with pytest.raises(ValueError):
        iaaft_surrogate(torch.zeros(2, 2, 2, 2))


def test_iaaft_surrogate_rejects_non_positive_max_iterations():
    x = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError):
        iaaft_surrogate(x, max_iterations=0)


@pytest.mark.parametrize("max_iterations", [2.5, 3.0, "3", None])
def test_iaaft_surrogate_rejects_non_integer_max_iterations(max_iterations):
    x = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError, match="max_iterations"):
        iaaft_surrogate(x, max_iterations=max_iterations)


def test_iaaft_surrogate_accepts_numpy_integer_max_iterations():
    x = torch.randn(20, dtype=torch.float64)
    first = iaaft_surrogate(x, max_iterations=np.int64(5), generator=torch.Generator().manual_seed(1))
    second = iaaft_surrogate(x, max_iterations=5, generator=torch.Generator().manual_seed(1))
    assert torch.equal(first, second)
