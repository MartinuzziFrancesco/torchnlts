"""Tests for torchnlts.dimension.dfa.

Dual ground truth:
  1. Paper: Peng et al. (1994) -- checked via a hand-derived example.
  2. Reference implementations: nolds.dfa (overlap=False, fit_exp='poly'
     matches this module's convention exactly) and antropy.detrended_fluctuation
     (order=1, non-overlapping, fixed log-spaced nvals sweep).
"""

import pytest
import torch

from torchnlts.dimension.dfa import dfa_exponent, dfa_fluctuation


def test_dfa_fluctuation_hand_derived():
    # x = [1,2,1,3,2,4,1,5], mean=19/8=2.375.
    # profile y = cumsum(x - mean):
    #   y = [-1.375, -1.75, -3.125, -2.5, -2.875, -1.25, -2.625, 0.0]
    # window_size=4, order=1 -> two windows [y0..y3], [y4..y7].
    # Window 1 ([-1.375,-1.75,-3.125,-2.5] at abscissa 0..3): OLS line has
    # slope=-0.475, intercept=-1.475 -> residuals [0.1, 0.2, -0.7, 0.4],
    # sum of squares = 0.01+0.04+0.49+0.16 = 0.7.
    # Window 2 ([-2.875,-1.25,-2.625,0.0]): slope=0.725, intercept=-2.775
    # -> residuals [-0.1, 0.8, -1.3, 0.6], sum of squares = 0.01+0.64+1.69+0.36 = 2.7.
    # F(4) = sqrt((0.7 + 2.7) / 8) = sqrt(0.425) = 0.6519202405202648.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    fluctuation = dfa_fluctuation(trajectory, 4, order=1)
    assert torch.isclose(fluctuation, torch.tensor(0.6519202405202648, dtype=torch.float64))


def test_dfa_fluctuation_leftover_samples_are_dropped():
    # T=10, window_size=4: only 2 full windows (8 samples) are used, the
    # trailing 2 samples are dropped -- same convention as nolds/antropy.
    trajectory = torch.arange(1.0, 11.0, dtype=torch.float64)
    full = dfa_fluctuation(trajectory, 4)
    truncated = dfa_fluctuation(trajectory[:8], 4)
    assert torch.isclose(full, truncated)


def test_dfa_fluctuation_window_larger_than_trajectory_is_nan():
    trajectory = torch.randn(5, dtype=torch.float64)
    result = dfa_fluctuation(trajectory, 10)
    assert torch.isnan(result)


def test_dfa_fluctuation_empty_trajectory_is_nan_not_crash():
    empty = torch.tensor([], dtype=torch.float64)
    result = dfa_fluctuation(empty, [4, 5])
    assert torch.isnan(result).all()


def test_dfa_fluctuation_constant_series_is_zero():
    trajectory = torch.full((20,), 5.0, dtype=torch.float64)
    result = dfa_fluctuation(trajectory, 4)
    assert result.item() == 0.0


def test_dfa_fluctuation_multivariate_raises():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    try:
        dfa_fluctuation(trajectory, 4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dfa_fluctuation_negative_order_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    try:
        dfa_fluctuation(trajectory, 4, order=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dfa_fluctuation_empty_window_size_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    try:
        dfa_fluctuation(trajectory, [])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dfa_fluctuation_window_too_small_for_order_raises():
    # window_size must be > order + 1, else the fit is exact and F(n) = 0
    # identically (no degrees of freedom left to show fluctuation).
    trajectory = torch.randn(10, dtype=torch.float64)
    try:
        dfa_fluctuation(trajectory, 2, order=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dfa_fluctuation_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 40, 1, dtype=torch.float64)
    window_sizes = [4, 5, 8, 10]
    swept = dfa_fluctuation(trajectory, window_sizes)
    assert swept.shape == (4, 4)
    for index, window_size in enumerate(window_sizes):
        expected = dfa_fluctuation(trajectory, window_size)
        assert expected.shape == (4,)
        assert torch.allclose(swept[index], expected)


def test_dfa_fluctuation_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 40, 1, dtype=torch.float64)
    window_sizes = [4, 5, 8]
    batched = dfa_fluctuation(trajectory, window_sizes)
    unbatched = dfa_fluctuation(trajectory[0], window_sizes)
    assert batched.shape == (3, 1)
    assert torch.allclose(batched[:, 0], unbatched)


def test_dfa_fluctuation_unbatched_scalar_window_size_is_scalar():
    trajectory = torch.randn(20, dtype=torch.float64)
    result = dfa_fluctuation(trajectory, 4)
    assert result.shape == ()


def test_dfa_fluctuation_gradcheck():
    torch.manual_seed(2)
    trajectory = torch.randn(25, dtype=torch.float64, requires_grad=True)

    def fluctuation(x):
        return dfa_fluctuation(x, [4, 5, 6, 8])

    assert torch.autograd.gradcheck(fluctuation, (trajectory,))


def test_dfa_exponent_gradcheck_through_trajectory():
    torch.manual_seed(3)
    trajectory = torch.randn(25, dtype=torch.float64, requires_grad=True)
    window_sizes = torch.tensor([4, 5, 6, 8], dtype=torch.float64)

    def exponent(x):
        return dfa_exponent(window_sizes, dfa_fluctuation(x, [4, 5, 6, 8]))

    assert torch.autograd.gradcheck(exponent, (trajectory,))


def test_dfa_exponent_single_window_size_is_nan_not_crash():
    result = dfa_exponent(torch.tensor([4.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    assert torch.isnan(result)


def test_dfa_exponent_white_noise_is_near_half():
    # Peng et al. (1994): alpha ~ 0.5 for uncorrelated (white) noise.
    torch.manual_seed(4)
    trajectory = torch.randn(8000, dtype=torch.float64)
    window_sizes = [10, 13, 16, 21, 26, 34, 43, 55, 70, 89, 113, 144, 183, 234, 298, 379, 483, 616, 785, 1000]
    fluctuation = dfa_fluctuation(trajectory, window_sizes)
    alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), fluctuation)
    assert abs(alpha.item() - 0.5) < 0.05


def test_dfa_exponent_brownian_motion_is_near_one_point_five():
    # Peng et al. (1994): alpha ~ 1.5 for Brownian motion (cumsum of
    # white noise) -- one order of integration above white noise.
    torch.manual_seed(5)
    white_noise = torch.randn(8000, dtype=torch.float64)
    brownian_motion = torch.cumsum(white_noise, dim=0)
    window_sizes = [10, 13, 16, 21, 26, 34, 43, 55, 70, 89, 113, 144, 183, 234, 298, 379, 483, 616, 785, 1000]
    fluctuation = dfa_fluctuation(brownian_motion, window_sizes)
    alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), fluctuation)
    assert abs(alpha.item() - 1.5) < 0.05


def test_dfa_matches_nolds_non_overlapping():
    import nolds

    torch.manual_seed(6)
    trajectory = torch.randn(2000, dtype=torch.float64)
    window_sizes = [4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 46, 59, 76, 98, 126, 161, 206]

    torch_fluctuation = dfa_fluctuation(trajectory, window_sizes, order=1)
    torch_alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), torch_fluctuation)

    # overlap=False and fit_exp='poly' match this module's convention
    # exactly (non-overlapping windows, plain OLS log-log slope); nolds'
    # own defaults (overlap=True, fit_exp='RANSAC') do not.
    nolds_alpha = nolds.dfa(trajectory.numpy(), nvals=window_sizes, overlap=False, order=1, fit_trend="poly", fit_exp="poly")

    assert abs(torch_alpha.item() - nolds_alpha) < 1e-9


def test_dfa_matches_antropy_default_sweep():
    from antropy.fractal import _log_n

    torch.manual_seed(7)
    trajectory = torch.randn(2000, dtype=torch.float64)
    # antropy.detrended_fluctuation's own fixed sweep: log-spaced nvals
    # from 4 to 0.1 * len(trajectory) by factor 1.2, order=1, non-overlapping.
    window_sizes = [int(value) for value in _log_n(4, 0.1 * len(trajectory), 1.2)]

    torch_fluctuation = dfa_fluctuation(trajectory, window_sizes, order=1)
    torch_alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), torch_fluctuation)

    import antropy

    antropy_alpha = antropy.detrended_fluctuation(trajectory.numpy())
    # Numba's internal accumulation order differs slightly from this
    # module's (torch) reduction order; both use the same formula and
    # sweep, so the tolerance only needs to absorb float64 roundoff.
    assert abs(torch_alpha.item() - antropy_alpha) < 1e-6


def test_dfa_matches_nolds_higher_order():
    import nolds

    torch.manual_seed(7)
    trajectory = torch.randn(1500, dtype=torch.float64)
    window_sizes = [10, 20, 40, 80, 160]
    for order in (0, 2, 3):
        fluctuation = dfa_fluctuation(trajectory, window_sizes, order=order)
        torch_alpha = dfa_exponent(window_sizes, fluctuation)
        nolds_alpha = nolds.dfa(trajectory.numpy(), nvals=window_sizes, overlap=False, order=order, fit_trend="poly", fit_exp="poly")
        assert abs(torch_alpha.item() - nolds_alpha) < 1e-9


def test_dfa_fluctuation_heterogeneous_batch_matches_individual():
    torch.manual_seed(8)
    white = torch.randn(400, dtype=torch.float64)
    brownian = torch.cumsum(torch.randn(400, dtype=torch.float64), dim=0)
    batch = torch.stack([white, brownian]).unsqueeze(-1)
    window_sizes = [8, 16, 32]
    result = dfa_fluctuation(batch, window_sizes, order=2)
    for index, series in enumerate((white, brownian)):
        assert torch.allclose(result[:, index], dfa_fluctuation(series, window_sizes, order=2))


def test_dfa_fluctuation_non_finite_raises():
    for bad in (torch.tensor([1.0, 2.0, 3.0, float("nan")]), torch.full((8,), float("inf"))):
        with pytest.raises(ValueError):
            dfa_fluctuation(bad, 4)


def test_dfa_fluctuation_integer_and_half_input():
    integer = torch.arange(8) ** 2
    assert torch.isclose(dfa_fluctuation(integer, 4).double(), dfa_fluctuation(integer.double(), 4), atol=1e-5)
    half = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0], dtype=torch.float16)
    result = dfa_fluctuation(half, 4)
    assert result.dtype == torch.float16
    assert torch.isclose(result.double(), dfa_fluctuation(half.double(), 4), atol=1e-2)


def test_dfa_fluctuation_zero_fluctuation_has_finite_gradient():
    # Nonconstant series whose single kept window has an exactly linear
    # profile: F(4) = 0, so the sqrt must be masked for backward.
    trajectory = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
    fluctuation = dfa_fluctuation(trajectory, 4)
    assert fluctuation.item() == 0.0
    fluctuation.backward()
    assert torch.isfinite(trajectory.grad).all()


def test_dfa_fluctuation_high_order_is_well_conditioned():
    # A degree-20 Legendre polynomial profile is removed exactly by an
    # order-20 detrend; a monomial Vandermonde basis in float32 left ~2.6e-2.
    import numpy as np

    abscissa = np.linspace(-1, 1, 64)
    profile = np.polynomial.legendre.legval(abscissa, [0] * 20 + [1])
    trajectory = torch.tensor(np.diff(np.r_[0, profile - profile[-1]]), dtype=torch.float32)
    assert dfa_fluctuation(trajectory, 64, order=20).item() < 1e-5


def test_dfa_exponent_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        dfa_exponent([4, 8], torch.tensor([2.0]))


def test_dfa_fluctuation_extreme_amplitude_scaling():
    # F is homogeneous of degree 1 in the amplitude; squaring raw float32
    # residuals overflowed at 1e20 and underflowed to 0 at 1e-25.
    base = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0])
    reference = dfa_fluctuation(base.double(), 4)
    for factor in (1e20, 1e-25):
        result = dfa_fluctuation(base * factor, 4)
        assert torch.isclose(result.double() / factor, reference, rtol=1e-5)
    # float16 input is promoted internally: raw squares of ~1e3 overflow half.
    half = dfa_fluctuation(base.half() * 1000, 4)
    assert torch.isfinite(half) and torch.isclose(half.double(), reference * 1000, rtol=1e-2)


def test_dfa_fluctuation_complex_input_raises():
    with pytest.raises(ValueError):
        dfa_fluctuation(torch.arange(16.0) + 1j, 4)
    with pytest.raises(ValueError):
        dfa_fluctuation(torch.arange(16.0), torch.tensor(4 + 2j))


def test_dfa_fluctuation_half_profile_exceeding_half_range():
    # The profile of this ramp reaches about -1e5, beyond float16's 65504;
    # computing in float16 would overflow the cumulative sum.
    ramp = torch.linspace(0.0, 200.0, 4000)
    result = dfa_fluctuation(ramp.half(), 400)
    assert torch.isfinite(result)
    assert torch.isclose(result.double(), dfa_fluctuation(ramp.half().double(), 400), rtol=1e-2)
