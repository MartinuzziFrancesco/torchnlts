"""Tests for torchnlts.fluctuations.dfa.

Dual ground truth:
  1. Paper: Peng et al. (1994) -- checked via a hand-derived example.
  2. Julia: FluctuationAnalysis.jl's ``dfa`` (bidirectional=false), plus
     nolds.dfa (overlap=False, fit_exp='poly') and
     antropy.detrended_fluctuation (order=1, its fixed log-spaced sweep).
"""

import pytest
import torch

from tests.conftest import LargestOutput, requires_nolds, run_julia
from torchnlts.fluctuations.dfa import dfa_exponent, dfa_fluctuation


def test_dfa_fluctuation_hand_derived():
    # Profile y = cumsum(x - 2.375). window_size=4, order=1: OLS residuals [0.1, 0.2, -0.7, 0.4] and [-0.1, 0.8, -1.3,
    # 0.6], sums of squares 0.7 and 2.7; F(4) = sqrt((0.7 + 2.7) / 8) = sqrt(0.425) = 0.6519202405202648.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    fluctuation = dfa_fluctuation(trajectory, 4, order=1)
    assert torch.isclose(fluctuation, torch.tensor(0.6519202405202648, dtype=torch.float64))


def test_dfa_fluctuation_leftover_samples_are_dropped():
    # Trailing samples that do not fill a window are dropped, as in nolds/antropy.
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
    with pytest.raises(ValueError):
        dfa_fluctuation(trajectory, 4)


def test_dfa_fluctuation_negative_order_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        dfa_fluctuation(trajectory, 4, order=-1)


def test_dfa_fluctuation_empty_window_size_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        dfa_fluctuation(trajectory, [])


def test_dfa_fluctuation_window_too_small_for_order_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        dfa_fluctuation(trajectory, 2, order=1)


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
    # Peng et al. (1994): alpha ~ 0.5 for white noise.
    torch.manual_seed(4)
    trajectory = torch.randn(8000, dtype=torch.float64)
    window_sizes = [10, 13, 16, 21, 26, 34, 43, 55, 70, 89, 113, 144, 183, 234, 298, 379, 483, 616, 785, 1000]
    fluctuation = dfa_fluctuation(trajectory, window_sizes)
    alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), fluctuation)
    assert abs(alpha.item() - 0.5) < 0.05


def test_dfa_exponent_brownian_motion_is_near_one_point_five():
    # Peng et al. (1994): alpha ~ 1.5 for Brownian motion.
    torch.manual_seed(5)
    white_noise = torch.randn(8000, dtype=torch.float64)
    brownian_motion = torch.cumsum(white_noise, dim=0)
    window_sizes = [10, 13, 16, 21, 26, 34, 43, 55, 70, 89, 113, 144, 183, 234, 298, 379, 483, 616, 785, 1000]
    fluctuation = dfa_fluctuation(brownian_motion, window_sizes)
    alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), fluctuation)
    assert abs(alpha.item() - 1.5) < 0.05


@requires_nolds
def test_dfa_matches_nolds_non_overlapping():
    import nolds

    torch.manual_seed(6)
    trajectory = torch.randn(2000, dtype=torch.float64)
    window_sizes = [4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 46, 59, 76, 98, 126, 161, 206]

    torch_fluctuation = dfa_fluctuation(trajectory, window_sizes, order=1)
    torch_alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), torch_fluctuation)

    # overlap=False and fit_exp='poly' match this module's non-overlapping windows and OLS log-log slope.
    nolds_alpha = nolds.dfa(
        trajectory.numpy(), nvals=window_sizes, overlap=False, order=1, fit_trend="poly", fit_exp="poly"
    )

    assert abs(torch_alpha.item() - nolds_alpha) < 1e-9


def test_dfa_matches_antropy_default_sweep():
    from antropy.fractal import _log_n

    torch.manual_seed(7)
    trajectory = torch.randn(2000, dtype=torch.float64)
    # antropy's fixed sweep: log-spaced nvals from 4 to 0.1 * len(trajectory) by factor 1.2, order=1, non-overlapping.
    window_sizes = [int(value) for value in _log_n(4, 0.1 * len(trajectory), 1.2)]

    torch_fluctuation = dfa_fluctuation(trajectory, window_sizes, order=1)
    torch_alpha = dfa_exponent(torch.tensor(window_sizes, dtype=torch.float64), torch_fluctuation)

    import antropy

    antropy_alpha = antropy.detrended_fluctuation(trajectory.numpy())
    # Numba and torch reduce in a different order; the tolerance only absorbs float64 roundoff.
    assert abs(torch_alpha.item() - antropy_alpha) < 1e-6


@requires_nolds
def test_dfa_matches_nolds_higher_order():
    import nolds

    torch.manual_seed(7)
    trajectory = torch.randn(1500, dtype=torch.float64)
    window_sizes = [10, 20, 40, 80, 160]
    for order in (0, 2, 3):
        fluctuation = dfa_fluctuation(trajectory, window_sizes, order=order)
        torch_alpha = dfa_exponent(window_sizes, fluctuation)
        nolds_alpha = nolds.dfa(
            trajectory.numpy(), nvals=window_sizes, overlap=False, order=order, fit_trend="poly", fit_exp="poly"
        )
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


def test_dfa_fluctuation_integer_and_half_input():
    integer = torch.arange(8) ** 2
    assert torch.isclose(dfa_fluctuation(integer, 4).double(), dfa_fluctuation(integer.double(), 4), atol=1e-5)
    half = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0], dtype=torch.float16)
    result = dfa_fluctuation(half, 4)
    assert result.dtype == torch.float16
    assert torch.isclose(result.double(), dfa_fluctuation(half.double(), 4), atol=1e-2)


def test_dfa_fluctuation_zero_fluctuation_has_finite_gradient():
    # Single kept window with an exactly linear profile: F(4) = 0, so the sqrt must be masked for backward.
    trajectory = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
    fluctuation = dfa_fluctuation(trajectory, 4)
    assert fluctuation.item() == 0.0
    fluctuation.backward()
    assert torch.isfinite(trajectory.grad).all()


def test_dfa_fluctuation_high_order_is_well_conditioned():
    # A degree-20 Legendre profile is removed exactly by an order-20 detrend; a monomial basis in float32 left ~2.6e-2.
    import numpy as np

    abscissa = np.linspace(-1, 1, 64)
    profile = np.polynomial.legendre.legval(abscissa, [0] * 20 + [1])
    trajectory = torch.tensor(np.diff(np.r_[0, profile - profile[-1]]), dtype=torch.float32)
    assert dfa_fluctuation(trajectory, 64, order=20).item() < 1e-5


def test_dfa_exponent_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        dfa_exponent([4, 8], torch.tensor([2.0]))


def test_dfa_fluctuation_extreme_amplitude_scaling():
    # F is homogeneous of degree 1; squaring raw float32 residuals overflowed at 1e20 and underflowed to 0 at 1e-25.
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
    # The ramp's profile reaches about -1e5, beyond float16's 65504; a float16 cumulative sum would overflow.
    ramp = torch.linspace(0.0, 200.0, 4000)
    result = dfa_fluctuation(ramp.half(), 400)
    assert torch.isfinite(result)
    assert torch.isclose(result.double(), dfa_fluctuation(ramp.half().double(), 400), rtol=1e-2)


def test_dfa_fluctuation_profile_does_not_overflow():
    # Finite float32 input whose mean/profile sums overflow float32 unless the amplitude is normalized first.
    base = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0])
    result = dfa_fluctuation(base * 5e37, 4)
    assert torch.isfinite(result)
    assert torch.isclose(result.double(), dfa_fluctuation(base.double(), 4) * 5e37, rtol=1e-5)


def test_dfa_fluctuation_complex_window_size_any_type_raises():
    import numpy as np

    for window_size in (np.array([4 + 2j]), 4 + 2j, [4 + 0j]):
        with pytest.raises(ValueError):
            dfa_fluctuation(torch.arange(16.0), window_size)


def test_dfa_exponent_degenerate_column_does_not_poison_shared_gradient():
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    fluctuation = parameter * torch.tensor([[0.0, 2.0], [0.0, 4.0], [0.0, 8.0]], dtype=torch.float64)
    exponent = dfa_exponent([4, 8, 16], fluctuation)
    assert torch.isnan(exponent[0]) and torch.isclose(exponent[1], torch.tensor(1.0, dtype=torch.float64))
    (gradient,) = torch.autograd.grad(exponent[1], parameter)
    assert torch.isclose(gradient, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_dfa_matches_fluctuationanalysis_jl():
    # bidirectional=false is Julia's forward-only segmentation; both are float64, so agreement is to rounding.
    import pathlib

    project = pathlib.Path.home() / ".julia" / "dev" / "FluctuationAnalysis"
    if not project.exists():
        pytest.skip("FluctuationAnalysis.jl not available")
    torch.manual_seed(3)
    trajectory = torch.randn(1003, dtype=torch.float64)
    window_sizes = [10, 20, 40, 80, 160]
    values = ", ".join(f"{value:.17g}" for value in trajectory.numpy())
    for order in (1, 2):
        julia_code = f"""
        using FluctuationAnalysis
        x = [{values}]
        result = dfa(x; scales={window_sizes}, order={order}, bidirectional=false)
        println(join(result.fluctuations, ","))
        println(scaling_exponent(result))
        """
        stdout = run_julia(julia_code, str(project))
        lines = stdout.strip().split("\n")
        julia_fluctuation = torch.tensor([float(item) for item in lines[0].split(",")], dtype=torch.float64)
        fluctuation = dfa_fluctuation(trajectory, window_sizes, order=order)
        assert torch.allclose(fluctuation, julia_fluctuation, rtol=0, atol=1e-12)
        assert abs(dfa_exponent(window_sizes, fluctuation).item() - float(lines[1])) < 1e-12


def test_dfa_fluctuation_large_amplitude_gradient_is_scale_invariant():
    base = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0])
    unit = base.clone().requires_grad_(True)
    (unit_gradient,) = torch.autograd.grad(dfa_fluctuation(unit, 4), unit)
    large = (base * 1e20).requires_grad_(True)
    (large_gradient,) = torch.autograd.grad(dfa_fluctuation(large, 4), large)
    assert torch.allclose(large_gradient, unit_gradient, atol=1e-6)


def test_dfa_fluctuation_second_derivative():
    # Amplitude scaling must be exact for double backward too (a forward-only scaling trick gave a Hessian off by the
    # scale).
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda series: dfa_fluctuation(series, 4), (trajectory,))


def test_dfa_fluctuation_small_kept_window_next_to_large_dropped_samples():
    # The kept window is ~1e-25 while the dropped tail is ~1; normalizing by the row maximum alone underflowed the
    # squares to 0.
    for dtype, tiny in ((torch.float32, 1e-25), (torch.float64, 1e-170)):
        trajectory = torch.tensor([1 * tiny, 2 * tiny, 1 * tiny, 3 * tiny, 1.0, -1.0], dtype=dtype, requires_grad=True)
        result = dfa_fluctuation(trajectory, 4)
        reference = (
            dfa_fluctuation(torch.tensor([1.0, 2.0, 1.0, 3.0, 1e25, -1e25], dtype=torch.float64), 4).item() * tiny
        )
        assert abs(result.item() / reference - 1) < 1e-4
        (gradient,) = torch.autograd.grad(result, trajectory)
        assert torch.isfinite(gradient).all() and gradient[:4].abs().sum() > 0


def test_dfa_fluctuation_clongdouble_window_size_raises():
    import numpy as np

    for window_size in (np.clongdouble(4 + 0j), np.array([4 + 0j], dtype=np.clongdouble)):
        with pytest.raises(ValueError):
            dfa_fluctuation(torch.arange(8.0), window_size)


def test_dfa_exponent_invalid_window_size_does_not_poison_shared_gradient():
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    exponent = dfa_exponent(
        parameter * torch.tensor([0.0, 4.0, 8.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64),
    )
    assert torch.isnan(exponent)
    (gradient,) = torch.autograd.grad(torch.nan_to_num(exponent) + parameter, parameter)
    assert gradient.item() == 1.0


def test_dfa_fluctuation_bfloat16_matches_float32():
    torch.manual_seed(11)
    trajectory = torch.randn(4096).bfloat16()
    result = dfa_fluctuation(trajectory, [8, 16, 1024])
    assert result.dtype == torch.bfloat16
    reference = dfa_fluctuation(trajectory.float(), [8, 16, 1024])
    assert torch.allclose(result.float(), reference, rtol=1e-2)


def test_dfa_fluctuation_same_sign_dropped_tail():
    # The dropped tail [1, 1] shifts the global mean to ~1/3; a global profile erased the 1e-10 kept window (F 715x too
    # large).
    trajectory = torch.tensor([1e-10, 2e-10, 1e-10, 3e-10, 1.0, 1.0], dtype=torch.float32, requires_grad=True)
    result = dfa_fluctuation(trajectory, 4)
    assert abs(result.item() / ((0.7 / 4) ** 0.5 * 1e-10) - 1) < 1e-4
    (gradient,) = torch.autograd.grad(result, trajectory)
    unit = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    (unit_gradient,) = torch.autograd.grad(dfa_fluctuation(unit, 4), unit)
    assert torch.allclose(gradient[:4].double(), unit_gradient, atol=1e-5)
    assert (gradient[4:] == 0).all()


@pytest.mark.parametrize("order", [0, 1, 2])
def test_large_offset_matches_float64_reference(order):
    # Samples near 1e6 are exact in float32; F must not lose their spread.
    # The last sample makes the float32 mean inexact (sum 1217 over 512 samples).
    series = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0]).repeat(64) + 1e6
    series[-1] += 1.0
    series.requires_grad_()
    precise = series.detach().double().requires_grad_()
    actual = dfa_fluctuation(series, 256, order=order)
    expected = dfa_fluctuation(precise - 1e6, 256, order=order)
    assert torch.allclose(actual.double(), expected, rtol=1e-4)
    (actual_gradient,) = torch.autograd.grad(actual, series)
    (expected_gradient,) = torch.autograd.grad(expected, precise)
    assert torch.allclose(actual_gradient.double(), expected_gradient, atol=1e-5)


def test_half_input_accumulates_in_float32():
    # A half-precision cumulative sum over 8192 samples is off by ~2e-3.
    generator = torch.Generator().manual_seed(0)
    series = (torch.rand(8192, generator=generator) + 1.0).half()
    result = dfa_fluctuation(series, 8192, order=0)
    assert torch.isclose(result.double(), dfa_fluctuation(series.double(), 8192, order=0), rtol=1e-3)


def test_order_zero_large_float64_offset():
    shifted = torch.tensor([0.0, 2.0, 0.0, 4.0], dtype=torch.float64) + 1e16
    assert torch.isclose(dfa_fluctuation(shifted, 4, order=0), torch.tensor((13 / 16) ** 0.5, dtype=torch.float64))


def test_bfloat16_input_is_computed_in_float32():
    series = torch.randn(256, generator=torch.Generator().manual_seed(0)).bfloat16()
    assert dfa_fluctuation(series, 16) == dfa_fluctuation(series.float(), 16).bfloat16()


def test_spike_cancelled_by_detrending():
    series = torch.tensor([1e8, 1.0, 2.0, 3.0], requires_grad=True)
    result = dfa_fluctuation(series, 4)
    (gradient,) = torch.autograd.grad(result, series)
    assert torch.isclose(result, torch.tensor(0.5))
    assert torch.allclose(gradient, torch.tensor([0.0, -0.25, 0.0, 0.25]), atol=1e-6)


def test_dominant_dropped_tail_does_not_affect_detrended_windows():
    # For order >= 1, F depends only on the kept samples; a global profile would carry the 1e14 tail's mean into every
    # window.
    kept = torch.randn(512, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    series = torch.cat([kept, torch.full((100,), 1e14, dtype=torch.float64)])
    for order in (1, 2):
        assert torch.isclose(
            dfa_fluctuation(series, 256, order=order), dfa_fluctuation(kept, 256, order=order), rtol=1e-10
        )


@pytest.mark.parametrize("order", [1, 2])
def test_large_float64_offset_needs_window_centering(order):
    # An uncentered cumulative sum of 1e10-sized samples loses the unit-sized residual far beyond epsilon times the
    # largest sample.
    base = torch.randn(1024, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    expected = dfa_fluctuation(base, 256, order=order)
    assert torch.isclose(dfa_fluctuation(base + 1e10, 256, order=order), expected, rtol=1e-6)


def test_inexact_constant_series_has_zero_fluctuation_and_gradient():
    # 0.1 is not binary-exact: window means round, so only the mask gives exact zeros.
    series = torch.full((6,), 0.1, dtype=torch.float64, requires_grad=True)
    result = dfa_fluctuation(series, 3)
    (gradient,) = torch.autograd.grad(result, series)
    assert result == 0 and torch.all(gradient == 0)


def test_memory_does_not_grow_with_the_square_of_the_window_size():
    # A dense window_size x window_size detrending matrix is 4e8 elements (3.2 GB in float64) at window_size 20000.
    import numpy as np

    num_samples, window_size = 40000, 20000
    series = torch.randn(num_samples, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with LargestOutput() as recorder:
        fluctuation = dfa_fluctuation(series, window_size)
    assert recorder.numel <= 2 * num_samples

    index = np.arange(window_size)
    squares = []
    for window in series.numpy().reshape(-1, window_size):
        profile = np.cumsum(window - window.mean())
        squares.append((profile - np.polyval(np.polyfit(index, profile, 1), index)) ** 2)
    assert fluctuation.item() == pytest.approx(np.sqrt(np.mean(squares)), rel=1e-9)


def test_order_accepts_numpy_integers_and_rejects_other_types():
    import numpy as np

    trajectory = torch.randn(64, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    expected = dfa_fluctuation(trajectory, [8, 16], order=2)
    for order in (np.int64(2), np.int32(2), np.array(2), torch.tensor(2)):
        assert torch.equal(dfa_fluctuation(trajectory, [8, 16], order=order), expected)
    for order in (1.0, np.float64(1.0), 1.5, "1", None, -1, np.int64(-1)):
        with pytest.raises(ValueError, match="order"):
            dfa_fluctuation(trajectory, [8, 16], order=order)
