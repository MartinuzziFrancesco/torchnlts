"""Tests for torchnlts.dimension.variogram.

Dual ground truth:
  1. Paper: Gneiting, Sevcikova & Percival (2012), Statist. Sci. 27(2),
     eq. (4)/(6) -- checked via a hand-derived example, and via known
     analytic limits: a straight line (D=1), Brownian motion (D~1.5),
     and fractional Gaussian noise of known Hurst H (D~2-H).
  2. Reference implementation: no Python/Julia equivalent is installed,
     and R itself is not available in this environment. The R package
     `fractaldim`'s own source (CRAN, `cran/fractaldim` on GitHub,
     `R/variation.R`) was read directly and confirms this module's
     normalization, default two-lag sweep, and D = 2 - slope/p
     regression convention -- see the module docstring for the details
     of that comparison. In place of a runtime cross-check, the
     analytic-limit tests above serve as the numerical ground truth.
"""

import math

import numpy
import pytest
import torch

from torchnlts.dimension.variogram import power_variogram, variogram_dimension


def _fgn_davies_harte(hurst: float, length: int, dtype=torch.float64) -> torch.Tensor:
    """Fractional Gaussian noise via Davies & Harte (1987) circulant
    embedding, algorithm confirmed against the reference Python
    implementation in the `fbm` package (crflynn/fbm, `_daviesharte`).
    Used only to build a known-Hurst fBm ground truth for this test
    module, not part of the public API.
    """

    def autocovariance(lag: int) -> float:
        return 0.5 * (abs(lag - 1) ** (2 * hurst) - 2 * abs(lag) ** (2 * hurst) + abs(lag + 1) ** (2 * hurst))

    row = (
        [autocovariance(0)]
        + [autocovariance(index) for index in range(1, length)]
        + [0.0]
        + [autocovariance(index) for index in range(length - 1, 0, -1)]
    )
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    eigenvalues = torch.fft.fft(torch.tensor(row, dtype=dtype).to(complex_dtype)).real
    assert (eigenvalues >= -1e-6).all(), "Davies-Harte embedding is not positive semi-definite for this (hurst, length)"
    eigenvalues = eigenvalues.clamp(min=0.0)

    gaussian_one = torch.randn(2 * length, dtype=dtype)
    gaussian_two = torch.randn(2 * length, dtype=dtype)
    weights = torch.zeros(2 * length, dtype=complex_dtype)
    weights[0] = torch.sqrt(eigenvalues[0] / (2 * length)) * gaussian_one[0]
    weights[length] = torch.sqrt(eigenvalues[length] / (2 * length)) * gaussian_two[0]
    middle = torch.arange(1, length)
    weights[middle] = torch.sqrt(eigenvalues[middle] / (4 * length)) * (
        gaussian_one[middle] + 1j * gaussian_two[middle]
    )
    weights[2 * length - middle] = torch.sqrt(eigenvalues[2 * length - middle] / (4 * length)) * (
        gaussian_one[middle] - 1j * gaussian_two[middle]
    )
    return torch.fft.fft(weights).real[:length]


def test_power_variogram_hand_derived():
    # x = [1, 2, 4, 3, 6], n=5.
    # lag=1: diffs = 2-1,4-2,3-4,6-3 = 1,2,-1,3 -> |.|=1,2,1,3, 4 terms.
    #   V_1(1) = (1+2+1+3)/(2*4) = 7/8 = 0.875
    #   V_2(1) = (1+4+1+9)/(2*4) = 15/8 = 1.875
    # lag=2: diffs = 4-1,3-2,6-4 = 3,1,2 -> |.|=3,1,2, 3 terms.
    #   V_1(2) = (3+1+2)/(2*3) = 6/6 = 1.0
    #   V_2(2) = (9+1+4)/(2*3) = 14/6 = 2.3333333333333335
    trajectory = torch.tensor([1.0, 2.0, 4.0, 3.0, 6.0], dtype=torch.float64)
    madogram = power_variogram(trajectory, [1, 2], power=1.0)
    assert torch.allclose(madogram, torch.tensor([0.875, 1.0], dtype=torch.float64))
    variogram = power_variogram(trajectory, [1, 2], power=2.0)
    assert torch.allclose(variogram, torch.tensor([1.875, 2.3333333333333335], dtype=torch.float64))


def test_variogram_dimension_hand_derived():
    # Continuing the example above: D = 2 - slope/p, slope the OLS
    # (two-point) slope of log(V) against log(lag).
    lag = torch.tensor([1.0, 2.0], dtype=torch.float64)
    madogram = torch.tensor([0.875, 1.0], dtype=torch.float64)
    slope_p1 = (math.log(1.0) - math.log(0.875)) / (math.log(2.0) - math.log(1.0))
    expected_p1 = 2.0 - slope_p1 / 1.0
    assert torch.isclose(variogram_dimension(lag, madogram, power=1.0), torch.tensor(expected_p1, dtype=torch.float64))

    variogram = torch.tensor([1.875, 2.3333333333333335], dtype=torch.float64)
    slope_p2 = (math.log(2.3333333333333335) - math.log(1.875)) / (math.log(2.0) - math.log(1.0))
    expected_p2 = 2.0 - slope_p2 / 2.0
    assert torch.isclose(variogram_dimension(lag, variogram, power=2.0), torch.tensor(expected_p2, dtype=torch.float64))


def test_variogram_dimension_straight_line_is_one():
    # V_p(l) is proportional to l**p on a straight line (every increment
    # over lag l is exactly l * slope), so D = 2 - p/p = 1, for any p.
    line = torch.arange(0.0, 200.0, dtype=torch.float64) * 2.5 + 1.0
    lag = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    for power in (0.5, 1.0, 2.0, 3.0):
        variogram = power_variogram(line, [1, 2, 4], power=power)
        dimension = variogram_dimension(lag, variogram, power=power)
        assert torch.isclose(dimension, torch.tensor(1.0, dtype=torch.float64), atol=1e-8)


def test_variogram_dimension_constant_trajectory_is_nan():
    # A constant series has V_p(l) = 0 at every lag -> log(0) = -inf ->
    # an indeterminate -inf - (-inf) in the OLS slope's centering step.
    constant = torch.full((50,), 3.0, dtype=torch.float64)
    variogram = power_variogram(constant, [1, 2])
    assert torch.equal(variogram, torch.zeros(2, dtype=torch.float64))
    dimension = variogram_dimension(torch.tensor([1.0, 2.0], dtype=torch.float64), variogram)
    assert torch.isnan(dimension)


def test_power_variogram_degenerate_lag_is_nan_not_crash():
    # n=3: lag=3 leaves 0 terms (n - lag < 1).
    trajectory = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(power_variogram(trajectory, 3))
    swept = power_variogram(trajectory, [1, 3])
    assert torch.isfinite(swept[0]) and torch.isnan(swept[1])


def test_power_variogram_empty_trajectory_is_nan_not_crash():
    empty = torch.tensor([], dtype=torch.float64)
    assert torch.isnan(power_variogram(empty, [1, 2])).all()


def test_power_variogram_multivariate_raises():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        power_variogram(trajectory, [1, 2])


def test_power_variogram_invalid_input_raises():
    for bad in (torch.tensor([0j, 1j, 2j]),):
        with pytest.raises(ValueError):
            power_variogram(bad, [1, 2])


def test_power_variogram_invalid_power_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    for bad_power in (0.0, -1.0):
        with pytest.raises(ValueError):
            power_variogram(trajectory, [1, 2], power=bad_power)
    with pytest.raises(ValueError):
        variogram_dimension(
            torch.tensor([1.0, 2.0], dtype=torch.float64), torch.tensor([1.0, 2.0], dtype=torch.float64), power=0.0
        )


def test_power_variogram_empty_or_bad_lag_raises():
    trajectory = torch.randn(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        power_variogram(trajectory, [])
    with pytest.raises(ValueError):
        power_variogram(trajectory, [1.5, 2.5])
    with pytest.raises(ValueError):
        power_variogram(trajectory, [0, 1])


def test_power_variogram_batched_matches_unbatched():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 30, 1, dtype=torch.float64)
    batched = power_variogram(trajectory, [1, 2, 3])
    assert batched.shape == (3, 4)
    for index in range(4):
        assert torch.allclose(batched[:, index], power_variogram(trajectory[index], [1, 2, 3]))


def test_power_variogram_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 30, 1, dtype=torch.float64)
    batched = power_variogram(trajectory, [1, 2])
    unbatched = power_variogram(trajectory[0], [1, 2])
    assert batched.shape == (2, 1)
    assert torch.allclose(batched[:, 0], unbatched)


def test_power_variogram_unbatched_scalar_lag_is_scalar():
    trajectory = torch.randn(20, dtype=torch.float64)
    assert power_variogram(trajectory, 3).shape == ()


def test_variogram_dimension_batched_matches_unbatched():
    torch.manual_seed(2)
    trajectory = torch.randn(5, 100, 1, dtype=torch.float64)
    lag = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    batched_variogram = power_variogram(trajectory, [1, 2, 3])
    batched_dimension = variogram_dimension(lag, batched_variogram)
    assert batched_dimension.shape == (5,)
    for index in range(5):
        unbatched_variogram = power_variogram(trajectory[index], [1, 2, 3])
        assert torch.isclose(batched_dimension[index], variogram_dimension(lag, unbatched_variogram))


def test_variogram_dimension_single_lag_is_nan_not_crash():
    result = variogram_dimension(torch.tensor([1.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    assert torch.isnan(result)


def test_power_variogram_gradcheck_power_two():
    torch.manual_seed(3)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: power_variogram(values, [1, 2, 3], power=2.0), (trajectory,))


def test_power_variogram_gradcheck_power_one():
    torch.manual_seed(4)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: power_variogram(values, [1, 2, 3], power=1.0), (trajectory,))


def test_variogram_dimension_gradcheck_through_trajectory():
    torch.manual_seed(5)
    trajectory = torch.randn(25, dtype=torch.float64, requires_grad=True)
    lag = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    def dimension(values):
        return variogram_dimension(lag, power_variogram(values, [1, 2, 3], power=2.0), power=2.0)

    assert torch.autograd.gradcheck(dimension, (trajectory,))


def test_power_variogram_sub_one_power_zero_increment_gradient_is_finite():
    # power<1: d|x|**p/dx is unbounded at a zero increment (x[0] == x[1] in
    # row 0); that term is masked to a zero gradient, so no NaN anywhere.
    torch.manual_seed(6)
    row_with_tie = torch.tensor([1.0, 1.0, 2.0, 5.0, 3.0], dtype=torch.float64)
    row_without_tie = torch.randn(5, dtype=torch.float64)
    batch = torch.stack([row_with_tie, row_without_tie]).unsqueeze(-1)
    batch.requires_grad_(True)

    result = power_variogram(batch, [1, 2], power=0.5)
    result.sum().backward()

    assert torch.isfinite(batch.grad).all()
    # Masking must not change the forward value: the zero increment adds 0.
    expected = (torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64).sqrt().sum() / (2 * 4)).item()
    assert abs(result[0, 0].item() - expected) < 1e-12


def test_power_variogram_madogram_kink_gradient_is_finite():
    # power=1: abs() has a defined (zero) subgradient at a tie, not an
    # infinite derivative -- unlike power<1 above.
    trajectory = torch.tensor([1.0, 1.0, 3.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True)
    power_variogram(trajectory, [1, 2], power=1.0).sum().backward()
    assert torch.isfinite(trajectory.grad).all()


def test_power_variogram_matches_fractaldim_r_convention_normalization():
    # Direct check of the normalization documented from fractaldim's
    # fd.Variation (R/variation.R): g[j] = sum(|diff|**p) / (2*(n-lag)).
    # No R runtime available; this reproduces that exact formula
    # independently in plain Python and compares against this module.
    torch.manual_seed(7)
    trajectory = torch.randn(15, dtype=torch.float64)
    lag = 3
    power = 2.0
    values = trajectory.tolist()
    num_points = len(values)
    expected = sum(abs(values[index + lag] - values[index]) ** power for index in range(num_points - lag)) / (
        2 * (num_points - lag)
    )
    assert torch.isclose(power_variogram(trajectory, lag, power=power), torch.tensor(expected, dtype=torch.float64))


def test_power_variogram_integer_and_half_input():
    unsigned = torch.tensor([1, 2, 1, 3], dtype=torch.uint8)  # 1 - 2 would wrap in uint8
    assert torch.isclose(power_variogram(unsigned, [1]).double(), power_variogram(unsigned.double(), [1]), atol=1e-6)
    half = power_variogram(torch.zeros(50, dtype=torch.float16), [1, 2])
    assert half.dtype == torch.float16 and torch.equal(half, torch.zeros(2, dtype=torch.float16))
    # power=2 pushes values into the thousands, where float16's absolute
    # step size is itself large; compare in relative terms.
    wide = torch.tensor([0.0, 300.0, 0.0, 100.0])
    assert torch.allclose(
        power_variogram(wide.half(), [1, 2]).double(), power_variogram(wide.double(), [1, 2]), rtol=1e-2
    )


def test_variogram_dimension_brownian_motion_is_one_half():
    # A cumulative sum of white noise is Brownian motion, whose known
    # graph dimension is D=1.5 for both the variogram and the madogram.
    torch.manual_seed(8)
    brownian = torch.randn(5000, dtype=torch.float64).cumsum(0)
    lag = torch.tensor([1.0, 2.0], dtype=torch.float64)
    variogram_estimate = variogram_dimension(lag, power_variogram(brownian, [1, 2], power=2.0), power=2.0)
    madogram_estimate = variogram_dimension(lag, power_variogram(brownian, [1, 2], power=1.0), power=1.0)
    assert torch.isclose(variogram_estimate, torch.tensor(1.5, dtype=torch.float64), atol=0.05)
    assert torch.isclose(madogram_estimate, torch.tensor(1.5, dtype=torch.float64), atol=0.05)


@pytest.mark.parametrize("hurst", [0.2, 0.35, 0.5, 0.65, 0.8])
def test_variogram_dimension_fractional_gaussian_noise_known_hurst(hurst):
    # fBm (cumulative fGn) of Hurst exponent H has known graph dimension
    # D = 2 - H, for the variogram or the madogram estimator alike.
    # fGn generated by exact Davies-Harte circulant embedding (see
    # _fgn_davies_harte above). The two-lag-only fit the paper
    # recommends is noisy on a single realization (empirically, std
    # ~0.007 across seeds at this length); a length of 20000 and
    # atol=0.03 keep this test from flaking while still checking the
    # right value to within the estimator's own known variance, not
    # exact equality.
    torch.manual_seed(int(hurst * 100))
    fractional_gaussian_noise = _fgn_davies_harte(hurst, 20000)
    fractional_brownian_motion = fractional_gaussian_noise.cumsum(0)
    lag = torch.tensor([1.0, 2.0], dtype=torch.float64)
    variogram_estimate = variogram_dimension(
        lag, power_variogram(fractional_brownian_motion, [1, 2], power=2.0), power=2.0
    )
    madogram_estimate = variogram_dimension(
        lag, power_variogram(fractional_brownian_motion, [1, 2], power=1.0), power=1.0
    )
    expected = 2.0 - hurst
    assert torch.isclose(variogram_estimate, torch.tensor(expected, dtype=torch.float64), atol=0.03)
    assert torch.isclose(madogram_estimate, torch.tensor(expected, dtype=torch.float64), atol=0.03)


def test_variogram_dimension_degenerate_column_does_not_poison_shared_gradient():
    values = torch.tensor([1.0, 4.0], dtype=torch.float64, requires_grad=True)
    batch = torch.stack([values * 0, values], dim=1)  # column 0: V = 0, log undefined
    dimension = variogram_dimension([1, 2], batch)
    assert torch.isnan(dimension[0])
    dimension[1].backward()
    # D = 2 - (log V2 - log V1) / (p log 2), p = 2: dD/dV1 = 1/(2 log2 V1), dD/dV2 = -1/(2 log2 V2).
    expected = torch.tensor([1 / (2 * math.log(2) * 1.0), -1 / (2 * math.log(2) * 4.0)], dtype=torch.float64)
    assert torch.allclose(values.grad, expected)


def test_variogram_dimension_half_precision_is_promoted():
    variogram = torch.tensor([1.0, 4.0], dtype=torch.float16)
    # lags above float16's range (65504) and close lags that float16 cannot separate well.
    result = variogram_dimension([100000, 200000], variogram)
    assert result.dtype == torch.float16 and result.item() == 1.0
    close = variogram_dimension([256, 257], variogram)
    assert torch.isclose(close.float(), variogram_dimension([256, 257], variogram.float()), rtol=1e-2)


def test_variogram_non_finite_power_raises():
    for power in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            power_variogram(torch.arange(10.0), [1, 2], power=power)
        with pytest.raises(ValueError):
            variogram_dimension([1, 2], torch.tensor([1.0, 2.0]), power=power)


def test_power_variogram_sub_one_power_gradient_matches_analytic():
    # V_p(l) = sum |dx|**p / (2 (n - l)); dV/dx_j from each increment
    # dx = x_{i+l} - x_i is p |dx|**(p-1) sign(dx) at x_{i+l}, minus that at x_i.
    # A zero increment contributes 0 (masked).
    power = 0.5
    trajectory = torch.tensor([1.0, 1.0, 2.0, 5.0, 3.0], dtype=torch.float64, requires_grad=True)
    for lag in (1, 2):
        (gradient,) = torch.autograd.grad(power_variogram(trajectory, lag, power=power), trajectory)
        expected = torch.zeros(5, dtype=torch.float64)
        values = trajectory.detach()
        for index in range(5 - lag):
            increment = values[index + lag] - values[index]
            if increment != 0:
                term = power * increment.abs() ** (power - 1) * torch.sign(increment) / (2 * (5 - lag))
                expected[index + lag] += term
                expected[index] -= term
        assert torch.allclose(gradient, expected)


def test_variogram_dimension_half_close_values_and_integer_input():
    # log(10008) - log(10000) = 8e-4 is below float16 resolution at 9.2.
    half = variogram_dimension([1000, 1001], torch.tensor([10000.0, 10008.0]).half())
    assert half.dtype == torch.float16
    assert torch.isclose(
        half.float(),
        variogram_dimension([1000, 1001], torch.tensor([10000.0, 10008.0], dtype=torch.float64)).float(),
        atol=2e-3,
    )
    integer = variogram_dimension([1, 2], torch.tensor([1, 4]))
    assert integer.dtype == torch.get_default_dtype() and integer.item() == 1.0


def test_variogram_complex_lag_raises():
    for lag in (torch.tensor([1 + 2j, 2 + 3j]), [1 + 0j, 2 + 0j]):
        with pytest.raises(ValueError):
            power_variogram(torch.arange(5.0), lag)
        with pytest.raises(ValueError):
            variogram_dimension(lag, torch.tensor([1.0, 4.0]))


def test_fgn_generator_rejects_invalid_embedding():
    # Davies-Harte's circulant embedding has negative eigenvalues for H close
    # to 1 at short lengths; the generators must refuse rather than clip.
    import pathlib

    with pytest.raises(AssertionError):
        _fgn_davies_harte(0.9, 4)
    # The tutorial's generator: run the script's definitions only (it starts
    # its simulation at torch.manual_seed).
    source = (pathlib.Path(__file__).parent.parent / "tutorials" / "variogram_dimension_fbm.py").read_text()
    namespace = {}
    exec(source[: source.index("torch.manual_seed(0)")], namespace)
    with pytest.raises(AssertionError):
        namespace["fractional_gaussian_noise"](0.95, 16)


def test_power_variogram_large_increments_float32():
    # Squares of 1e19 increments overflow float32 before averaging.
    trajectory = torch.arange(4.0) * 1e19
    result = power_variogram(trajectory, [1, 2])
    assert torch.allclose(result.double(), power_variogram(trajectory.double(), [1, 2]), rtol=1e-5)


def test_power_variogram_small_powers_and_half():
    # Unit increments: V_p(1) = 1 / 2 for every p > 0. (2n)**(1/p) overflowed
    # at p = 0.01 and raised OverflowError at p = 0.001.
    trajectory = torch.arange(8.0)
    for power in (0.001, 0.01, 0.5, 2.0, 50.0):
        assert torch.isclose(power_variogram(trajectory, 1, power=power), torch.tensor(0.5))
    # Without float32 promotion, float16 gives 0 here.
    assert power_variogram(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float16), 1, power=0.05).item() == 0.5


def test_power_must_be_a_real_scalar():
    for power in (1 + 0j, torch.tensor([1.0, 2.0])):
        with pytest.raises(ValueError, match="power"):
            power_variogram(torch.arange(5.0), 1, power=power)
        with pytest.raises(ValueError, match="power"):
            variogram_dimension([1, 2], torch.tensor([1.0, 4.0]), power=power)


def test_half_input_sums_in_float32():
    # 1e5 unit terms overflow a float16 sum.
    trajectory = (torch.arange(100000) % 2).half()
    assert power_variogram(trajectory, 1) == 0.5


def test_small_increment_gradients_do_not_underflow():
    # Scaled by the largest increment, (1e-10)**6 underflows float32.
    values = torch.tensor([0.0, 1e-5, 0.0, 1e5], requires_grad=True)
    (actual,) = torch.autograd.grad(power_variogram(values, 1, 6.0), values)
    reference = (values[1:] - values[:-1]).double().abs().pow(6).sum() / 6
    (expected,) = torch.autograd.grad(reference, values)
    assert torch.allclose(actual[:2].double(), expected[:2].double(), rtol=1e-5, atol=0)


def test_singleton_tensor_power_keeps_shapes():
    assert power_variogram(torch.arange(5.0), [1, 2], power=torch.tensor([[2.0]])).shape == (2,)
    assert variogram_dimension([1, 2], torch.tensor([0.5, 2.0]), power=torch.tensor([2.0])).shape == ()


def test_power_validation_edge_cases():
    power = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    power_variogram(torch.arange(6.0, dtype=torch.float64) ** 2, 1, power=power).backward()
    assert power.grad is not None and torch.isfinite(power.grad)
    for invalid in (10**400, numpy.array([1.0, 2.0]), None):
        with pytest.raises(ValueError, match="power"):
            power_variogram(torch.arange(5.0), 1, power=invalid)
        with pytest.raises(ValueError, match="power"):
            variogram_dimension([1, 2], torch.tensor([1.0, 4.0]), power=invalid)


def test_degenerate_column_keeps_power_gradient_finite():
    power = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    variogram = torch.tensor([[0.0, 1.0], [0.0, 4.0]], dtype=torch.float64)
    dimension = variogram_dimension([1, 2], variogram, power)
    (gradient,) = torch.autograd.grad(dimension[1], power)
    assert dimension[0].isnan() and torch.isclose(gradient, torch.tensor(0.5, dtype=torch.float64))
