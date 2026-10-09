"""Tests for torchnlts.scaling.

Dual ground truth:
  1. Hand-derived piecewise-linear curves.
  2. Julia: FractalDimensions.jl ``linear_regions``, ``linear_region`` and
     ``slopefit(x, y, LargestLinearRegion())`` (tol = 0.25, dxi = 1) on the
     curves below; the expected values are copied from that package's
     output (1-based inclusive ranges converted to 0-based ``[start, stop)``).
"""

import functools

import numpy
import pytest
import torch
from dysts.flows import Lorenz

from torchnlts._shape import _ols_slope
from torchnlts.dimension.correlation import correlation_dimension, correlation_sum
from torchnlts.dimension.generalized import generalized_dimension
from torchnlts.dimension.higuchi import higuchi_dimension
from torchnlts.dimension.variogram import variogram_dimension
from torchnlts.fluctuations.dfa import dfa_exponent
from torchnlts.lyapunov.rosenstein import rosenstein_lyapunov
from torchnlts.scaling import largest_linear_region, linear_regions, scaling_slope

DOUBLE = torch.float64


def _curve(sweep, values):
    return torch.tensor(sweep, dtype=DOUBLE), torch.tensor(values, dtype=DOUBLE)


def _regions(labels):
    """``[(start, stop)]`` point ranges from segment labels ``[R - 1]``."""
    ranges = []
    for label in range(int(labels.max()) + 1):
        segments = (labels == label).nonzero().flatten()
        ranges.append((int(segments[0]), int(segments[-1]) + 2))
    return ranges


@functools.cache
def _lorenz_curve():
    state = torch.as_tensor(Lorenz().make_trajectory(3000)).to(DOUBLE)
    state = (state - state.mean(dim=0)) / state.std(dim=0)
    radius = torch.logspace(-1.8, 0.2, 20, dtype=DOUBLE)
    return radius, correlation_sum(state, radius, theiler_window=10)


CURVES = {
    "piecewise": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5]),
    "three_slopes": (list(range(11)), [0, 2, 4, 6, 7, 8, 9, 9.2, 9.4, 9.6, 9.8]),
    "wiggly": (
        [0, 0.5, 1.1, 1.5, 2.2, 3, 3.4, 4, 4.9, 5.5, 6, 7],
        [0.1, 0.45, 0.9, 1.1, 1.8, 2.2, 2.3, 2.9, 3.1, 3.2, 4.0, 4.4],
    ),
    "decreasing": (list(range(8)), [3.4, 3.3, 3.2, 3.1, 3, 2, 1, 0]),
    "tie": (list(range(9)), [0, 1, 2, 3, 4, 6, 8, 10, 12]),
}
SATURATING = (
    list(range(16)),
    [0, 0.004, 0.008, 1, 2, 3.1, 4.1, 5, 5.003, 5.006, 5.009, 5.012, 5.015, 5.018, 5.021, 5.024],
)

# name -> (regions, largest region, slope), Julia output with 1-based inclusive ranges.
JULIA = {
    "piecewise": ([(1, 5), (5, 10)], (5, 10), 0.5),
    "three_slopes": ([(1, 4), (4, 7), (7, 11)], (7, 11), 0.20000000000000018),
    "wiggly": (
        [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 11), (11, 12)],
        (1, 3),
        0.7280219780219779,
    ),
    "decreasing": ([(1, 5), (5, 8)], (1, 5), -0.09999999999999995),
    "tie": ([(1, 5), (5, 9)], (1, 5), 1.0),
}


def _zero_based(region):
    return region[0] - 1, region[1]


@pytest.mark.parametrize("name", list(JULIA))
def test_matches_julia_fractaldimensions(name):
    sweep, values = _curve(*CURVES[name])
    regions, largest, slope = JULIA[name]
    assert _regions(linear_regions(sweep, values)) == [_zero_based(region) for region in regions]
    start, stop = largest_linear_region(sweep, values)
    assert (int(start), int(stop)) == _zero_based(largest)
    assert scaling_slope(sweep, values).item() == pytest.approx(slope, abs=1e-12)


def test_matches_julia_on_lorenz_correlation_sum():
    radius, correlation = _lorenz_curve()
    sweep, values = radius.log(), correlation.log()
    # Julia: regions (1,3), (3,18), (18,20); largest (3,18), slope 2.093878809611943.
    assert _regions(linear_regions(sweep, values)) == [(0, 3), (2, 18), (17, 20)]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (2, 18)
    assert scaling_slope(sweep, values).item() == pytest.approx(2.093878809611943, abs=1e-10)


def test_saturation_matches_julia():
    sweep, values = _curve(*SATURATING)
    # Julia regions (1,3), (3,8), (8,16). Without saturation handling the
    # flat tail wins (slope 0.003); with it the rising part wins.
    assert _regions(linear_regions(sweep, values)) == [(0, 3), (2, 8), (7, 16)]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (7, 16)
    assert scaling_slope(sweep, values).item() == pytest.approx(0.002999999999999966, abs=1e-12)
    assert tuple(int(item) for item in largest_linear_region(sweep, values, saturation=0.01)) == (2, 8)
    assert scaling_slope(sweep, values, saturation=0.01).item() == pytest.approx(1.0102857142857142, abs=1e-12)
    labels = linear_regions(sweep, values, saturation=0.01)
    assert labels[:2].tolist() == [-1, -1] and labels[-8:].tolist() == [-1] * 8


def test_hand_derived_two_region_curve():
    # Slope 1 on x = 0..3 (4 points), slope 0.5 on x = 3..8 (6 points, sharing x = 3):
    # the second region is larger and its OLS slope is exactly 0.5.
    sweep, values = _curve(list(range(9)), [0, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5])
    assert linear_regions(sweep, values).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    start, stop = largest_linear_region(sweep, values)
    assert (int(start), int(stop)) == (3, 9)
    assert scaling_slope(sweep, values).item() == pytest.approx(0.5, abs=1e-12)


def test_tolerance_controls_merging():
    sweep, values = _curve(list(range(6)), [0, 1, 2, 3.2, 4.4, 5.6])
    assert scaling_slope(sweep, values, tolerance=0.25).item() == pytest.approx(
        1.1257142857142857, abs=1e-12
    )  # OLS over all 6 points
    # Slopes 1,1,1.2,1.2,1.2: tolerance 0.1 splits at the change (3 vs 4 points -> 1.2 region).
    assert _regions(linear_regions(sweep, values, tolerance=0.1)) == [(0, 3), (2, 6)]
    assert scaling_slope(sweep, values, tolerance=0.1).item() == pytest.approx(1.2, abs=1e-12)


def test_batched_rows_have_different_regions():
    sweep, values = _curve(*CURVES["piecewise"])
    other = values.flip(0)
    batch = torch.stack([values, other], dim=1)
    expected = torch.stack([scaling_slope(sweep, values), scaling_slope(sweep, other)])
    torch.testing.assert_close(scaling_slope(sweep, batch), expected)
    starts, stops = largest_linear_region(sweep, batch)
    assert starts.shape == stops.shape == (2,)
    assert starts.tolist() == [4, 0] and stops.tolist() == [10, 6]


def test_shape_conventions():
    sweep, values = _curve(*CURVES["piecewise"])
    assert scaling_slope(sweep, values).shape == ()
    assert scaling_slope(sweep, values.unsqueeze(1)).shape == (1,)
    assert linear_regions(sweep, values).shape == (9,)
    assert linear_regions(sweep, values.repeat(3, 1).T).shape == (9, 3)
    assert largest_linear_region(sweep, values)[0].shape == ()


@pytest.mark.parametrize(
    "estimator",
    ["correlation", "dfa", "higuchi", "generalized", "rosenstein", "variogram"],
)
def test_matches_estimator_on_detected_region(estimator):
    torch.manual_seed(0)
    steps = torch.arange(1, 13, dtype=DOUBLE)
    # Two power-law regimes with a bend, so the detected region is a proper subset.
    curve = torch.where(steps <= 5, steps, 5.0 * (steps / 5.0) ** 2.0)
    curve = curve * (1 + 0.01 * torch.randn(12, dtype=DOUBLE))
    if estimator == "correlation":
        sweep, values = steps.log(), curve.log()
        reference = lambda s: correlation_dimension(steps[s], curve[s])
    elif estimator == "dfa":
        sweep, values = steps.log(), curve.log()
        reference = lambda s: dfa_exponent(steps[s], curve[s])
    elif estimator == "higuchi":
        sweep, values = -steps.log(), curve.log()
        reference = lambda s: higuchi_dimension(steps[s], curve[s])
    elif estimator == "generalized":
        sweep, values = -steps.log(), curve
        reference = lambda s: generalized_dimension(steps[s], curve[s])
    elif estimator == "rosenstein":
        sweep, values = steps, curve
        reference = lambda s: rosenstein_lyapunov(steps[s], curve[s])
    else:
        sweep, values = steps.log(), curve.log()
        reference = lambda s: variogram_dimension(steps[s], curve[s], power=2.0)
    start, stop = largest_linear_region(sweep, values)
    assert 2 <= int(stop) - int(start) < 12
    window = slice(int(start), int(stop))
    slope = scaling_slope(sweep, values)
    expected = reference(window)
    if estimator == "variogram":
        slope = 2.0 - slope / 2.0  # variogram_dimension returns 2 - slope / power
    torch.testing.assert_close(slope, expected, rtol=1e-10, atol=1e-10)


def test_correlation_dimension_on_lorenz_region():
    radius, correlation = _lorenz_curve()
    start, stop = largest_linear_region(radius.log(), correlation.log())
    window = slice(int(start), int(stop))
    torch.testing.assert_close(
        scaling_slope(radius.log(), correlation.log()),
        correlation_dimension(radius[window], correlation[window]),
    )


def test_nan_row_is_isolated():
    sweep, values = _curve(*CURVES["piecewise"])
    batch = torch.stack([values, values, values], dim=1).clone()
    batch[3, 1] = float("nan")
    batch[5, 2] = float("inf")
    batch.requires_grad_(True)
    slope = scaling_slope(sweep, batch)
    assert torch.isnan(slope[1]) and torch.isnan(slope[2]) and slope[0].item() == pytest.approx(0.5)
    slope[0].backward()
    assert torch.isfinite(batch.grad).all()
    assert (batch.grad[:, 1:] == 0).all()
    labels = linear_regions(sweep, batch.detach())
    assert (labels[:, 1:] == -1).all() and (labels[:, 0] >= 0).all()
    start, stop = largest_linear_region(sweep, batch.detach())
    assert start.tolist()[1:] == [-1, -1] and stop.tolist()[1:] == [-1, -1]


def test_nan_row_gradient_of_other_rows_matches_clean_call():
    sweep, values = _curve(*CURVES["three_slopes"])
    clean = values.clone().requires_grad_(True)
    scaling_slope(sweep, clean).backward()
    batch = torch.stack([values, values], dim=1).clone()
    batch[2, 1] = float("nan")
    batch.requires_grad_(True)
    scaling_slope(sweep, batch)[0].backward()
    torch.testing.assert_close(batch.grad[:, 0], clean.grad)


def test_nonfinite_sweep_gives_nan():
    sweep, values = _curve(*CURVES["piecewise"])
    sweep[2] = float("nan")
    assert torch.isnan(scaling_slope(sweep, values))


def test_gradcheck_fixed_region():
    sweep, values = _curve(*CURVES["three_slopes"])
    values = values.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda curve: scaling_slope(sweep, curve), (values,), eps=1e-7, atol=1e-6)
    batch = torch.stack([values.detach(), values.detach().flip(0)], dim=1).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda curve: scaling_slope(sweep, curve), (batch,), eps=1e-7, atol=1e-6)


def test_gradient_equals_ols_gradient_over_region():
    sweep, values = _curve(*CURVES["three_slopes"])
    values = values.clone().requires_grad_(True)
    scaling_slope(sweep, values).backward()
    # Region is points 6..10: gradient is (x - mean x) / sum((x - mean x)^2) there, zero elsewhere.
    centered = sweep[6:] - sweep[6:].mean()
    expected = torch.zeros(11, dtype=DOUBLE)
    expected[6:] = centered / (centered**2).sum()
    torch.testing.assert_close(values.grad, expected)


def test_half_precision_computes_in_float32_and_casts_back():
    sweep, values = _curve(*CURVES["wiggly"])
    half_result = scaling_slope(sweep.to(torch.float16), values.to(torch.float16))
    assert half_result.dtype == torch.float16
    # Same as computing the half-rounded inputs in float32 and casting back.
    reference = scaling_slope(sweep.to(torch.float16).float(), values.to(torch.float16).float()).to(torch.float16)
    assert half_result.item() == reference.item()
    assert scaling_slope(sweep, values.float()).dtype == torch.float32
    assert scaling_slope(sweep, torch.arange(12)).dtype == torch.get_default_dtype()


def test_half_labels_use_float32_slopes():
    # Slopes 1000 and 1001 are one half-precision step apart (1000 / 1001 both
    # representable only up to spacing 0.5 near 1024): tolerance 1e-4 separates
    # them in float32 but not if the segment slopes were rounded to half.
    sweep = torch.arange(4, dtype=torch.float16)
    values = torch.tensor([0.0, 1000.0, 2001.0, 3002.0], dtype=torch.float16)
    reference = linear_regions(sweep.float(), values.float(), tolerance=1e-4)
    assert linear_regions(sweep, values, tolerance=1e-4).tolist() == reference.tolist()


def test_gradcheck_with_respect_to_sweep():
    sweep, values = _curve(*CURVES["three_slopes"])
    sweep = (sweep + 0.1 * sweep**1.5).requires_grad_(True)
    values = values.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(scaling_slope, (sweep, values), eps=1e-7, atol=1e-6)


def test_float32_close_sweep_points_stay_distinct():
    # Regression: a two-point region far from zero needs translation by its first point.
    sweep = torch.tensor([2.0**24, 2.0**24 + 2], dtype=torch.float32)
    values = torch.tensor([0.0, 2.0], dtype=torch.float32)
    assert scaling_slope(sweep, values).item() == _ols_slope(sweep, values).item() == 1.0


def test_constant_huge_values_give_zero_slope():
    # Regression: constant values must stay exactly constant after centering.
    sweep = torch.arange(10, dtype=torch.float32) * 0.1
    values = torch.full((10,), 1e20)
    assert scaling_slope(sweep, values).item() == _ols_slope(sweep, values).item() == 0.0


def test_repeated_huge_sweep_is_nan():
    sweep = torch.full((7,), 1e20)
    values = torch.arange(7, dtype=torch.float32)
    assert torch.isnan(scaling_slope(sweep, values)) and torch.isnan(_ols_slope(sweep, values))


def test_large_magnitude_slope_does_not_overflow():
    # Regression: centered products reach 6e38 (> float32 max) without scaling.
    sweep = 10 * torch.arange(10, dtype=torch.float32)
    assert scaling_slope(sweep, 3e35 * sweep).item() == pytest.approx(3e35, rel=1e-5)


def test_zero_width_segments_match_julia():
    # Julia: x = [0, 0, 1] gives regions 1:1, 1:2, 2:3, largest 1:2, slope NaN.
    sweep, values = _curve([0, 0, 1], [0, 1, 2])
    assert linear_regions(sweep, values).tolist() == [0, 1]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (0, 2)
    assert torch.isnan(scaling_slope(sweep, values))
    # Julia: x = [0 1 1 2 3 4] gives regions (1,2), (2,3), (3,6); largest (3,6), slope 1.
    sweep, values = _curve([0, 1, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5])
    assert linear_regions(sweep, values).tolist() == [0, 1, 2, 2, 2]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (2, 6)
    assert scaling_slope(sweep, values).item() == pytest.approx(1.0)
    # Julia: x = [0 1 2 3 3 4 5 6]: regions (1,4), (4,5), (5,8); largest (1,4), slope 1 (first of a tie).
    sweep, values = _curve([0, 1, 2, 3, 3, 4, 5, 6], list(range(8)))
    assert linear_regions(sweep, values).tolist() == [0, 0, 0, 1, 2, 2, 2]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (0, 4)


@pytest.mark.parametrize("bad", ["bad", None, [1.0], 1j])
def test_non_numeric_arguments_raise_value_error(bad):
    sweep, values = _curve(*CURVES["piecewise"])
    for function in (linear_regions, largest_linear_region, scaling_slope):
        with pytest.raises(ValueError):
            function(sweep, values, tolerance=bad)
        if bad is not None:
            with pytest.raises(ValueError):
                function(sweep, values, saturation=bad)


def test_vmap():
    sweep, values = _curve(*CURVES["piecewise"])
    batch = torch.stack([values, values.flip(0), values * 2], dim=1)
    mapped = torch.vmap(lambda column: scaling_slope(sweep, column), in_dims=1)(batch)
    torch.testing.assert_close(mapped, scaling_slope(sweep, batch))


@pytest.mark.parametrize(
    "kwargs",
    [{"tolerance": 0.0}, {"tolerance": -1.0}, {"tolerance": float("nan")}, {"saturation": -0.1}],
)
def test_invalid_arguments(kwargs):
    sweep, values = _curve(*CURVES["piecewise"])
    for function in (linear_regions, largest_linear_region, scaling_slope):
        with pytest.raises(ValueError):
            function(sweep, values, **kwargs)


def test_invalid_shapes():
    sweep, values = _curve(*CURVES["piecewise"])
    for function in (linear_regions, largest_linear_region, scaling_slope):
        with pytest.raises(ValueError):
            function(sweep[:5], values)
        with pytest.raises(ValueError):
            function(sweep.unsqueeze(1), values)
        with pytest.raises(ValueError):
            function(sweep[:1], values[:1])
        with pytest.raises(ValueError):
            function(sweep, values.reshape(2, 5, 1))


def test_slope_tolerance_boundary_is_inclusive():
    # Slopes 3 and 4: |3 - 4| = 1 = 0.25 * max(3, 4) exactly, so one region.
    sweep, values = _curve(list(range(5)), [0, 3, 6, 10, 14])
    assert linear_regions(sweep, values).tolist() == [0, 0, 0, 0]


def test_region_compares_to_its_opening_slope():
    # Slopes 1, 1.2, 1.4, 1.6: 1.4 differs from the opening slope 1 by more
    # than 25% (0.4 > 0.35), 1.6 is within 25% of the new opening slope 1.4.
    sweep, values = _curve(list(range(5)), [0, 1, 2.2, 3.6, 5.2])
    assert linear_regions(sweep, values).tolist() == [0, 0, 1, 1]
    assert tuple(int(item) for item in largest_linear_region(sweep, values)) == (0, 3)


def test_saturation_threshold_is_exclusive():
    sweep, values = _curve(list(range(5)), [0, 1, 3, 5, 7])
    assert linear_regions(sweep, values, saturation=1.0).tolist() == [-1, 0, 0, 0]


def test_saturation_first_active_segment_opens_region():
    # Leading slope 0.008 is ignored; 0.0102 is within 25% of it but 0.0126 is
    # not, so the first active segment must be the comparison slope.
    sweep, values = _curve([0, 1, 2, 3], [0, 0.008, 0.0182, 0.0308])
    assert linear_regions(sweep, values, saturation=0.01).tolist() == [-1, 0, 0]


def test_all_flat_curve_keeps_every_segment():
    sweep, values = _curve(list(range(5)), [2.0] * 5)
    assert linear_regions(sweep, values, saturation=0.01).tolist() == [0, 0, 0, 0]
    assert scaling_slope(sweep, values, saturation=0.01).item() == 0.0


def test_overflowing_slopes_are_never_close():
    # Both segment slopes overflow float32 to inf; inf must not count as close.
    sweep = torch.tensor([0.0, 1e-10, 2e-10], dtype=torch.float32)
    values = torch.tensor([0.0, 1e38, 2e38], dtype=torch.float32)
    assert linear_regions(sweep, values).tolist() == [0, 1]


def test_complex_scalar_arguments_raise_value_error():
    series = torch.arange(3.0)
    for value in (torch.tensor(0.25 + 0j), numpy.complex128(0.25)):
        for argument in ("tolerance", "saturation"):
            with pytest.raises(ValueError):
                scaling_slope(series, series, **{argument: value})
