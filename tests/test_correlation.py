"""Tests for torchnlts.dimension.correlation.

Dual ground truth:
  1. Paper: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1 --
     checked here via a hand-derived example for the hard variant. No
     Julia package implements the smooth kernel variant, so the soft
     variant instead gets a hand-derived analytic limit (temperature -> 0
     recovers the epsilon=0 hard correlation sum) plus a gradient check
     backing its "smooth" gradient-class claim.
  2. Julia: FractalDimensions.jl's `correlationsum` (hard variant only).
"""

import subprocess

import torch

from torchnlts.dimension.correlation import correlation_dimension, correlation_sum, soft_correlation_sum

JULIA_PROJECT = "juliaenv"


def test_correlation_sum_hand_derived():
    # 3 points [0, 1, 0.3]: off-diagonal distances 1, 0.3, 0.7 (6 ordered
    # pairs after excluding the diagonal with the default theiler_window=0).
    # r=0.5 keeps only the two pairs at distance 0.3 -> C = 2/6.
    # r=1.5 keeps every pair -> C = 1.0.
    trajectory = torch.tensor([0.0, 1.0, 0.3])
    assert torch.isclose(correlation_sum(trajectory, radius=0.5), torch.tensor(2 / 6))
    assert torch.isclose(correlation_sum(trajectory, radius=1.5), torch.tensor(1.0))


def test_correlation_sum_radius_sweep():
    trajectory = torch.tensor([0.0, 1.0, 0.3])
    result = correlation_sum(trajectory, radius=torch.tensor([0.5, 1.5]))
    assert torch.allclose(result, torch.tensor([2 / 6, 1.0]))


def test_correlation_sum_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 20, 2, dtype=torch.float64)
    scalar_radius = correlation_sum(trajectory, radius=0.5)
    assert scalar_radius.shape == (4,)
    swept = correlation_sum(trajectory, radius=torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64))
    assert swept.shape == (3, 4)
    assert torch.allclose(swept[1], scalar_radius)


def test_correlation_sum_monotone_in_radius():
    torch.manual_seed(1)
    trajectory = torch.randn(30, dtype=torch.float64)
    radii = torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float64)
    result = correlation_sum(trajectory, radius=radii)
    assert torch.all(result[1:] >= result[:-1])


def test_correlation_sum_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0])
    result = correlation_sum(trajectory, radius=0.5, theiler_window=1)
    assert torch.isnan(result)


def test_correlation_sum_matches_fractaldimensions_jl():
    torch.manual_seed(2)
    trajectory = torch.randn(30, dtype=torch.float64)
    radii = [0.3, 0.75, 1.5]

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    for theiler_window in (0, 2):
        julia_code = f"""
        using FractalDimensions
        x = [{values}]
        X = StateSpaceSet(reshape(x, :, 1))
        cs = correlationsum(X, {radii}; w={theiler_window})
        println(join(cs, ","))
        """
        result = subprocess.run(
            ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        julia_cs = torch.tensor([float(v) for v in result.stdout.strip().split(",")], dtype=torch.float64)
        torch_cs = correlation_sum(
            trajectory, radius=torch.tensor(radii, dtype=torch.float64), theiler_window=theiler_window
        )
        assert torch.allclose(torch_cs, julia_cs, atol=1e-9), (theiler_window, torch_cs, julia_cs)


def test_soft_correlation_sum_low_temperature_limit_matches_radius_zero_hard():
    trajectory = torch.tensor([0.0, 1.0, 0.3, 1.0])
    soft = soft_correlation_sum(trajectory, temperature=1e-6)
    hard = correlation_sum(trajectory, radius=0.0)
    assert torch.allclose(soft, hard, atol=1e-4)


def test_soft_correlation_sum_batched_and_sweep_shapes():
    torch.manual_seed(3)
    trajectory = torch.randn(4, 10, 2, dtype=torch.float64)
    temperatures = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    swept = soft_correlation_sum(trajectory, temperature=temperatures)
    assert swept.shape == (3, 4)
    for i, temperature in enumerate(temperatures):
        scalar = soft_correlation_sum(trajectory, temperature=temperature)
        assert scalar.shape == (4,)
        assert torch.allclose(swept[i], scalar)


def test_correlation_sum_batched_and_sweep_shapes_every_row():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 20, 2, dtype=torch.float64)
    radii = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    swept = correlation_sum(trajectory, radius=radii)
    for i, radius in enumerate(radii):
        assert torch.allclose(swept[i], correlation_sum(trajectory, radius=radius))


def test_soft_correlation_sum_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0])
    result = soft_correlation_sum(trajectory, temperature=0.5, theiler_window=1)
    assert torch.isnan(result)


def test_soft_correlation_sum_batched_and_swept_theiler_window_excludes_everything_is_nan():
    trajectory = torch.randn(3, 2, 2, dtype=torch.float64)
    result = soft_correlation_sum(
        trajectory, temperature=torch.tensor([0.3, 0.5], dtype=torch.float64), theiler_window=1
    )
    assert result.shape == (2, 3)
    assert torch.all(torch.isnan(result))


def test_correlation_sum_batched_and_swept_theiler_window_excludes_everything_is_nan():
    trajectory = torch.randn(3, 2, 2, dtype=torch.float64)
    result = correlation_sum(trajectory, radius=torch.tensor([0.3, 0.5], dtype=torch.float64), theiler_window=1)
    assert result.shape == (2, 3)
    assert torch.all(torch.isnan(result))


def test_soft_correlation_sum_gradcheck():
    torch.manual_seed(4)
    trajectory = torch.randn(6, dtype=torch.float64, requires_grad=True)

    def cs(x):
        return soft_correlation_sum(x, temperature=0.5)

    assert torch.autograd.gradcheck(cs, (trajectory,))


def test_soft_correlation_sum_gradcheck_exponential():
    torch.manual_seed(5)
    trajectory = torch.randn(6, dtype=torch.float64, requires_grad=True)

    def cs(x):
        return soft_correlation_sum(x, temperature=0.5, kind="exponential")

    assert torch.autograd.gradcheck(cs, (trajectory,))


def test_soft_correlation_sum_gradcheck_temperature():
    torch.manual_seed(6)
    trajectory = torch.randn(6, dtype=torch.float64)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def cs(t):
        return soft_correlation_sum(trajectory, temperature=t)

    assert torch.autograd.gradcheck(cs, (temperature,))


def test_correlation_dimension_hand_derived_exact_power_law():
    # C(r) = r^2 exactly -> log C = 2 log r -> slope 2, exact regression fit.
    radius = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    correlation_sum_values = radius**2
    assert torch.isclose(correlation_dimension(radius, correlation_sum_values), torch.tensor(2.0, dtype=torch.float64))


def test_correlation_dimension_two_points_is_exact_slope():
    radius = torch.tensor([1.0, 2.0], dtype=torch.float64)
    correlation_sum_values = torch.tensor([0.1, 0.4], dtype=torch.float64)
    expected = (torch.log(correlation_sum_values[1]) - torch.log(correlation_sum_values[0])) / (
        torch.log(radius[1]) - torch.log(radius[0])
    )
    assert torch.isclose(correlation_dimension(radius, correlation_sum_values), expected)


def test_correlation_dimension_batched_shape():
    torch.manual_seed(7)
    trajectory = torch.randn(4, 30, 2, dtype=torch.float64)
    radius = torch.tensor([0.5, 0.75, 1.0, 1.5], dtype=torch.float64)
    correlation_sum_values = correlation_sum(trajectory, radius=radius)
    dimension = correlation_dimension(radius, correlation_sum_values)
    assert dimension.shape == (4,)
    for i in range(4):
        expected = correlation_dimension(radius, correlation_sum(trajectory[i], radius=radius))
        assert torch.isclose(dimension[i], expected)


def test_correlation_dimension_single_radius_is_nan_not_crash():
    result = correlation_dimension(torch.tensor([1.0]), torch.tensor([0.5]))
    assert torch.isnan(result)


def test_correlation_dimension_identical_radii_is_nan_not_crash():
    result = correlation_dimension(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.2, 0.5, 0.3]))
    assert torch.isnan(result)


def test_correlation_dimension_zero_correlation_sum_is_nan_not_crash():
    result = correlation_dimension(torch.tensor([1.0, 2.0, 4.0]), torch.tensor([0.0, 0.2, 0.5]))
    assert torch.isnan(result)


def test_correlation_dimension_gradcheck_direct_radius_and_correlation_sum():
    # Gradcheck of the OLS formula w.r.t. both inputs, including radius (e.g. learned sweep points).
    radius = torch.tensor([0.5, 0.75, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    correlation_sum_values = torch.tensor([0.1, 0.2, 0.35, 0.6], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(correlation_dimension, (radius, correlation_sum_values))


def test_correlation_dimension_gradcheck_through_soft_correlation_sum():
    torch.manual_seed(8)
    trajectory = torch.randn(30, dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor([0.3, 0.5, 0.75, 1.0], dtype=torch.float64)

    def dimension(x):
        return correlation_dimension(temperature, soft_correlation_sum(x, temperature=temperature))

    assert torch.autograd.gradcheck(dimension, (trajectory,))


def test_correlation_dimension_matches_fractaldimensions_jl_linear_regression():
    torch.manual_seed(9)
    trajectory = torch.randn(40, dtype=torch.float64)
    radii = [0.4, 0.6, 0.9, 1.3, 1.9]

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    # ignore_saturation=false: slopefit's default can trim endpoints, which would not match our all-point OLS.
    julia_code = f"""
    using FractalDimensions
    x = [{values}]
    X = StateSpaceSet(reshape(x, :, 1))
    cs = correlationsum(X, {radii})
    slope, _, _ = slopefit(log2.({radii}), log2.(cs), LinearRegression(); ignore_saturation=false)
    println(slope)
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_dimension = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    radius = torch.tensor(radii, dtype=torch.float64)
    torch_dimension = correlation_dimension(radius, correlation_sum(trajectory, radius=radius))
    # atol=1e-9: both sides do closed-form OLS in float64; the residual is parsed-string and log-base conversion noise.
    assert torch.isclose(torch_dimension, julia_dimension, atol=1e-9)


def test_correlation_sums_nan_for_nonfinite_rows():
    # Regression: pairs with a non-finite point used to be skipped, giving a finite value for that row.
    torch.manual_seed(0)
    for bad in (float("nan"), float("inf")):
        batch = torch.randn(3, 40, 2, dtype=torch.float64)
        batch[1, 5, 0] = bad
        for function in (correlation_sum, soft_correlation_sum):
            values = function(batch, torch.tensor([0.5, 1.0], dtype=torch.float64))
            assert torch.isnan(values[:, 1]).all()
            assert torch.equal(values[:, 0], function(batch[0], torch.tensor([0.5, 1.0], dtype=torch.float64)))
            assert torch.isnan(function(batch[1], 0.5))


def test_soft_correlation_sum_nonfinite_row_gradient_isolated():
    torch.manual_seed(1)
    batch = torch.randn(2, 30, 1, dtype=torch.float64, requires_grad=True)
    poisoned = torch.zeros(2, 30, 1, dtype=torch.float64)
    poisoned[1, 4] = float("nan")
    values = soft_correlation_sum(batch + poisoned, 0.5)
    (gradient,) = torch.autograd.grad(values[0], batch)
    assert torch.isnan(values[1]) and torch.isfinite(gradient).all()
    (expected,) = torch.autograd.grad(soft_correlation_sum(batch[0], 0.5), batch)
    assert torch.equal(gradient, expected)
