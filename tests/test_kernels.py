"""Tests for torchnlts.neighbors.kernels.trajectory_kernel.

No Julia equivalent exists, so this validates against:
  1. The defining paper formulas, hand-derived: Judd (1992) eq. for the
     Gaussian correlation-sum kernel, Sugihara et al. (2012) eq. for the
     CCM exponential weight.
  2. A gradcheck (mandatory for the "smooth" gradient class).
"""

import math

import pytest
import torch

from torchnlts.neighbors.kernels import trajectory_kernel


def test_trajectory_kernel_gaussian_hand_derived():
    # Judd (1992): exp(-d^2 / (2 h^2)).
    d = torch.tensor([0.0, 1.0, 2.0])
    h = 2.0
    w = trajectory_kernel(d, h, kind="gaussian")
    expected = torch.tensor([1.0, math.exp(-1 / 8), math.exp(-4 / 8)])
    assert torch.allclose(w, expected, atol=1e-10)


def test_trajectory_kernel_exponential_hand_derived():
    # Sugihara et al. (2012): u_i = exp(-d_i / d_1), with d_1 the nearest-neighbor distance as a per-row temperature.
    d = torch.tensor([[1.0, 2.0, 4.0]])
    d1 = d[:, :1]
    w = trajectory_kernel(d, d1, kind="exponential")
    expected = torch.tensor([[math.exp(-1.0), math.exp(-2.0), math.exp(-4.0)]])
    assert torch.allclose(w, expected, atol=1e-10)


def test_trajectory_kernel_range_and_monotonicity():
    d = torch.linspace(0, 10, 50)
    for kind in ("gaussian", "exponential"):
        w = trajectory_kernel(d, temperature=1.5, kind=kind)
        assert torch.all((w > 0) & (w <= 1.0))
        assert torch.all(w[:-1] >= w[1:])
        assert torch.isclose(w[0], torch.tensor(1.0))


def test_trajectory_kernel_invalid_kind():
    with pytest.raises(ValueError):
        trajectory_kernel(torch.tensor([1.0]), 1.0, kind="bogus")


def test_trajectory_kernel_inf_distance_no_nan_gradient():
    # Regression: inf distances (Theiler/knn padding) must give weight exactly 0 and a finite temperature gradient.
    d = torch.tensor([1.0, float("inf")])
    for kind in ("gaussian", "exponential"):
        t = torch.tensor(2.0, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w[1] == 0.0
        w.sum().backward()
        assert torch.isfinite(t.grad).all(), (kind, t.grad)


def test_trajectory_kernel_gaussian_large_scale_no_overflow():
    # Regression: squaring distance and temperature separately overflows float32; d == t must always give weight 1.
    d = torch.tensor(1e20, dtype=torch.float32)
    t = torch.tensor(1e20, dtype=torch.float32)
    w = trajectory_kernel(d, t, kind="gaussian")
    assert torch.isfinite(w) and w > 0


def test_trajectory_kernel_tiny_temperature_no_nan_gradient():
    # Regression: -distance / temperature**2 overflows for a near-zero temperature, and a zero upstream gradient then
    # gives NaN.
    d = torch.tensor(1.0)
    for kind in ("gaussian", "exponential"):
        t = torch.tensor(1e-20, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w == 0.0
        w.backward()
        assert torch.isfinite(t.grad), (kind, t.grad)


def test_trajectory_kernel_small_scale_ratio_preserved():
    # Regression: an absolute epsilon floor on temperature corrupted small-scale inputs whose ratio is normal.
    d = torch.tensor(1e-8, dtype=torch.float32)
    t = torch.tensor(1e-8, dtype=torch.float32, requires_grad=True)
    gaussian = trajectory_kernel(d, t, kind="gaussian")
    exponential = trajectory_kernel(d, t, kind="exponential")
    assert torch.isclose(gaussian, torch.tensor(math.exp(-0.5)), atol=1e-4)
    assert torch.isclose(exponential, torch.tensor(math.exp(-1.0)), atol=1e-4)
    gaussian.backward()
    assert t.grad != 0


def test_trajectory_kernel_huge_distance_no_nan_gradient():
    # Regression: distance=1e38, temperature=0.1: the backward overflows unless temperature is floored relative to
    # distance.
    for kind in ("gaussian", "exponential"):
        d = torch.tensor(1e38, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w == 0.0
        w.backward()
        assert torch.isfinite(t.grad), (kind, t.grad)
        assert torch.isfinite(d.grad), (kind, d.grad)


def test_trajectory_kernel_float32_subnormal_scale_known_ceiling():
    # Documented limitation (kernels.py): near float32's floor (~1.2e-38) temperature**2 underflows to 0 and the
    # gradient blows up; float64 is the upgrade path.
    d_val, t_val = 2e-37, 2e-38
    d32 = torch.tensor(d_val, dtype=torch.float32)
    t32 = torch.tensor(t_val, dtype=torch.float32, requires_grad=True)
    trajectory_kernel(d32, t32, kind="gaussian").backward()
    assert not torch.isfinite(t32.grad)

    d64 = torch.tensor(d_val, dtype=torch.float64)
    t64 = torch.tensor(t_val, dtype=torch.float64, requires_grad=True)
    trajectory_kernel(d64, t64, kind="gaussian").backward()
    assert torch.isfinite(t64.grad) and t64.grad > 0


def test_trajectory_kernel_gradcheck():
    d = torch.rand(6, dtype=torch.float64, requires_grad=True) + 0.1
    t = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    for kind in ("gaussian", "exponential"):
        assert torch.autograd.gradcheck(lambda d, t: trajectory_kernel(d, t, kind=kind), (d, t))


@pytest.mark.parametrize("kind", ["gaussian", "exponential"])
@pytest.mark.parametrize(
    "temperature",
    [
        0,
        0.0,
        -1.0,
        float("nan"),
        torch.tensor(0.0),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("nan")]),
        1j,
        torch.tensor(1.0 + 0j),
        "1.0",
        None,
        [1.0],
    ],
)
def test_trajectory_kernel_rejects_invalid_temperature(kind, temperature):
    # Regression: a temperature <= 0 gave NaN weights at distance 0.
    with pytest.raises(ValueError, match="temperature"):
        trajectory_kernel(torch.tensor([0.0, 1.0]), temperature, kind=kind)


def test_trajectory_kernel_accepts_infinite_and_integer_temperature():
    distance = torch.tensor([0.0, 1.0, 2.0])
    assert torch.equal(trajectory_kernel(distance, float("inf")), torch.ones(3))
    assert torch.equal(trajectory_kernel(distance, 2), trajectory_kernel(distance, 2.0))
    assert torch.equal(trajectory_kernel(distance, torch.tensor(2)), trajectory_kernel(distance, 2.0))


def test_trajectory_kernel_temperature_check_under_torch_func():
    distance = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    temperatures = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    mapped = torch.vmap(lambda temperature: trajectory_kernel(distance, temperature))(temperatures)
    assert torch.equal(mapped, torch.stack([trajectory_kernel(distance, temperature) for temperature in temperatures]))

    def total(temperature):
        return trajectory_kernel(distance, temperature).sum()

    gradient = torch.vmap(torch.func.grad(total))(temperatures)
    expected = []
    for value in temperatures:
        leaf = value.clone().requires_grad_()
        expected.append(torch.autograd.grad(total(leaf), leaf)[0])
    assert torch.allclose(gradient, torch.stack(expected))
    # A concrete invalid temperature is still rejected inside vmap.
    with pytest.raises(ValueError, match="temperature"):
        torch.vmap(lambda row: trajectory_kernel(row, 0.0))(distance.reshape(3, 1))


def test_trajectory_kernel_weights_lie_in_the_closed_unit_interval():
    # Non-finite distances and weights below the dtype's range are exactly 0, not positive.
    distance = torch.tensor([0.0, 1.0, 200.0, float("inf"), float("nan")])
    for kind in ("gaussian", "exponential"):
        weight = trajectory_kernel(distance, 0.5, kind=kind)
        assert weight[0] == 1.0 and 0 < weight[1] < 1
        assert weight[2:].tolist() == [0.0, 0.0, 0.0], kind
