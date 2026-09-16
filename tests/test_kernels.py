"""Tests for torchnlts.neighbors.kernels.trajectory_kernel.

No Julia equivalent exists (see kernels.py docstring), so this
validates against:
  1. The defining paper formulas, hand-derived: Judd (1992) eq. for the
     Gaussian correlation-sum kernel, Sugihara et al. (2012) eq. for the
     CCM exponential weight.
  2. A gradcheck (mandatory for the "smooth" gradient class).
"""

import math

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
    # Sugihara et al. (2012): u_i = exp(-d_i / d_1); with d_1 (the
    # nearest-neighbor distance) as a per-row temperature.
    d = torch.tensor([[1.0, 2.0, 4.0]])  # d_1 = 1.0 is the first entry
    d1 = d[:, :1]
    w = trajectory_kernel(d, d1, kind="exponential")
    expected = torch.tensor([[math.exp(-1.0), math.exp(-2.0), math.exp(-4.0)]])
    assert torch.allclose(w, expected, atol=1e-10)


def test_trajectory_kernel_range_and_monotonicity():
    d = torch.linspace(0, 10, 50)
    for kind in ("gaussian", "exponential"):
        w = trajectory_kernel(d, temperature=1.5, kind=kind)
        assert torch.all((w > 0) & (w <= 1.0))
        assert torch.all(w[:-1] >= w[1:])  # monotonically non-increasing in distance
        assert torch.isclose(w[0], torch.tensor(1.0))  # d=0 -> full weight


def test_trajectory_kernel_invalid_kind():
    try:
        trajectory_kernel(torch.tensor([1.0]), 1.0, kind="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


def test_trajectory_kernel_inf_distance_no_nan_gradient():
    # Regression: distances from apply_theiler_window/knn_search padding
    # are often inf; the forward weight must be exactly 0 and the
    # temperature gradient must be finite, not NaN.
    d = torch.tensor([1.0, float("inf")])
    for kind in ("gaussian", "exponential"):
        t = torch.tensor(2.0, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w[1] == 0.0
        w.sum().backward()
        assert torch.isfinite(t.grad).all(), (kind, t.grad)


def test_trajectory_kernel_gaussian_large_scale_no_overflow():
    # Regression: squaring distance and temperature independently
    # overflows float32 well before their ratio would; d == t must
    # always give weight 1 regardless of their shared magnitude.
    d = torch.tensor(1e20, dtype=torch.float32)
    t = torch.tensor(1e20, dtype=torch.float32)
    w = trajectory_kernel(d, t, kind="gaussian")
    assert torch.isfinite(w) and w > 0


def test_trajectory_kernel_tiny_temperature_no_nan_gradient():
    # Regression: distance / temperature's own backward formula
    # (-distance / temperature**2) overflows to inf for a near-zero
    # temperature; multiplying by even a zero upstream gradient then
    # gives NaN. Both distance and temperature are finite, valid inputs.
    d = torch.tensor(1.0)
    for kind in ("gaussian", "exponential"):
        t = torch.tensor(1e-20, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w == 0.0
        w.backward()
        assert torch.isfinite(t.grad), (kind, t.grad)


def test_trajectory_kernel_small_scale_ratio_preserved():
    # Regression: an earlier fix used an *absolute* epsilon floor on
    # temperature, which corrupted small-scale inputs where distance and
    # temperature are both tiny but their ratio is normal.
    d = torch.tensor(1e-8, dtype=torch.float32)
    t = torch.tensor(1e-8, dtype=torch.float32, requires_grad=True)
    gaussian = trajectory_kernel(d, t, kind="gaussian")
    exponential = trajectory_kernel(d, t, kind="exponential")
    assert torch.isclose(gaussian, torch.tensor(math.exp(-0.5)), atol=1e-4)
    assert torch.isclose(exponential, torch.tensor(math.exp(-1.0)), atol=1e-4)
    gaussian.backward()
    assert t.grad != 0


def test_trajectory_kernel_huge_distance_no_nan_gradient():
    # Regression: distance=1e38, temperature=0.1 are both finite and
    # individually unremarkable, but their ratio's backward overflows
    # (-distance / temperature**2 ~ -1e40) unless temperature is floored
    # relative to distance rather than by an absolute epsilon.
    for kind in ("gaussian", "exponential"):
        d = torch.tensor(1e38, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
        w = trajectory_kernel(d, t, kind=kind)
        assert w == 0.0
        w.backward()
        assert torch.isfinite(t.grad), (kind, t.grad)
        assert torch.isfinite(d.grad), (kind, d.grad)


def test_trajectory_kernel_float32_subnormal_scale_known_ceiling():
    # Documented ceiling (see kernels.py's "Limitation:" note): distance and
    # temperature both within ~1 order of magnitude of float32's
    # representable floor (~1.2e-38) can still blow up the temperature
    # gradient in float32, because temperature**2 underflows to exactly 0
    # independent of how the forward pass is arranged. This is not
    # something torchnlts can fix in float32; the upgrade path is float64.
    d_val, t_val = 2e-37, 2e-38
    d32 = torch.tensor(d_val, dtype=torch.float32)
    t32 = torch.tensor(t_val, dtype=torch.float32, requires_grad=True)
    trajectory_kernel(d32, t32, kind="gaussian").backward()
    assert not torch.isfinite(t32.grad)  # the documented ceiling itself

    d64 = torch.tensor(d_val, dtype=torch.float64)
    t64 = torch.tensor(t_val, dtype=torch.float64, requires_grad=True)
    trajectory_kernel(d64, t64, kind="gaussian").backward()
    assert torch.isfinite(t64.grad) and t64.grad > 0  # the upgrade path


def test_trajectory_kernel_gradcheck():
    d = torch.rand(6, dtype=torch.float64, requires_grad=True) + 0.1
    t = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    for kind in ("gaussian", "exponential"):
        assert torch.autograd.gradcheck(lambda d, t: trajectory_kernel(d, t, kind=kind), (d, t))
