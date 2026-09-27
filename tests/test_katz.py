"""Tests for torchnlts.dimension.katz.

Dual ground truth:
  1. Paper: Katz (1988), eq. (6), distances in the (time index,
     amplitude) plane -- checked via a hand-derived example.
  2. Reference implementation: antropy's ``katz_fd``, which uses
     amplitude differences only (no time-index term) -- checked via
     ``amplitude_only=True``, which reproduces it.
"""

import math

import pytest
import torch

from torchnlts.dimension.katz import katz_dimension


def test_katz_dimension_hand_derived_paper_convention():
    # x = [1, 2, 1, 3]. Points in the (index, amplitude) plane:
    # (0,1), (1,2), (2,1), (3,3). n=3 steps.
    # L = sqrt(1+(2-1)^2) + sqrt(1+(1-2)^2) + sqrt(1+(3-1)^2)
    #   = sqrt(2) + sqrt(2) + sqrt(5) = 5.06449510224598...
    # d = max distance from (0,1): to (1,2)=sqrt(1+1)=sqrt(2)=1.41421...,
    #   to (2,1)=sqrt(4+0)=2, to (3,3)=sqrt(9+4)=sqrt(13)=3.60555128...
    #   -> d = sqrt(13).
    # D = log10(3) / (log10(3) + log10(sqrt(13)/L)) = 1.4477664861880815
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    dimension = katz_dimension(trajectory)
    assert torch.isclose(dimension, torch.tensor(1.4477664861880815, dtype=torch.float64))


def test_katz_dimension_hand_derived_amplitude_only():
    # Same series, amplitude-only convention (antropy's ``katz_fd``):
    # dists = |2-1|,|1-2|,|3-1| = 1,1,2 -> L=4, a=mean(dists)=4/3.
    # d = max(|2-1|,|1-1|,|3-1|) = 2. n=3.
    # D = log10(3) / (log10(3) + log10(2/4)) = log10(3)/(log10(3)-log10(2))
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    dimension = katz_dimension(trajectory, amplitude_only=True)
    expected = math.log10(3) / (math.log10(3) + math.log10(2.0 / 4.0))
    assert torch.isclose(dimension, torch.tensor(expected, dtype=torch.float64))


def test_katz_dimension_matches_antropy_amplitude_only():
    import antropy

    torch.manual_seed(0)
    trajectory = torch.randn(200, dtype=torch.float64)
    torch_value = katz_dimension(trajectory, amplitude_only=True)
    antropy_value = antropy.katz_fd(trajectory.numpy())
    # Both compute the identical closed-form expression in float64;
    # tolerance only covers order-of-operations rounding.
    assert torch.isclose(torch_value, torch.tensor(float(antropy_value), dtype=torch.float64), atol=1e-9)


def test_katz_dimension_straight_line_is_one():
    # A straight line is a straight line in the (index, amplitude) plane
    # too: d equals the endpoint-to-start distance, which equals L
    # exactly (every step is identical), so d/L=1 and D=log10(n)/log10(n)=1
    # for any slope, any n -- true under both conventions.
    line = torch.arange(0.0, 100.0, dtype=torch.float64) * 2.5 + 1.0
    assert torch.isclose(katz_dimension(line), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(katz_dimension(line, amplitude_only=True), torch.tensor(1.0, dtype=torch.float64))


def test_katz_dimension_constant_trajectory_paper_convention_is_one():
    # Under the paper's (index, amplitude)-plane convention, a constant
    # series is still a straight (horizontal) line -- D=1, not degenerate.
    constant = torch.full((50,), 3.0, dtype=torch.float64)
    assert torch.isclose(katz_dimension(constant), torch.tensor(1.0, dtype=torch.float64))


def test_katz_dimension_constant_trajectory_amplitude_only_is_nan():
    # Amplitude-only: L=d=0 for a constant series -> 0/0, not a crash.
    constant = torch.full((50,), 3.0, dtype=torch.float64)
    assert torch.isnan(katz_dimension(constant, amplitude_only=True))


def test_katz_dimension_two_points_is_nan_not_crash():
    # n=1, and d equals the single step's own length, so d/L=1: both
    # log10 terms are exactly zero, 0/0.
    trajectory = torch.tensor([1.0, 5.0], dtype=torch.float64)
    assert torch.isnan(katz_dimension(trajectory))


def test_katz_dimension_single_point_is_nan_not_crash():
    trajectory = torch.tensor([1.0], dtype=torch.float64)
    assert torch.isnan(katz_dimension(trajectory))


def test_katz_dimension_empty_trajectory_is_nan_not_crash():
    trajectory = torch.tensor([], dtype=torch.float64)
    assert torch.isnan(katz_dimension(trajectory))


def test_katz_dimension_multivariate_raises():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    try:
        katz_dimension(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_katz_dimension_batched_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(4, 30, 1, dtype=torch.float64)
    batched = katz_dimension(trajectory)
    assert batched.shape == (4,)
    for index in range(4):
        assert torch.allclose(batched[index], katz_dimension(trajectory[index]))


def test_katz_dimension_batch_size_one_matches_unbatched():
    torch.manual_seed(2)
    trajectory = torch.randn(1, 30, 1, dtype=torch.float64)
    batched = katz_dimension(trajectory)
    unbatched = katz_dimension(trajectory[0])
    assert batched.shape == (1,)
    assert torch.allclose(batched[0], unbatched)


def test_katz_dimension_unbatched_is_scalar():
    trajectory = torch.randn(20, dtype=torch.float64)
    assert katz_dimension(trajectory).shape == ()


def test_katz_dimension_gradcheck_paper_convention():
    torch.manual_seed(3)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: katz_dimension(values), (trajectory,))


def test_katz_dimension_gradcheck_amplitude_only():
    torch.manual_seed(4)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: katz_dimension(values, amplitude_only=True), (trajectory,))


def test_katz_dimension_finite_gradient_at_farthest_point_tie():
    # Planar: from (0, 0), (3, 4) and (5, 0) are both at distance 5.
    # Amplitude only: |3 - 0| = |-3 - 0|. max() is not differentiable at a
    # tie, so check a finite subgradient rather than gradcheck.
    for values, amplitude_only in (([0.0, 1.0, -1.0, 4.0, 2.0, 0.0], False), ([0.0, 3.0, -3.0, 1.0], True)):
        trajectory = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        katz_dimension(trajectory, amplitude_only=amplitude_only).backward()
        assert torch.isfinite(trajectory.grad).all()


def test_katz_dimension_interior_farthest_point():
    # [0, 100, 0, 0]: the farthest point from the start is interior (i=1).
    # Planar: L = sqrt(1+100^2)*2 + 1, d = sqrt(1+100^2).
    # Amplitude only: L = 200, d = 100.
    trajectory = torch.tensor([0.0, 100.0, 0.0, 0.0], dtype=torch.float64)
    step = math.sqrt(1 + 100**2)
    for amplitude_only, length, extent in ((False, 2 * step + 1, step), (True, 200.0, 100.0)):
        expected = math.log10(3) / (math.log10(3) + math.log10(extent / length))
        assert torch.isclose(katz_dimension(trajectory, amplitude_only=amplitude_only), torch.tensor(expected, dtype=torch.float64))


def test_katz_dimension_zero_denominator_is_nan():
    # Amplitude only, [0, 1, 0, 1]: L = 3, d = 1, n = 3, so d / L = 1 / n and
    # the denominator is zero (log10 rounding alone gave -5.6e-17 -> D ~ 1.6e16).
    for dtype in (torch.float32, torch.float64):
        assert torch.isnan(katz_dimension(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype), amplitude_only=True))
        assert torch.isnan(katz_dimension(torch.tensor([0.0, 1.0, 0.0], dtype=dtype), amplitude_only=True))


def test_katz_dimension_degenerate_row_does_not_poison_batch_gradient():
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, -1.0, 5.0]], dtype=torch.float64).unsqueeze(-1)
    result = katz_dimension(scale * batch, amplitude_only=True)
    assert torch.isnan(result[0])
    (gradient,) = torch.autograd.grad(result[1], scale)
    # Amplitude-only D is invariant to a positive rescaling.
    assert torch.isclose(gradient, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_katz_dimension_invalid_input_raises():
    for bad in (torch.tensor([0.0, float("inf"), 1.0]), torch.tensor([0.0, float("nan"), 1.0]), torch.tensor([0j, 1j, 2j])):
        with pytest.raises(ValueError):
            katz_dimension(bad)


def test_katz_dimension_integer_and_half_input():
    unsigned = torch.tensor([1, 2, 1, 3], dtype=torch.uint8)  # 1 - 2 would wrap in uint8
    assert torch.isclose(katz_dimension(unsigned, amplitude_only=True).double(), katz_dimension(unsigned.double(), amplitude_only=True), atol=1e-6)
    # index 256 squared overflows float16; a constant series is a straight line.
    half = katz_dimension(torch.zeros(257, dtype=torch.float16))
    assert half.dtype == torch.float16 and half.item() == 1.0
    wide = torch.tensor([0.0, 300.0, 0.0, 100.0])
    assert torch.isclose(katz_dimension(wide.half()).double(), katz_dimension(wide.double()), atol=1e-2)


def test_katz_dimension_bounded_for_typical_waveforms():
    # Katz's documented range: close to 1 for a smooth signal, higher
    # (up to roughly 1.5 in practice per Katz's own paper) for a rough
    # one -- checked on two clearly-contrasting real signals rather than
    # trusting the range claim on faith. Unlike Higuchi's D_H, Katz's D
    # is not bounded to [1, 2] even in theory (see module docstring).
    time_points = torch.linspace(0, 20 * torch.pi, 2000, dtype=torch.float64)
    sine_dimension = katz_dimension(torch.sin(time_points))
    assert 1.0 <= sine_dimension.item() < 1.1

    torch.manual_seed(5)
    noise_dimension = katz_dimension(torch.randn(2000, dtype=torch.float64))
    assert noise_dimension.item() > sine_dimension.item()
