"""Tests for torchnlts.dimension.katz.

Dual ground truth:
  1. Paper: Katz (1988), eq. (6), distances in the (time index,
     amplitude) plane -- checked via a hand-derived example.
  2. Reference implementation: antropy's ``katz_fd``, which uses
     amplitude differences only (no time-index term) -- checked via
     ``amplitude_only=True``, which reproduces it.
"""

import decimal
import math

import pytest
import torch

from torchnlts.dimension.katz import katz_dimension


def test_katz_dimension_hand_derived_paper_convention():
    # x = [1, 2, 1, 3], points (0,1),(1,2),(2,1),(3,3), n=3: L = 2*sqrt(2)+sqrt(5), d = sqrt(13).
    # D = log10(3) / (log10(3) + log10(d/L)) = 1.4477664861880815
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    dimension = katz_dimension(trajectory)
    assert torch.isclose(dimension, torch.tensor(1.4477664861880815, dtype=torch.float64))


def test_katz_dimension_hand_derived_amplitude_only():
    # Amplitude-only (antropy's katz_fd): steps 1,1,2 give L=4, d=2, n=3.
    # D = log10(3) / (log10(3) + log10(d/L)) = log10(3) / (log10(3) - log10(2))
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
    assert torch.isclose(torch_value, torch.tensor(float(antropy_value), dtype=torch.float64), atol=1e-9)


def test_katz_dimension_straight_line_is_one():
    # A straight line: d equals L for any slope and n, so D = 1 under both conventions.
    line = torch.arange(0.0, 100.0, dtype=torch.float64) * 2.5 + 1.0
    assert torch.isclose(katz_dimension(line), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(katz_dimension(line, amplitude_only=True), torch.tensor(1.0, dtype=torch.float64))


def test_katz_dimension_constant_trajectory_paper_convention_is_one():
    # A constant series is a horizontal line in the planar convention: D = 1.
    constant = torch.full((50,), 3.0, dtype=torch.float64)
    assert torch.isclose(katz_dimension(constant), torch.tensor(1.0, dtype=torch.float64))


def test_katz_dimension_constant_trajectory_amplitude_only_is_nan():
    constant = torch.full((50,), 3.0, dtype=torch.float64)
    assert torch.isnan(katz_dimension(constant, amplitude_only=True))


def test_katz_dimension_two_points_is_nan_not_crash():
    # n=1: d equals the single step, so both log10 terms are zero: 0/0.
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
    # Planar: (3, 4) and (5, 0) are both 5 from the origin. max() has no gradient at a tie: check a finite subgradient.
    for values, amplitude_only in (([0.0, 1.0, -1.0, 4.0, 2.0, 0.0], False), ([0.0, 3.0, -3.0, 1.0], True)):
        trajectory = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        katz_dimension(trajectory, amplitude_only=amplitude_only).backward()
        assert torch.isfinite(trajectory.grad).all()


def test_katz_dimension_interior_farthest_point():
    # [0, 100, 0, 0]: the farthest point is interior (i=1). Planar: L = 2*sqrt(1+100^2) + 1, d = sqrt(1+100^2);
    # amplitude-only: L = 200, d = 100.
    trajectory = torch.tensor([0.0, 100.0, 0.0, 0.0], dtype=torch.float64)
    step = math.sqrt(1 + 100**2)
    for amplitude_only, length, extent in ((False, 2 * step + 1, step), (True, 200.0, 100.0)):
        expected = math.log10(3) / (math.log10(3) + math.log10(extent / length))
        assert torch.isclose(
            katz_dimension(trajectory, amplitude_only=amplitude_only), torch.tensor(expected, dtype=torch.float64)
        )


def test_katz_dimension_zero_denominator_is_nan():
    # Amplitude-only [0, 1, 0, 1]: d / L = 1 / n, so the denominator is zero (rounding alone gave D ~ 1.6e16).
    for dtype in (torch.float32, torch.float64):
        assert torch.isnan(katz_dimension(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype), amplitude_only=True))
        assert torch.isnan(katz_dimension(torch.tensor([0.0, 1.0, 0.0], dtype=dtype), amplitude_only=True))


def test_katz_dimension_degenerate_row_does_not_poison_batch_gradient():
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, -1.0, 5.0]], dtype=torch.float64).unsqueeze(-1)
    result = katz_dimension(scale * batch, amplitude_only=True)
    assert torch.isnan(result[0])
    (gradient,) = torch.autograd.grad(result[1], scale)
    assert torch.isclose(gradient, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_katz_dimension_invalid_input_raises():
    for bad in (torch.tensor([0j, 1j, 2j]),):
        with pytest.raises(ValueError):
            katz_dimension(bad)


def test_katz_dimension_integer_and_half_input():
    unsigned = torch.tensor([1, 2, 1, 3], dtype=torch.uint8)  # 1 - 2 would wrap in uint8
    assert torch.isclose(
        katz_dimension(unsigned, amplitude_only=True).double(),
        katz_dimension(unsigned.double(), amplitude_only=True),
        atol=1e-6,
    )
    # Index 256 squared overflows float16.
    half = katz_dimension(torch.zeros(257, dtype=torch.float16))
    assert half.dtype == torch.float16 and half.item() == 1.0
    wide = torch.tensor([0.0, 300.0, 0.0, 100.0])
    assert torch.isclose(katz_dimension(wide.half()).double(), katz_dimension(wide.double()), atol=1e-2)


def test_katz_dimension_bounded_for_typical_waveforms():
    # Katz's D is near 1 for a smooth signal and higher for a rough one (not bounded to [1, 2], see module docstring).
    time_points = torch.linspace(0, 20 * torch.pi, 2000, dtype=torch.float64)
    sine_dimension = katz_dimension(torch.sin(time_points))
    assert 1.0 <= sine_dimension.item() < 1.1

    torch.manual_seed(5)
    noise_dimension = katz_dimension(torch.randn(2000, dtype=torch.float64))
    assert noise_dimension.item() > sine_dimension.item()


def test_katz_dimension_pole_within_rounding_is_nan():
    # Exactly singular alternating series (L = n * d) and inputs one ulp from the pole:
    # the denominator is at rounding level, so D is undetermined.
    for dtype, count, amplitude in ((torch.float32, 7, 0.3), (torch.float64, 10, 0.1)):
        assert torch.isnan(katz_dimension((torch.arange(count) % 2).to(dtype) * amplitude, amplitude_only=True))
    # One float64 ulp from the pole gives NaN; float32 is resolved since Katz computes in float64.
    one = torch.tensor(1.0, dtype=torch.float64)
    below = torch.nextafter(one, torch.zeros_like(one))
    assert torch.isnan(katz_dimension(torch.stack([one * 0, one, one * 0, below]), amplitude_only=True))
    single = torch.stack([one * 0, one, one * 0, torch.nextafter(one.float(), torch.zeros(())).double()]).float()
    assert torch.isclose(
        katz_dimension(single, amplitude_only=True).double(),
        katz_dimension(single.double(), amplitude_only=True),
        rtol=1e-6,
    )
    # Large but determined: [0, 1, 0, 0.9], amplitude-only L = 2.9, d = 1, n = 3.
    expected = math.log10(3) / (math.log10(3) + math.log10(1 / 2.9))
    assert torch.isclose(
        katz_dimension(torch.tensor([0.0, 1.0, 0.0, 0.9], dtype=torch.float64), amplitude_only=True),
        torch.tensor(expected, dtype=torch.float64),
    )


def test_katz_dimension_extreme_amplitude():
    # Amplitude-only D is scale invariant; at these amplitudes the planar index term is negligible.
    reference = katz_dimension(torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float64), amplitude_only=True)
    for dtype, amplitude in ((torch.float32, 1e36), (torch.float64, 1e300)):
        values = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=dtype) * amplitude
        for amplitude_only in (True, False):
            assert torch.isclose(katz_dimension(values, amplitude_only=amplitude_only).double(), reference, atol=1e-5)


def test_katz_dimension_tiny_singular_row_does_not_poison_batch_gradient():
    for dtype, amplitude in ((torch.float32, 1e-40), (torch.float64, 1e-310)):
        scale = torch.tensor(1.0, dtype=dtype, requires_grad=True)
        batch = torch.tensor([[0.0, amplitude, 0.0, amplitude], [0.0, 2.0, -1.0, 5.0]], dtype=dtype).unsqueeze(-1)
        result = katz_dimension(scale * batch, amplitude_only=True)
        assert torch.isnan(result[0])
        (gradient,) = torch.autograd.grad(result[1], scale)
        assert torch.isclose(gradient, torch.zeros_like(gradient), atol=1e-6)


def test_katz_dimension_planar_tiny_amplitude_is_near_one():
    # A subnormal float32 series is almost a straight line in the planar convention: D near 1, no index-term overflow.
    torch.manual_seed(9)
    values = torch.randn(50) * 1e-39
    assert abs(katz_dimension(values).item() - 1.0) < 1e-6


def test_katz_dimension_pole_with_offset_is_nan():
    # [1000, 1001, 999.5, 1000], amplitude-only: L = 3 * d (exact pole); scaling before centering gave -2.96e13.
    for dtype in (torch.float32, torch.float64):
        trajectory = torch.tensor([1000.0, 1001.0, 999.5, 1000.0], dtype=dtype)
        assert torch.isnan(katz_dimension(trajectory, amplitude_only=True))
        assert torch.isfinite(katz_dimension(trajectory))


def test_katz_dimension_resolved_value_near_pole_is_kept():
    # Alternating series, last value moved by 0.1: gap 1e-4, D ~ 69000 is determined; float32 must match float64.
    trajectory = (torch.arange(1000) % 2).to(torch.float32)
    trajectory[-1] -= 0.1
    single = katz_dimension(trajectory, amplitude_only=True)
    double = katz_dimension(trajectory.double(), amplitude_only=True)
    assert torch.isfinite(single) and abs(single.item() / double.item() - 1) < 1e-3


def test_katz_dimension_exact_pole_needs_rounding_tolerance():
    # [0, 3, -1, 1], amplitude-only: L = 3 * d = 9. With zero tolerance D came out as -1.4e7 / 7.4e15 instead of NaN.
    for dtype in (torch.float32, torch.float64):
        assert torch.isnan(katz_dimension(torch.tensor([0.0, 3.0, -1.0, 1.0], dtype=dtype), amplitude_only=True))


def test_katz_dimension_tolerance_is_at_rounding_level_float64():
    # Last value moved by 1e-9: relative gap ~1e-12, above float64 rounding (~2e-14): determined.
    trajectory = (torch.arange(1000) % 2).to(torch.float64)
    trajectory[-1] -= 1e-9
    assert torch.isfinite(katz_dimension(trajectory, amplitude_only=True))
    trajectory = (torch.arange(1000) % 2).to(torch.float32)
    trajectory[-1] -= 1 / 256
    single = katz_dimension(trajectory, amplitude_only=True)
    assert torch.isclose(single.double(), katz_dimension(trajectory.double(), amplitude_only=True), rtol=1e-6)


def test_katz_dimension_steps_from_raw_values():
    # float64 spacing at 1e17 is 16: subtracting the first value before differencing rounds steps of 3 to 0.
    trajectory = torch.cat([torch.tensor([1e17], dtype=torch.float64), (torch.arange(10000) % 2).to(torch.float64) * 3])
    steps = trajectory[1:] - trajectory[:-1]
    length = steps.abs().sum()
    extent = (trajectory[1:] - trajectory[0]).abs().max()
    num_steps = torch.tensor(float(trajectory.shape[0] - 1), dtype=torch.float64)
    expected = torch.log10(num_steps) / (torch.log10(num_steps) + torch.log10(extent / length))
    assert torch.isclose(katz_dimension(trajectory, amplitude_only=True), expected, rtol=1e-12)
    assert katz_dimension(trajectory, amplitude_only=True).item() != 1.0


def test_near_pole_matches_high_precision_oracle():
    # Denominator log10(T d / L) ~ 1e-13 here; log10 of the ratio loses ~7e-5.
    trajectory = torch.tensor([0.0, 1.0, 0.0, 1.0 - 2**-40], dtype=torch.float64)
    decimal.getcontext().prec = 80
    expected = float(decimal.Decimal(3).ln() / (decimal.Decimal(3) / (3 - decimal.Decimal(2) ** -40)).ln())
    assert math.isclose(katz_dimension(trajectory, amplitude_only=True).item(), expected, rel_tol=1e-10)


def test_pole_criterion_uses_relative_gap():
    # Gap 2**-46 is resolved in float64; the denominator (~1e-14) is below the tolerance,
    # so a denominator-based pole test wrongly gives NaN.
    trajectory = torch.tensor([0.0, 1.0, 0.0, 1.0 - 2**-46], dtype=torch.float64)
    decimal.getcontext().prec = 80
    expected = float(decimal.Decimal(3).ln() / (decimal.Decimal(3) / (3 - decimal.Decimal(2) ** -46)).ln())
    assert math.isclose(katz_dimension(trajectory, amplitude_only=True).item(), expected, rel_tol=1e-6)
