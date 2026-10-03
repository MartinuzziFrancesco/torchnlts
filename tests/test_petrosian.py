"""Tests for torchnlts.dimension.petrosian.

Dual ground truth:
  1. Paper: Petrosian (1995) -- checked via hand-derived examples.
  2. Python reference: antropy.petrosian_fd.
"""

import torch

from torchnlts.dimension.petrosian import petrosian_dimension


def test_petrosian_hand_derived_monotonic():
    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], N=10
    # Derivative: [1, 1, 1, 1, 1, 1, 1, 1, 1], all positive -> N_delta=0
    # D = log10(10) / (log10(10) + log10(10 / (10 + 0))) = 1 / 1 = 1.0
    trajectory = torch.arange(10.0, dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_petrosian_hand_derived_alternating():
    # x = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1], N=10
    # Derivative: [-2, 2, -2, 2, -2, 2, -2, 2, -2], sign flips every time -> N_delta=8
    # D = log10(10) / (log10(10) + log10(10 / (10 + 0.4*8)))
    #   = log10(10) / (log10(10) + log10(10/13.2))
    #   = 1.0 / (1.0 + log10(0.757575...))
    #   ≈ 1.137105
    trajectory = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    expected = 1.137105
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-5)


def test_petrosian_hand_derived_constant():
    # x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], N=10
    # Derivative: [0, 0, 0, 0, 0, 0, 0, 0, 0], all zero (treated as positive) -> N_delta=0
    # D = 1.0
    trajectory = torch.ones(10, dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_petrosian_matches_antropy_sine():
    from antropy.fractal import petrosian_fd

    torch.manual_seed(42)
    time_points = torch.linspace(0, 4 * torch.pi, 1000, dtype=torch.float64)
    signal = torch.sin(time_points)

    torchnlts_result = petrosian_dimension(signal).item()
    antropy_result = petrosian_fd(signal.numpy())

    # Tolerance: antropy uses float64, torch should match closely
    assert abs(torchnlts_result - antropy_result) < 1e-5


def test_petrosian_matches_antropy_noise():
    from antropy.fractal import petrosian_fd

    torch.manual_seed(42)
    noise = torch.randn(1000, dtype=torch.float64)

    torchnlts_result = petrosian_dimension(noise).item()
    antropy_result = petrosian_fd(noise.numpy())

    assert abs(torchnlts_result - antropy_result) < 1e-5


def test_petrosian_matches_antropy_monotonic():
    from antropy.fractal import petrosian_fd

    trajectory = torch.arange(100.0, dtype=torch.float64)

    torchnlts_result = petrosian_dimension(trajectory).item()
    antropy_result = petrosian_fd(trajectory.numpy())

    assert abs(torchnlts_result - antropy_result) < 1e-10


def test_petrosian_short_trajectory_is_nan():
    # N=1: no differences to compute, N_delta undefined -> NaN
    trajectory = torch.tensor([1.0], dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert torch.isnan(result)


def test_petrosian_two_element_series():
    # N=2: one difference, no sign change -> N_delta=0, D=1.0
    trajectory = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_petrosian_multivariate_raises():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    try:
        petrosian_dimension(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_petrosian_unbatched_scalar_shape():
    trajectory = torch.randn(20, dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert result.shape == ()


def test_petrosian_batched_shape():
    torch.manual_seed(1)
    trajectory = torch.randn(4, 30, 1, dtype=torch.float64)
    result = petrosian_dimension(trajectory)
    assert result.shape == (4,)


def test_petrosian_batch_size_one_matches_unbatched():
    torch.manual_seed(2)
    trajectory = torch.randn(1, 30, 1, dtype=torch.float64)
    batched = petrosian_dimension(trajectory)
    unbatched = petrosian_dimension(trajectory[0])
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_petrosian_does_not_require_grad():
    # Gradient class is none: output should not have requires_grad=True
    # even if input does (count-based metric, not differentiable)
    trajectory = torch.randn(30, dtype=torch.float64, requires_grad=True)
    result = petrosian_dimension(trajectory)
    assert not result.requires_grad, "Petrosian output should not require grad (count-based statistic)"


def test_petrosian_with_grad_input_does_not_crash():
    # Input with requires_grad=True should not crash; the output is just
    # not backpropagable (count-based).
    trajectory = torch.randn(50, dtype=torch.float64, requires_grad=True)
    result = petrosian_dimension(trajectory)
    assert result.dtype == trajectory.dtype
    assert not result.requires_grad


def test_petrosian_heterogeneous_batch_matches_individual():
    monotonic = torch.arange(10.0, dtype=torch.float64)
    alternating = torch.tensor([1.0, -1.0] * 5, dtype=torch.float64)
    batch = torch.stack([monotonic, alternating]).unsqueeze(-1)
    result = petrosian_dimension(batch)
    expected = torch.stack([petrosian_dimension(monotonic), petrosian_dimension(alternating)])
    assert torch.allclose(result, expected)
    assert torch.allclose(result, torch.tensor([1.0, 1.1371052], dtype=torch.float64), atol=1e-6)


def test_petrosian_integer_input_matches_float():
    # int8 subtraction would overflow (127 - -128); input is cast to float first.
    integer = torch.tensor([0, 127, -128], dtype=torch.int8)
    result = petrosian_dimension(integer)
    assert result.dtype == torch.get_default_dtype()
    assert torch.isclose(result.double(), petrosian_dimension(integer.double()), atol=1e-6)
    assert torch.isnan(petrosian_dimension(torch.empty(0, dtype=torch.int64)))


def test_petrosian_float16_no_overflow():
    alternating = torch.tensor([1.0, -1.0], dtype=torch.float16).repeat(25000)
    result = petrosian_dimension(alternating)
    assert torch.isclose(result.double(), petrosian_dimension(alternating.double()), atol=1e-3)


def test_petrosian_plateau_counts_zero_difference_as_positive():
    # diffs [1, 0]: signbit False for both -> N_delta=0 -> D=1 (antropy's
    # convention); counting 0 as negative would give N_delta=1.
    from antropy.fractal import petrosian_fd

    trajectory = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    assert petrosian_dimension(trajectory).item() == 1.0
    assert petrosian_dimension(trajectory).item() == petrosian_fd(trajectory.numpy())


def test_petrosian_preserves_floating_dtype():
    trajectory = torch.tensor([1.0, -1.0, 2.0, 0.0])
    for dtype in (torch.float16, torch.float32, torch.float64):
        assert petrosian_dimension(trajectory.to(dtype)).dtype == dtype


def test_petrosian_complex_raises():
    try:
        petrosian_dimension(torch.tensor([0 + 1j, 1 + 0j, 0 + 2j]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_petrosian_negative_zero_difference_counts_as_negative():
    # 0 - 0 = +0 but -0 - 0 = -0: signbit differs, matching antropy.
    from antropy.fractal import petrosian_fd

    trajectory = torch.tensor([0.0, -0.0, 0.0], dtype=torch.float64)
    assert petrosian_dimension(trajectory).item() == petrosian_fd(trajectory.numpy())
    assert petrosian_dimension(trajectory).item() != 1.0


def test_petrosian_integer_input_follows_default_dtype():
    previous = torch.get_default_dtype()
    try:
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            assert petrosian_dimension(torch.tensor([0, 1, 0])).dtype == default
    finally:
        torch.set_default_dtype(previous)
