"""Tests for torchnlts.embedding.fractional_delay.fractional_delay_embedding.

No Julia equivalent exists (DelayEmbeddings.jl's `embed` only accepts
integer lags -- checked directly), so this validates against:
  1. The paper: Laakso et al. (1996)'s linear-interpolation fractional
     delay, hand-derived.
  2. Internal consistency: for integer lags, must match
     torchnlts.embedding.delay.delay_embedding exactly (itself already
     dual-validated against Takens (1981) and DelayEmbeddings.jl).
  3. A gradcheck (gradient class "piecewise", away from the integer-lag
     boundary).
"""

import math

import pytest
import torch

from torchnlts.embedding.delay import delay_embedding
from torchnlts.embedding.fractional_delay import fractional_delay_embedding


def test_fractional_delay_embedding_hand_derived():
    # trajectory[n] = n, so linear interpolation at t gives exactly t.
    trajectory = torch.arange(10.0)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0.0, 1.5, 3.25]))
    expected = torch.stack(
        [torch.arange(6.0) + 0.0, torch.arange(6.0) + 1.5, torch.arange(6.0) + 3.25],
        dim=1,
    )
    assert torch.allclose(embedded, expected, atol=1e-6)


def test_fractional_delay_embedding_matches_integer_delay_embedding():
    torch.manual_seed(0)
    trajectory = torch.randn(20, 2, dtype=torch.float64)
    embedding_dim, delay = 4, 3
    exact = delay_embedding(trajectory, embedding_dim, delay=delay)
    fractional = fractional_delay_embedding(trajectory, delays=torch.arange(embedding_dim, dtype=torch.float64) * delay)
    assert torch.allclose(exact, fractional, atol=1e-10)


def test_fractional_delay_embedding_shapes():
    trajectory = torch.randn(2, 20, 3)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0.0, 2.5, 5.0]))
    max_delay = 5.0
    expected_n = math.floor((20 - 1) - max_delay) + 1
    assert embedded.shape == (2, expected_n, 3 * 3)


def test_fractional_delay_embedding_rejects_negative_delays():
    with pytest.raises(ValueError):
        fractional_delay_embedding(torch.randn(10), delays=torch.tensor([0.0, -1.0]))


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_fractional_delay_embedding_rejects_non_finite_delays(bad):
    # Regression: an infinite delay escaped as an OverflowError from math.floor, NaN as an unrelated ValueError.
    with pytest.raises(ValueError, match="finite"):
        fractional_delay_embedding(torch.randn(10), delays=torch.tensor([0.0, bad]))


def test_fractional_delay_embedding_rejects_empty_delays():
    with pytest.raises(ValueError):
        fractional_delay_embedding(torch.randn(10), delays=torch.tensor([]))


def test_fractional_delay_embedding_float64_trajectory_integer_delays_full_precision():
    # Regression: integer delays were cast to float32 instead of a float64 trajectory's dtype.
    trajectory = torch.arange(10, dtype=torch.float64) ** 3 + 1 / 3
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 2]))
    exact = delay_embedding(trajectory, embedding_dim=2, delay=2)
    assert torch.equal(embedded, exact)


def test_fractional_delay_embedding_low_precision_trajectory_no_index_corruption():
    # Regression: using the trajectory dtype for integer delays broke float16/bfloat16 (arange loses integers).
    trajectory = torch.arange(2000, dtype=torch.bfloat16)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 300]))
    assert embedded.shape == (1700, 2)


def test_fractional_delay_embedding_float32_delays_float64_trajectory_full_precision():
    # Regression: float32 delays with a float64 trajectory still lost precision to a float32 round-trip.
    trajectory = torch.arange(17_000_500, dtype=torch.float64)
    out32 = fractional_delay_embedding(trajectory, delays=torch.tensor([0.5], dtype=torch.float32))
    out64 = fractional_delay_embedding(trajectory, delays=torch.tensor([0.5], dtype=torch.float64))
    assert torch.equal(out32, out64)


def test_fractional_delay_embedding_bfloat16_default_dtype_no_index_corruption():
    # Regression: bit-width comparison picked bfloat16 (coarse mantissa) as precise enough when it was the default.
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        trajectory = torch.arange(2000, dtype=torch.float16)
        embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 300]))
        assert embedded.shape == (1700, 2)
    finally:
        torch.set_default_dtype(previous_default)


def test_fractional_delay_embedding_too_short_raises():
    with pytest.raises(ValueError):
        fractional_delay_embedding(torch.randn(5), delays=torch.tensor([0.0, 10.0]))


def test_fractional_delay_embedding_integer_trajectory_preserves_fraction():
    # Regression: fractional lags were truncated (1.5 -> 1).
    trajectory = torch.arange(6)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0.0, 1.5]))
    expected = torch.stack([torch.arange(4.0), torch.arange(4.0) + 1.5], dim=1)
    assert torch.allclose(embedded, expected)


def test_fractional_delay_embedding_preserves_delays_requires_grad():
    # Regression: casting delays to an integer trajectory dtype also dropped requires_grad.
    trajectory = torch.arange(6)
    delays = torch.tensor([0.0, 1.5], requires_grad=True)
    embedded = fractional_delay_embedding(trajectory, delays)
    embedded.sum().backward()
    assert delays.grad is not None


def test_fractional_delay_embedding_gradcheck():
    trajectory = torch.randn(10, dtype=torch.float64, requires_grad=True)
    # Half-integer delays, away from the integer-lag boundary where the derivative jumps.
    delays = torch.tensor([0.5, 2.5, 4.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fractional_delay_embedding, (trajectory, delays))
