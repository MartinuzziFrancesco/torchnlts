"""Tests for torchnlts.embedding.fractional_delay.fractional_delay_embedding.

No Julia equivalent exists (DelayEmbeddings.jl's `embed` only accepts
integer lags -- checked directly), so this validates
against:
  1. The paper: Laakso et al. (1996)'s linear-interpolation fractional
     delay, hand-derived.
  2. Internal consistency: for integer lags, must match
     torchnlts.embedding.delay.delay_embedding exactly (itself already
     dual-validated against Takens (1981) and DelayEmbeddings.jl).
  3. A gradcheck (gradient class "piecewise", away from the integer-lag
     boundary).
"""

import math

import torch

from torchnlts.embedding.delay import delay_embedding
from torchnlts.embedding.fractional_delay import fractional_delay_embedding


def test_fractional_delay_embedding_hand_derived():
    # trajectory[n] = n, so linear interpolation at any real time t
    # gives exactly t (the trajectory is itself linear).
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
    exact = delay_embedding(trajectory, embedding_dim, delay)
    fractional = fractional_delay_embedding(trajectory, delays=torch.arange(embedding_dim, dtype=torch.float64) * delay)
    assert torch.allclose(exact, fractional, atol=1e-10)


def test_fractional_delay_embedding_shapes():
    trajectory = torch.randn(2, 20, 3)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0.0, 2.5, 5.0]))
    max_delay = 5.0
    expected_n = math.floor((20 - 1) - max_delay) + 1
    assert embedded.shape == (2, expected_n, 3 * 3)


def test_fractional_delay_embedding_rejects_negative_delays():
    try:
        fractional_delay_embedding(torch.randn(10), delays=torch.tensor([0.0, -1.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a negative delay")


def test_fractional_delay_embedding_rejects_empty_delays():
    try:
        fractional_delay_embedding(torch.randn(10), delays=torch.tensor([]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty delays")


def test_fractional_delay_embedding_float64_trajectory_integer_delays_full_precision():
    # Regression: integer delays were cast to the global default dtype
    # (float32) instead of a float64 trajectory's own dtype, losing
    # precision through an unnecessary float32 round-trip.
    trajectory = torch.arange(10, dtype=torch.float64) ** 3 + 1 / 3
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 2]))
    exact = delay_embedding(trajectory, embedding_dim=2, delay=2)
    assert torch.equal(embedded, exact)


def test_fractional_delay_embedding_low_precision_trajectory_no_index_corruption():
    # Regression: preferring the trajectory's own dtype for integer delays
    # (to fix the float64 case above) also applied to bfloat16/float16
    # trajectories, where `starts`'s arange stops representing consecutive
    # integers past a few hundred/thousand, corrupting indices out of bounds.
    trajectory = torch.arange(2000, dtype=torch.bfloat16)
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 300]))
    assert embedded.shape == (1700, 2)


def test_fractional_delay_embedding_float32_delays_float64_trajectory_full_precision():
    # Regression: the dtype-precision fix only triggered for integer
    # delays; already-floating delays (the common case for genuinely
    # fractional lags) skipped it entirely, so float32 delays paired with
    # a float64 trajectory still lost precision to an intermediate
    # float32 round-trip.
    trajectory = torch.arange(17_000_500, dtype=torch.float64)
    out32 = fractional_delay_embedding(trajectory, delays=torch.tensor([0.5], dtype=torch.float32))
    out64 = fractional_delay_embedding(trajectory, delays=torch.tensor([0.5], dtype=torch.float64))
    assert torch.equal(out32, out64)


def test_fractional_delay_embedding_bfloat16_default_dtype_no_index_corruption():
    # Regression: comparing dtypes by bit width picked bfloat16 (same 16
    # bits as float16 but a much coarser mantissa) as "precise enough"
    # when it was the global default, corrupting indices exactly like the
    # low_precision_trajectory regression above, triggered from the
    # opposite direction (default dtype low-precision, not the trajectory).
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        trajectory = torch.arange(2000, dtype=torch.float16)
        embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0, 300]))
        assert embedded.shape == (1700, 2)
    finally:
        torch.set_default_dtype(previous_default)


def test_fractional_delay_embedding_too_short_raises():
    try:
        fractional_delay_embedding(torch.randn(5), delays=torch.tensor([0.0, 10.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a too-short trajectory")


def test_fractional_delay_embedding_integer_trajectory_preserves_fraction():
    # Regression: casting delays to an integer trajectory's own dtype
    # silently truncated fractional lags (1.5 -> 1).
    trajectory = torch.arange(6)  # int64
    embedded = fractional_delay_embedding(trajectory, delays=torch.tensor([0.0, 1.5]))
    expected = torch.stack([torch.arange(4.0), torch.arange(4.0) + 1.5], dim=1)
    assert torch.allclose(embedded, expected)


def test_fractional_delay_embedding_preserves_delays_requires_grad():
    # Regression: casting delays to an integer trajectory's dtype also
    # silently dropped requires_grad off a learnable delays tensor.
    trajectory = torch.arange(6)
    delays = torch.tensor([0.0, 1.5], requires_grad=True)
    embedded = fractional_delay_embedding(trajectory, delays)
    embedded.sum().backward()
    assert delays.grad is not None


def test_fractional_delay_embedding_gradcheck():
    trajectory = torch.randn(10, dtype=torch.float64, requires_grad=True)
    # Half-integer delays, safely away from the integer-lag boundary
    # (where the interpolation derivative jumps, documented in the module).
    delays = torch.tensor([0.5, 2.5, 4.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fractional_delay_embedding, (trajectory, delays))
