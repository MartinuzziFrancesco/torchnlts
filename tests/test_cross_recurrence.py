"""Tests for torchnlts.recurrence.cross_recurrence.

Dual ground truth:
  1. Paper: Marwan, Romano, Thiel & Kurths (2007), Physics Reports
     438(5-6), 237-329, Sec. 3.3 -- checked here via a hand-derived example
     for the hard variant. No Julia package implements the smooth kernel
     variant, so the soft variant instead gets a hand-derived analytic
     limit (temperature -> 0 recovers the epsilon=0 hard matrix) plus a
     gradient check backing its "smooth" gradient-class claim.
  2. Julia: RecurrenceAnalysis.jl's `CrossRecurrenceMatrix` (hard variant only).
"""

import subprocess

import torch

from torchnlts.recurrence.cross_recurrence import hard_cross_recurrence_matrix, soft_cross_recurrence_matrix

JULIA_PROJECT = "juliaenv"


def test_hard_cross_recurrence_matrix_hand_derived():
    # x = [0, 1, 0.3], y = [0.2, 0.9], epsilon=0.5. Distances:
    # |0-0.2|=0.2<=eps, |0-0.9|=0.9>eps, |1-0.2|=0.8>eps, |1-0.9|=0.1<=eps,
    # |0.3-0.2|=0.1<=eps, |0.3-0.9|=0.6>eps.
    x = torch.tensor([0.0, 1.0, 0.3])
    y = torch.tensor([0.2, 0.9])
    expected = torch.tensor(
        [
            [True, False],
            [False, True],
            [True, False],
        ]
    )
    actual = hard_cross_recurrence_matrix(x, y, epsilon=0.5)
    assert torch.equal(actual, expected)


def test_hard_cross_recurrence_matrix_not_square_or_symmetric():
    x = torch.tensor([0.0, 1.0, 0.3])
    y = torch.tensor([0.2, 0.9])
    actual = hard_cross_recurrence_matrix(x, y, epsilon=0.5)
    assert actual.shape == (3, 2)


def test_hard_cross_recurrence_matrix_rejects_batch_mismatch():
    # Regression: batch/feature-dim mismatches must surface here as in pairwise_distance (test_distance.py).
    unbatched = torch.randn(3)
    batched = torch.randn(2, 5, 1)
    try:
        hard_cross_recurrence_matrix(unbatched, batched, epsilon=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a batched/unbatched mismatch")


def test_hard_cross_recurrence_matrix_batched():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 3, dtype=torch.float64)
    y = torch.randn(4, 5, 3, dtype=torch.float64)
    actual = hard_cross_recurrence_matrix(x, y, epsilon=1.0)
    assert actual.shape == (4, 8, 5)
    brute = torch.zeros(4, 8, 5, dtype=torch.bool)
    for b in range(4):
        for i in range(8):
            for j in range(5):
                brute[b, i, j] = torch.linalg.norm(x[b, i] - y[b, j]) <= 1.0
    assert torch.equal(actual, brute)


def test_hard_cross_recurrence_matrix_chebyshev():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    y = torch.tensor([[0.0, 1.0]])
    actual = hard_cross_recurrence_matrix(x, y, epsilon=1.0, norm_order=float("inf"))
    assert torch.equal(actual, torch.tensor([[True], [False]]))


def test_hard_cross_recurrence_matrix_matches_recurrenceanalysis_jl():
    torch.manual_seed(2)
    x = torch.randn(20, dtype=torch.float64)
    y = torch.randn(15, dtype=torch.float64)
    epsilon = 0.75

    x_values = ", ".join(f"{v:.17g}" for v in x.numpy())
    y_values = ", ".join(f"{v:.17g}" for v in y.numpy())
    julia_code = f"""
    using RecurrenceAnalysis
    x = [{x_values}]
    y = [{y_values}]
    R = CrossRecurrenceMatrix(x, y, {epsilon})
    for row in eachrow(Matrix(R))
        println(join(Int.(row), ","))
    end
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_matrix = torch.tensor(
        [[int(v) for v in line.split(",")] for line in result.stdout.strip().splitlines()],
        dtype=torch.bool,
    )
    torch_matrix = hard_cross_recurrence_matrix(x, y, epsilon=epsilon)
    assert torch.equal(torch_matrix, julia_matrix)


def test_soft_cross_recurrence_matrix_monotone_in_distance():
    # y=0 is fixed; x=1 is closer than x=4, so it gets the larger recurrence weight.
    x = torch.tensor([1.0, 4.0])
    y = torch.tensor([0.0])
    actual = soft_cross_recurrence_matrix(x, y, temperature=1.0)
    assert actual[0, 0] > actual[1, 0]


def test_soft_cross_recurrence_matrix_low_temperature_limit_matches_epsilon_zero_hard():
    x = torch.tensor([0.0, 1.0, 0.3])
    y = torch.tensor([0.0, 1.0])
    soft = soft_cross_recurrence_matrix(x, y, temperature=1e-6)
    hard = hard_cross_recurrence_matrix(x, y, epsilon=0.0)
    assert torch.allclose(soft, hard.to(soft.dtype), atol=1e-4)


def test_soft_cross_recurrence_matrix_batched_and_exponential():
    torch.manual_seed(3)
    x = torch.randn(4, 6, 2, dtype=torch.float64)
    y = torch.randn(4, 5, 2, dtype=torch.float64)
    actual = soft_cross_recurrence_matrix(x, y, temperature=0.5, kind="exponential")
    assert actual.shape == (4, 6, 5)
    brute = torch.zeros(4, 6, 5, dtype=torch.float64)
    for b in range(4):
        for i in range(6):
            for j in range(5):
                brute[b, i, j] = torch.exp(-torch.linalg.norm(x[b, i] - y[b, j]) / 0.5)
    assert torch.allclose(actual, brute, atol=1e-10)


def test_soft_cross_recurrence_matrix_gradcheck():
    torch.manual_seed(4)
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, dtype=torch.float64, requires_grad=True)

    def soft(x, y):
        return soft_cross_recurrence_matrix(x, y, temperature=0.5)

    assert torch.autograd.gradcheck(soft, (x, y))
