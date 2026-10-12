"""Tests for torchnlts.recurrence.matrix.

Dual ground truth:
  1. Paper: Marwan, Romano, Thiel & Kurths (2007), Physics Reports
     438(5-6), 237-329, eq. 1 -- checked here via a hand-derived example
     for the hard variant. No Julia package implements the smooth kernel
     variant, so the soft variant instead gets a hand-derived analytic
     limit (temperature -> 0 recovers the epsilon=0 hard matrix) plus a
     monotonicity check straight from the paper's underlying intuition
     (closer points are more recurrent).
  2. Julia: RecurrenceAnalysis.jl's `RecurrenceMatrix` (hard variant only).
"""

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix


def test_hard_recurrence_matrix_hand_derived():
    # Distances: |0-1|=1, |0-0.3|=0.3, |1-0.3|=0.7 (via 1D points).
    trajectory = torch.tensor([0.0, 1.0, 0.3])
    expected = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ]
    )
    actual = hard_recurrence_matrix(trajectory, epsilon=0.5)
    assert torch.equal(actual, expected)


def test_hard_recurrence_matrix_diagonal_always_true():
    trajectory = torch.tensor([0.0, 5.0, -3.0, 100.0])
    actual = hard_recurrence_matrix(trajectory, epsilon=0.0)
    assert torch.equal(torch.diagonal(actual), torch.ones(4, dtype=torch.bool))


def test_hard_recurrence_matrix_symmetric_and_batched():
    torch.manual_seed(0)
    trajectory = torch.randn(5, 8, 3, dtype=torch.float64)
    actual = hard_recurrence_matrix(trajectory, epsilon=1.0)
    assert actual.shape == (5, 8, 8)
    assert torch.equal(actual, actual.transpose(-1, -2))


def test_hard_recurrence_matrix_matches_recurrenceanalysis_jl():
    torch.manual_seed(2)
    trajectory = torch.randn(30, dtype=torch.float64)
    epsilon = 0.75

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using RecurrenceAnalysis
    x = [{values}]
    R = RecurrenceMatrix(x, {epsilon})
    for row in eachrow(Matrix(R))
        println(join(Int.(row), ","))
    end
    """
    stdout = run_julia(julia_code)
    julia_matrix = torch.tensor(
        [[int(v) for v in line.split(",")] for line in stdout.strip().splitlines()],
        dtype=torch.bool,
    )
    torch_matrix = hard_recurrence_matrix(trajectory, epsilon=epsilon)
    assert torch.equal(torch_matrix, julia_matrix)


def test_soft_recurrence_matrix_diagonal_is_one():
    trajectory = torch.tensor([0.0, 5.0, -3.0, 100.0])
    actual = soft_recurrence_matrix(trajectory, temperature=1.0)
    assert torch.allclose(torch.diagonal(actual), torch.ones(4))


def test_soft_recurrence_matrix_monotone_in_distance():
    # Points 0, 1, 4: the point closer to 0 must get the larger recurrence weight.
    trajectory = torch.tensor([0.0, 1.0, 4.0])
    actual = soft_recurrence_matrix(trajectory, temperature=1.0)
    assert actual[0, 1] > actual[0, 2]


def test_soft_recurrence_matrix_low_temperature_limit_matches_epsilon_zero_hard():
    trajectory = torch.tensor([0.0, 1.0, 0.3, 1.0])
    soft = soft_recurrence_matrix(trajectory, temperature=1e-6)
    hard = hard_recurrence_matrix(trajectory, epsilon=0.0)
    assert torch.allclose(soft, hard.to(soft.dtype), atol=1e-4)


def test_soft_recurrence_matrix_batched_and_exponential():
    torch.manual_seed(3)
    trajectory = torch.randn(4, 6, 2, dtype=torch.float64)
    actual = soft_recurrence_matrix(trajectory, temperature=0.5, kind="exponential")
    assert actual.shape == (4, 6, 6)
    assert torch.allclose(torch.diagonal(actual, dim1=-2, dim2=-1), torch.ones(4, 6, dtype=torch.float64))
    assert torch.equal(actual, actual.transpose(-1, -2))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), torch.tensor([0.5, 0.0])])
def test_soft_recurrence_matrix_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        soft_recurrence_matrix(torch.randn(8), temperature)


def test_soft_recurrence_matrix_weights_lie_in_the_closed_unit_interval():
    trajectory = torch.tensor([0.0, 1.0, 300.0, float("nan")])
    weights = soft_recurrence_matrix(trajectory, temperature=0.5)
    assert ((weights >= 0) & (weights <= 1)).all()
    assert weights[0, 2] == 0 and weights[3].tolist() == [0.0] * 4 and weights[:, 3].tolist() == [0.0] * 4
