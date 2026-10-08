"""Tests for torchnlts.recurrence.joint_recurrence.

Dual ground truth:
  1. Paper: Marwan, Romano, Thiel & Kurths (2007), Physics Reports
     438(5-6), 237-329, Sec. 3.4 -- checked here via a hand-derived example
     for the hard variant. No Julia package implements the smooth kernel
     variant, so the soft variant instead gets a hand-derived analytic
     limit (temperature -> 0 recovers the epsilon=0 hard matrix) plus a
     gradient check backing its "smooth" gradient-class claim.
  2. Julia: RecurrenceAnalysis.jl's `RecurrenceMatrix` element-wise AND'd
     (the documented definition of `JointRecurrenceMatrix`; the package's
     own two-argument `JointRecurrenceMatrix(R1, R2)` constructor errors in
     the pinned version, so the element-wise AND is used directly -- it's
     the same computation `JointRecurrenceMatrix(x, y, rthres)` performs
     internally, confirmed to agree with it for a shared threshold).
"""

import subprocess

import torch

from torchnlts.recurrence.joint_recurrence import hard_joint_recurrence_matrix, soft_joint_recurrence_matrix

JULIA_PROJECT = "juliaenv"


def test_hard_joint_recurrence_matrix_hand_derived():
    # x = [0, 1, 2], epsilon_x=1.5: R(x) = [[T,T,F],[T,T,T],[F,T,T]].
    # y = [0, 3, 1], epsilon_y=1.5: R(y) = [[T,F,T],[F,T,F],[T,F,T]].
    # Element-wise AND leaves only the diagonal.
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([0.0, 3.0, 1.0])
    actual = hard_joint_recurrence_matrix(x, y, epsilon=1.5)
    assert torch.equal(actual, torch.eye(3, dtype=torch.bool))


def test_hard_joint_recurrence_matrix_different_epsilon_per_trajectory():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([0.0, 3.0, 1.0])
    # epsilon_y=3.5 makes every pair of y recurrent, so JR should equal R(x) alone.
    actual = hard_joint_recurrence_matrix(x, y, epsilon=1.5, other_epsilon=3.5)
    from torchnlts.recurrence.matrix import hard_recurrence_matrix

    assert torch.equal(actual, hard_recurrence_matrix(x, epsilon=1.5))


def test_hard_joint_recurrence_matrix_truncates_to_shorter_length():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([0.0, 3.0, 1.0])
    actual = hard_joint_recurrence_matrix(x, y, epsilon=1.5)
    assert actual.shape == (3, 3)
    # Value, not just shape: must equal truncate-then-compute, i.e. the
    # matrix is identical to running on the already-shortened trajectories.
    from torchnlts.recurrence.matrix import hard_recurrence_matrix

    truncated_first = hard_recurrence_matrix(x[:3], epsilon=1.5) & hard_recurrence_matrix(y[:3], epsilon=1.5)
    assert torch.equal(actual, truncated_first)


def test_hard_joint_recurrence_matrix_batched():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 3, dtype=torch.float64)
    y = torch.randn(4, 8, 2, dtype=torch.float64)  # different feature dim, same batch/length
    actual = hard_joint_recurrence_matrix(x, y, epsilon=1.0)
    assert actual.shape == (4, 8, 8)


def test_hard_joint_recurrence_matrix_batched_different_length():
    torch.manual_seed(1)
    x = torch.randn(3, 8, 2, dtype=torch.float64)
    y = torch.randn(3, 5, 4, dtype=torch.float64)
    actual = hard_joint_recurrence_matrix(x, y, epsilon=1.0)
    assert actual.shape == (3, 5, 5)
    from torchnlts.recurrence.matrix import hard_recurrence_matrix

    truncated_first = hard_recurrence_matrix(x[:, :5], epsilon=1.0) & hard_recurrence_matrix(y, epsilon=1.0)
    assert torch.equal(actual, truncated_first)


def test_hard_joint_recurrence_matrix_rejects_batch_mismatch():
    unbatched = torch.randn(3)
    batched = torch.randn(2, 5, 1)
    other_batched = torch.randn(3, 5, 1)
    for x, y in ((unbatched, batched), (batched, unbatched), (batched, other_batched)):
        try:
            hard_joint_recurrence_matrix(x, y, epsilon=0.5)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for mismatched shapes {tuple(x.shape)} vs {tuple(y.shape)}")


def test_hard_joint_recurrence_matrix_matches_recurrenceanalysis_jl():
    torch.manual_seed(2)
    x = torch.randn(20, dtype=torch.float64)
    y = torch.randn(20, dtype=torch.float64)
    epsilon_x, epsilon_y = 0.75, 0.5

    x_values = ", ".join(f"{v:.17g}" for v in x.numpy())
    y_values = ", ".join(f"{v:.17g}" for v in y.numpy())
    julia_code = f"""
    using RecurrenceAnalysis
    x = [{x_values}]
    y = [{y_values}]
    R1 = Matrix(RecurrenceMatrix(x, {epsilon_x}))
    R2 = Matrix(RecurrenceMatrix(y, {epsilon_y}))
    JR = R1 .& R2
    for row in eachrow(JR)
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
    torch_matrix = hard_joint_recurrence_matrix(x, y, epsilon=epsilon_x, other_epsilon=epsilon_y)
    assert torch.equal(torch_matrix, julia_matrix)


def test_soft_joint_recurrence_matrix_low_temperature_limit_matches_epsilon_zero_hard():
    x = torch.tensor([0.0, 1.0, 0.3])
    y = torch.tensor([0.0, 3.0, 0.25])
    soft = soft_joint_recurrence_matrix(x, y, temperature=1e-6)
    hard = hard_joint_recurrence_matrix(x, y, epsilon=0.0)
    assert torch.allclose(soft, hard.to(soft.dtype), atol=1e-4)


def test_soft_joint_recurrence_matrix_truncates_to_shorter_length():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y = torch.tensor([0.0, 3.0, 1.0], dtype=torch.float64)
    actual = soft_joint_recurrence_matrix(x, y, temperature=0.5)
    assert actual.shape == (3, 3)
    from torchnlts.recurrence.matrix import soft_recurrence_matrix

    truncated_first = soft_recurrence_matrix(x[:3], temperature=0.5) * soft_recurrence_matrix(y[:3], temperature=0.5)
    assert torch.allclose(actual, truncated_first)


def test_soft_joint_recurrence_matrix_different_temperature_per_trajectory():
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 3.0, 1.0], dtype=torch.float64)
    from torchnlts.recurrence.matrix import soft_recurrence_matrix

    actual = soft_joint_recurrence_matrix(x, y, temperature=0.5, other_temperature=2.0)
    expected = soft_recurrence_matrix(x, temperature=0.5) * soft_recurrence_matrix(y, temperature=2.0)
    assert torch.allclose(actual, expected)
    # A bug that silently ignored other_temperature would use temperature=0.5
    # for y too, giving a different (and detectably wrong) result.
    wrong = soft_recurrence_matrix(x, temperature=0.5) * soft_recurrence_matrix(y, temperature=0.5)
    assert not torch.allclose(actual, wrong)


def test_soft_joint_recurrence_matrix_batched():
    torch.manual_seed(3)
    x = torch.randn(4, 6, 2, dtype=torch.float64)
    y = torch.randn(4, 6, 3, dtype=torch.float64)
    actual = soft_joint_recurrence_matrix(x, y, temperature=0.5)
    assert actual.shape == (4, 6, 6)
    # Brute-force per-batch check catches an axis-transposition bug that a
    # shape-only assertion would miss.
    from torchnlts.neighbors.kernels import trajectory_kernel

    brute = torch.zeros(4, 6, 6, dtype=torch.float64)
    for b in range(4):
        for i in range(6):
            for j in range(6):
                wx = trajectory_kernel(torch.linalg.norm(x[b, i] - x[b, j]), 0.5)
                wy = trajectory_kernel(torch.linalg.norm(y[b, i] - y[b, j]), 0.5)
                brute[b, i, j] = wx * wy
    assert torch.allclose(actual, brute, atol=1e-10)


def test_soft_joint_recurrence_matrix_batched_different_length():
    torch.manual_seed(5)
    x = torch.randn(3, 8, 2, dtype=torch.float64)
    y = torch.randn(3, 5, 4, dtype=torch.float64)
    actual = soft_joint_recurrence_matrix(x, y, temperature=0.5)
    assert actual.shape == (3, 5, 5)
    from torchnlts.recurrence.matrix import soft_recurrence_matrix

    truncated_first = soft_recurrence_matrix(x[:, :5], temperature=0.5) * soft_recurrence_matrix(y, temperature=0.5)
    assert torch.allclose(actual, truncated_first)


def test_soft_joint_recurrence_matrix_rejects_batch_mismatch():
    unbatched = torch.randn(3)
    batched = torch.randn(2, 5, 1)
    other_batched = torch.randn(3, 5, 1)
    for x, y in ((unbatched, batched), (batched, unbatched), (batched, other_batched)):
        try:
            soft_joint_recurrence_matrix(x, y, temperature=0.5)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for mismatched shapes {tuple(x.shape)} vs {tuple(y.shape)}")


def test_soft_joint_recurrence_matrix_gradcheck():
    torch.manual_seed(4)
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, dtype=torch.float64, requires_grad=True)

    def soft(x, y):
        return soft_joint_recurrence_matrix(x, y, temperature=0.5)

    assert torch.autograd.gradcheck(soft, (x, y))
