"""Tests for the rate-based and relative thresholds of the hard recurrence matrices.

Ground truth: hand-derived examples and RecurrenceAnalysis.jl's
``GlobalRecurrenceRate``, ``LocalRecurrenceRate`` and ``RecurrenceThresholdScaled``.
"""

import subprocess

import pytest
import torch

from torchnlts.recurrence.cross_recurrence import hard_cross_recurrence_matrix
from torchnlts.recurrence.joint_recurrence import hard_joint_recurrence_matrix
from torchnlts.recurrence.matrix import _nan_quantile, hard_recurrence_matrix

JULIA_PROJECT = "juliaenv"


def _julia_matrix(expression: str, *arrays: torch.Tensor) -> torch.Tensor:
    """Evaluate ``expression`` (using ``x``, ``y``) in Julia and return the boolean matrix."""
    lines = ["using RecurrenceAnalysis, Statistics"]
    for name, array in zip("xy", arrays):
        rows = array.reshape(array.shape[0], -1)
        text = "; ".join(" ".join(f"{v:.17g}" for v in row) for row in rows.tolist())
        lines.append(f"{name} = StateSpaceSet([{text}])")
    lines.append(f"R = Matrix({expression})")
    lines.append('for row in eachrow(R) println(join(Int.(row), ",")) end')
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", "\n".join(lines)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return torch.tensor(
        [[int(v) for v in line.split(",")] for line in result.stdout.strip().splitlines()], dtype=torch.bool
    )


def _upper_mirror(matrix: torch.Tensor) -> torch.Tensor:
    """Upper triangle (with diagonal) mirrored below it, as RecurrenceAnalysis.jl
    stores a self-recurrence matrix (``Symmetric(..., :U)``)."""
    upper = torch.triu(matrix)
    return upper | torch.triu(matrix, diagonal=1).T


@pytest.fixture
def trajectory():
    torch.manual_seed(3)
    return torch.randn(40, 2, dtype=torch.float64)


def test_global_rate_hand_derived():
    # Distances (upper triangle): 1, 3, 2 plus nine entries of the 3x3 matrix
    # sorted: 0 0 0 1 1 2 2 3 3. Rate 5/9 interpolates at index 8*5/9, so
    # the quantile lands between 1 and 2 -> only distances <= 1 recur.
    trajectory = torch.tensor([0.0, 1.0, 3.0])
    actual = hard_recurrence_matrix(trajectory, recurrence_rate=0.5)
    expected = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    assert torch.equal(actual, expected)
    # Rate 1: threshold = max distance, everything recurs.
    assert hard_recurrence_matrix(trajectory, recurrence_rate=1.0).all()


def test_local_rate_hand_derived():
    # Points 0, 1, 3, 7. Column distances sorted (incl. self): col0: 0 1 3 7,
    # col1: 0 1 2 6, col2: 0 2 3 4, col3: 0 4 6 7. local_rate=0.25 -> one
    # neighbour besides self per column: the nearest one.
    trajectory = torch.tensor([0.0, 1.0, 3.0, 7.0])
    actual = hard_recurrence_matrix(trajectory, local_rate=0.25)
    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, actual.T)


def test_local_rate_columns_have_fixed_neighbours(trajectory):
    actual = hard_recurrence_matrix(trajectory, local_rate=0.25)
    # 0.25 * 40 = 10 neighbours besides the diagonal, per column.
    assert torch.equal(actual.sum(dim=0), torch.full((40,), 11))


def test_scaled_hand_derived():
    trajectory = torch.tensor([0.0, 1.0, 4.0])
    # distances 1, 4, 3: max 4, mean 8/3, std of the 9 entries (diagonal included)
    expected_max = hard_recurrence_matrix(trajectory, 2.0)
    assert torch.equal(hard_recurrence_matrix(trajectory, 0.5, scale="max"), expected_max)
    assert torch.equal(hard_recurrence_matrix(trajectory, 0.75, scale="mean"), hard_recurrence_matrix(trajectory, 2.0))
    entries = torch.tensor([0.0, 1, 4, 1, 0, 3, 4, 3, 0])
    epsilon = 1.0 * entries.std().item()
    assert torch.equal(
        hard_recurrence_matrix(trajectory, 1.0, scale="std"), hard_recurrence_matrix(trajectory, epsilon)
    )


def test_std_scale_uses_sample_std():
    # Entries 0, 1, 1, 0: sample std 1/sqrt(3) = 0.577, population std 0.5.
    # Ratio 1.8 gives a threshold above 1 only for the sample std.
    assert hard_recurrence_matrix(torch.tensor([0.0, 1.0]), 1.8, scale="std").all()
    assert not hard_recurrence_matrix(torch.tensor([0.0, 1.0]), 1.7, scale="std")[0, 1]


def test_achieved_rate_close_to_target(trajectory):
    matrix = hard_recurrence_matrix(trajectory, recurrence_rate=0.1)
    assert abs(matrix.double().mean().item() - 0.1) < 0.01


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.37, 0.9])
def test_global_rate_matches_julia(trajectory, rate):
    expected = _julia_matrix(f"RecurrenceMatrix(x, GlobalRecurrenceRate({rate}))", trajectory)
    assert torch.equal(hard_recurrence_matrix(trajectory, recurrence_rate=rate), expected)


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.3, 0.5])
def test_local_rate_matches_julia(trajectory, rate):
    # Deviation: RecurrenceAnalysis.jl thresholds column j with the j-th quantile
    # but stores only the upper triangle (mirrored), contradicting its own
    # documentation of exactly k entries per column. Here every column keeps its
    # k neighbours (asymmetric); the Julia matrix is this one's mirrored upper triangle.
    expected = _julia_matrix(f"RecurrenceMatrix(x, LocalRecurrenceRate({rate}))", trajectory)
    assert torch.equal(_upper_mirror(hard_recurrence_matrix(trajectory, local_rate=rate)), expected)


@pytest.mark.parametrize("scale", ["max", "mean", "std"])
def test_scaled_matches_julia(trajectory, scale):
    function = {"max": "maximum", "mean": "mean", "std": "std"}[scale]
    expected = _julia_matrix(f"RecurrenceMatrix(x, RecurrenceThresholdScaled(0.3, {function}))", trajectory)
    assert torch.equal(hard_recurrence_matrix(trajectory, 0.3, scale=scale), expected)


def test_chebyshev_global_rate_matches_julia(trajectory):
    expected = _julia_matrix("RecurrenceMatrix(x, GlobalRecurrenceRate(0.2); metric = Chebyshev())", trajectory)
    assert torch.equal(hard_recurrence_matrix(trajectory, norm_order=float("inf"), recurrence_rate=0.2), expected)


def test_cross_matches_julia(trajectory):
    other = trajectory[:25] * 0.7 + 0.1
    for option, rt in (
        ({"recurrence_rate": 0.2}, "GlobalRecurrenceRate(0.2)"),
        ({"local_rate": 0.2}, "LocalRecurrenceRate(0.2)"),
        ({"epsilon": 0.3, "scale": "max"}, "RecurrenceThresholdScaled(0.3, maximum)"),
        ({"epsilon": 0.3, "scale": "mean"}, "RecurrenceThresholdScaled(0.3, mean)"),
    ):
        expected = _julia_matrix(f"CrossRecurrenceMatrix(x, y, {rt})", trajectory, other)
        assert torch.equal(hard_cross_recurrence_matrix(trajectory, other, **option), expected), option


def test_joint_matches_julia_with_unequal_lengths(trajectory):
    other = torch.randn(30, 3, dtype=torch.float64)
    for option, rt in (
        ({"recurrence_rate": 0.2}, "GlobalRecurrenceRate(0.2)"),
        ({"local_rate": 0.2}, "LocalRecurrenceRate(0.2)"),
        ({"epsilon": 0.4, "scale": "max"}, "RecurrenceThresholdScaled(0.4, maximum)"),
    ):
        # Julia truncates, then thresholds each trajectory.
        expected = _julia_matrix(f"JointRecurrenceMatrix(x, y, {rt})", trajectory, other)
        actual = hard_joint_recurrence_matrix(trajectory, other, **option)
        if "local_rate" in option:  # see test_local_rate_matches_julia
            first = _upper_mirror(hard_recurrence_matrix(trajectory[:30], **option))
            second = _upper_mirror(hard_recurrence_matrix(other, **option))
            assert torch.equal(first & second, expected)
            assert torch.equal(
                actual, hard_recurrence_matrix(trajectory[:30], **option) & hard_recurrence_matrix(other, **option)
            )
        else:
            assert torch.equal(actual, expected), option


def test_batched_rows_have_own_threshold():
    torch.manual_seed(4)
    batch = (
        torch.randn(3, 20, 2, dtype=torch.float64)
        * torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)[:, None, None]
    )
    for option in ({"recurrence_rate": 0.15}, {"local_rate": 0.15}, {"epsilon": 0.3, "scale": "std"}):
        batched = hard_recurrence_matrix(batch, **option)
        for index in range(3):
            assert torch.equal(batched[index], hard_recurrence_matrix(batch[index], **option))
    # Rows differ only by scale, so a rate threshold gives identical matrices.
    matrices = hard_recurrence_matrix(
        batch / torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)[:, None, None], recurrence_rate=0.15
    )
    assert torch.equal(matrices, hard_recurrence_matrix(batch, recurrence_rate=0.15))


def test_nonfinite_rows_are_ignored_by_the_threshold():
    torch.manual_seed(5)
    clean = torch.randn(20, 2, dtype=torch.float64)
    dirty = torch.cat([clean, torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)])
    for option in ({"recurrence_rate": 0.2}, {"epsilon": 0.5, "scale": "max"}, {"epsilon": 0.5, "scale": "mean"}):
        matrix = hard_recurrence_matrix(dirty, **option)
        assert torch.equal(matrix[:20, :20], hard_recurrence_matrix(clean, **option)), option
        assert not matrix[20].any() and not matrix[:, 20].any()
    local = hard_recurrence_matrix(dirty, local_rate=0.2)
    assert not local[:, 20].any() and not local[20].any()
    all_bad = torch.full((5, 2), float("nan"), dtype=torch.float64)
    assert not hard_recurrence_matrix(all_bad, recurrence_rate=0.5).any()
    assert not hard_recurrence_matrix(all_bad, local_rate=0.5).any()
    assert not hard_recurrence_matrix(all_bad, 1.0, scale="std").any()


def test_vmap_over_batch():
    torch.manual_seed(6)
    batch = torch.randn(3, 15, 2, dtype=torch.float64)
    batch[1, 4] = float("nan")
    for option in ({"recurrence_rate": 0.2}, {"local_rate": 0.2}, {"epsilon": 0.4, "scale": "mean"}):
        mapped = torch.vmap(lambda row: hard_recurrence_matrix(row, **option))(batch)
        assert torch.equal(mapped, hard_recurrence_matrix(batch, **option))


@pytest.mark.parametrize(
    "option",
    [
        {},
        {"epsilon": 1.0, "recurrence_rate": 0.1},
        {"recurrence_rate": 0.1, "local_rate": 0.1},
        {"recurrence_rate": 0.0},
        {"recurrence_rate": 1.5},
        {"local_rate": -0.1},
        {"local_rate": float("nan")},
        {"recurrence_rate": 0.1, "scale": "max"},
        {"scale": "max"},
        {"epsilon": 1.0, "scale": "median"},
    ],
)
def test_invalid_options_raise(option):
    trajectory = torch.randn(10, 2)
    with pytest.raises(ValueError):
        hard_recurrence_matrix(trajectory, **option)
    with pytest.raises(ValueError):
        hard_cross_recurrence_matrix(trajectory, trajectory, **option)
    with pytest.raises(ValueError):
        hard_joint_recurrence_matrix(trajectory, trajectory, **option)


def test_joint_other_epsilon_conflicts_with_rate():
    trajectory = torch.randn(10, 2)
    with pytest.raises(ValueError):
        hard_joint_recurrence_matrix(trajectory, trajectory, other_epsilon=0.5, recurrence_rate=0.1)


def test_float32_and_integer_input():
    trajectory = torch.tensor([0, 1, 3, 7])
    assert hard_recurrence_matrix(trajectory, local_rate=0.25).dtype == torch.bool
    assert hard_recurrence_matrix(torch.randn(20, 2), recurrence_rate=0.3).dtype == torch.bool


@pytest.mark.parametrize("factor", [1e20, 1e-25])
def test_thresholds_are_scale_invariant_in_float32(factor):
    # Regression: sums and squares of unscaled distances overflowed or underflowed.
    trajectory = torch.tensor([0.0, 1.0, 4.0])
    for option in (
        {"epsilon": 0.8, "scale": "std"},
        {"epsilon": 0.8, "scale": "mean"},
        {"epsilon": 0.8, "scale": "max"},
        {"recurrence_rate": 0.5},
        {"local_rate": 0.4},
    ):
        assert torch.equal(
            hard_recurrence_matrix(trajectory * factor, **option), hard_recurrence_matrix(trajectory, **option)
        ), option


def test_std_ignores_nan_entries():
    # Regression: NaN entries counted in the std's denominator.
    clean = torch.tensor([0.0, 1.0, 4.0, 2.5], dtype=torch.float64)
    dirty = torch.cat([clean, torch.tensor([float("nan")], dtype=torch.float64)])
    for ratio in (0.5, 0.8, 0.9, 1.3):
        assert torch.equal(
            hard_recurrence_matrix(dirty, ratio, scale="std")[:4, :4], hard_recurrence_matrix(clean, ratio, scale="std")
        )


def test_nan_padding_preserves_local_rate_matrix():
    # Regression: the diagonal correction used the padded length.
    valid = torch.tensor([0.0, 1.0, 3.0, 7.0])
    padded = torch.cat([valid, torch.full((20,), float("nan"))])
    assert torch.equal(
        hard_recurrence_matrix(padded, local_rate=0.25)[:4, :4], hard_recurrence_matrix(valid, local_rate=0.25)
    )
    assert hard_recurrence_matrix(valid, local_rate=0.25).sum(dim=0).tolist() == [2, 2, 2, 2]
    other = torch.randn(4, 2)
    other_padded = torch.cat([other, torch.full((20, 2), float("nan"))])
    joint = hard_joint_recurrence_matrix(padded, other_padded, local_rate=0.25)
    assert torch.equal(joint[:4, :4], hard_joint_recurrence_matrix(valid, other, local_rate=0.25))


def test_quantile_index_arithmetic_is_float64():
    # Regression: float32 index arithmetic picks the wrong neighbour near an
    # order-statistic boundary: aleph = 1000000.99 rounds to 1000001 in float32.
    count = 2_000_001
    values = torch.arange(count, dtype=torch.float64)
    probability = (1_000_000 - 0.01) / (count - 1)
    result = _nan_quantile(values, probability, dim=-1)
    assert abs(result.item() - 999_999.99) < 1e-3


@pytest.mark.parametrize("value", [1j, torch.tensor(0.5 + 0j), 10**400, "a", None])
def test_rate_conversion_errors_are_value_errors(value):
    with pytest.raises(ValueError):
        hard_recurrence_matrix(torch.randn(5), recurrence_rate=value)
