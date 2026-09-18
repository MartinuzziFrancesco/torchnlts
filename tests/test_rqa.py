"""Tests for torchnlts.recurrence.rqa.

Dual ground truth, for each of
recurrence_rate (eq. 11), determinism (eq. 13), dl_average (eq. 12),
dl_max, and dl_entropy:
  1. Paper: Marwan et al. (2007) -- checked via a hand-derived example.
  2. Julia: RecurrenceAnalysis.jl's `recurrencerate`/`determinism`/
     `dl_average`/`dl_max`/`dl_entropy`, whose `theiler` keyword excludes
     `theiler` diagonals (`theiler=1` excludes only the LOI); it's
     related to this module's `theiler_window` by
     `theiler = theiler_window + 1`.
"""

import subprocess

import torch

from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix
from torchnlts.recurrence.rqa import determinism, dl_average, dl_entropy, dl_max, recurrence_rate

JULIA_PROJECT = "juliaenv"


def test_recurrence_rate_hand_derived():
    # 3x3 recurrence matrix, 5 of 9 entries True (diagonal always True).
    recurrence = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ]
    )
    # No Theiler exclusion: RR = 5/9.
    assert torch.isclose(recurrence_rate(recurrence, theiler_window=-1), torch.tensor(5 / 9))
    # Default (exclude diagonal): 2 recurrent pairs out of 6 off-diagonal entries.
    assert torch.isclose(recurrence_rate(recurrence), torch.tensor(2 / 6))


def test_recurrence_rate_batched_and_soft():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 10, 1, dtype=torch.float64)
    hard = hard_recurrence_matrix(trajectory, epsilon=0.5)
    soft = soft_recurrence_matrix(trajectory, temperature=0.5)
    rr_hard = recurrence_rate(hard)
    rr_soft = recurrence_rate(soft)
    assert rr_hard.shape == (4,)
    assert rr_soft.shape == (4,)
    assert torch.all((rr_hard >= 0) & (rr_hard <= 1))
    assert torch.all((rr_soft >= 0) & (rr_soft <= 1))


def test_recurrence_rate_soft_is_differentiable():
    torch.manual_seed(1)
    trajectory = torch.randn(10, dtype=torch.float64, requires_grad=True)
    soft = soft_recurrence_matrix(trajectory, temperature=0.5)
    recurrence_rate(soft).backward()
    assert trajectory.grad is not None


def test_recurrence_rate_gradcheck():
    torch.manual_seed(1)
    trajectory = torch.randn(6, dtype=torch.float64, requires_grad=True)

    def rr(x):
        return recurrence_rate(soft_recurrence_matrix(x, temperature=0.5))

    assert torch.autograd.gradcheck(rr, (trajectory,))


def test_recurrence_rate_matches_recurrenceanalysis_jl():
    torch.manual_seed(2)
    trajectory = torch.randn(30, dtype=torch.float64)
    epsilon = 0.75
    recurrence = hard_recurrence_matrix(trajectory, epsilon=epsilon)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    for theiler_window, julia_theiler in ((0, 1), (2, 3)):
        julia_code = f"""
        using RecurrenceAnalysis
        x = [{values}]
        R = RecurrenceMatrix(x, {epsilon})
        println(recurrencerate(R; theiler={julia_theiler}))
        """
        result = subprocess.run(
            ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        julia_rr = float(result.stdout.strip())
        torch_rr = recurrence_rate(recurrence, theiler_window=theiler_window).item()
        assert torch.isclose(torch.tensor(torch_rr), torch.tensor(julia_rr), atol=1e-9), (
            theiler_window,
            torch_rr,
            julia_rr,
        )


def test_recurrence_rate_theiler_window_excludes_everything_is_nan_not_crash():
    recurrence = torch.eye(2, dtype=torch.bool)
    result = recurrence_rate(recurrence, theiler_window=1)
    assert torch.isnan(result)


def test_recurrence_rate_rejects_non_square_matrix():
    try:
        recurrence_rate(torch.zeros(3, 5, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-square matrix")


def test_determinism_no_recurrence_is_zero_not_nan():
    # Matches RecurrenceAnalysis.jl exactly (verified directly): DET is
    # 0.0, not NaN, whenever no line reaches lmin -- true even when that
    # also means zero recurrent points at all (numerator and denominator
    # both 0), since Julia short-circuits on the histogram rather than
    # dividing. Covers both an empty *data-dependent* denominator (no
    # off-diagonal recurrent points) and a fully theiler_window-excluded
    # matrix (T <= theiler_window).
    assert torch.isclose(determinism(torch.eye(4, dtype=torch.bool)), torch.tensor(0.0))
    assert torch.isclose(determinism(torch.eye(2, dtype=torch.bool), theiler_window=1), torch.tensor(0.0))


def test_determinism_hand_derived():
    # 4x4: offsets +2 and -2 each have length 4 - 2 = 2, both >= lmin=2.
    # R has no other recurrent points beyond the diagonal (excluded by the
    # default theiler_window=0) and these two lines, so every off-diagonal
    # recurrent point belongs to a qualifying line -> DET = 1.0.
    recurrence = torch.zeros(4, 4, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    for offset in (2, -2):
        diag = torch.diagonal(recurrence, offset=offset)
        diag.fill_(True)
    assert torch.isclose(determinism(recurrence, lmin=2), torch.tensor(1.0))


def test_determinism_short_lines_excluded_by_lmin():
    # A single recurrent off-diagonal point (length-1 line) and nothing
    # else: with lmin=2 it never counts, giving DET = 0.
    recurrence = torch.zeros(4, 4, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    recurrence[0, 2] = True
    recurrence[2, 0] = True
    assert torch.isclose(determinism(recurrence, lmin=2), torch.tensor(0.0))


def test_determinism_batched():
    torch.manual_seed(5)
    trajectory = torch.randn(3, 20, 1, dtype=torch.float64)
    recurrence = hard_recurrence_matrix(trajectory, epsilon=0.6)
    result = determinism(recurrence)
    assert result.shape == (3,)
    assert torch.all((result >= 0) & (result <= 1))


def test_determinism_rejects_non_bool_input():
    try:
        determinism(torch.ones(4, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-boolean recurrence matrix")


def test_diagonal_line_measures_reject_negative_theiler_window():
    # Regression: a negative theiler_window (recurrence_rate's documented
    # "exclude nothing") would otherwise count the always-recurrent main
    # diagonal itself as one giant diagonal line -- rejected explicitly
    # rather than silently producing a dubious, unvalidated result.
    recurrence = torch.eye(4, dtype=torch.bool)
    for fn in (determinism, dl_average, dl_max, dl_entropy):
        try:
            fn(recurrence, theiler_window=-1)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError from {fn.__name__} for a negative theiler_window")


def test_determinism_matches_recurrenceanalysis_jl():
    torch.manual_seed(2)
    trajectory = torch.randn(30, dtype=torch.float64)
    epsilon = 0.75
    recurrence = hard_recurrence_matrix(trajectory, epsilon=epsilon)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    cases = [(0, 1, 2), (2, 3, 2), (0, 1, 3)]
    for theiler_window, julia_theiler, lmin in cases:
        julia_code = f"""
        using RecurrenceAnalysis
        x = [{values}]
        R = RecurrenceMatrix(x, {epsilon})
        println(determinism(R; theiler={julia_theiler}, lmin={lmin}))
        """
        result = subprocess.run(
            ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        julia_det = float(result.stdout.strip())
        torch_det = determinism(recurrence, lmin=lmin, theiler_window=theiler_window).item()
        assert torch.isclose(torch.tensor(torch_det), torch.tensor(julia_det), atol=1e-5), (
            theiler_window,
            lmin,
            torch_det,
            julia_det,
        )


def _no_qualifying_lines_recurrence():
    # Only the diagonal (excluded by default theiler_window=0) and two
    # isolated length-1 off-diagonal points -- no line reaches lmin=2.
    recurrence = torch.zeros(4, 4, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    recurrence[0, 2] = True
    recurrence[2, 0] = True
    return recurrence


def test_dl_average_and_max_hand_derived():
    # 5x5: offsets +-2 each have length 5 - 2 = 3.
    recurrence = torch.zeros(5, 5, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    for offset in (2, -2):
        torch.diagonal(recurrence, offset=offset).fill_(True)
    # Two length-3 lines (offsets +-2), nothing else -> average == max == 3.
    assert torch.isclose(dl_average(recurrence), torch.tensor(3.0))
    assert dl_max(recurrence) == 3


def test_dl_entropy_hand_derived():
    # 6x6: one length-2 line (offset +2, rows 0-1) and one length-3 line
    # (offset -3, rows 3-5), nothing else recurrent besides the diagonal
    # (excluded by default theiler_window=0). Two lines, one of each
    # length -> length-length distribution is a fair coin: entropy = log(2).
    recurrence = torch.zeros(6, 6, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    recurrence[0, 2] = True
    recurrence[1, 3] = True
    recurrence[3, 0] = True
    recurrence[4, 1] = True
    recurrence[5, 2] = True
    assert torch.isclose(dl_entropy(recurrence), torch.log(torch.tensor(2.0)))


def test_dl_average_max_entropy_empty_histogram_convention():
    # Matches RecurrenceAnalysis.jl: average/max default to 0 when no line
    # reaches lmin, but entropy of an empty distribution is NaN, not 0.
    recurrence = _no_qualifying_lines_recurrence()
    assert torch.isclose(dl_average(recurrence), torch.tensor(0.0))
    assert dl_max(recurrence) == 0
    assert torch.isnan(dl_entropy(recurrence))


def test_dl_max_lmin_exceeds_trajectory_length_no_crash():
    # Regression: histogram[..., lmin:] is empty whenever lmin exceeds the
    # longest possible diagonal line length, and amax has no identity for
    # an empty reduction -- dl_average/dl_entropy handle this fine via
    # .sum(), but dl_max's .amax() crashed.
    assert dl_max(torch.ones(1, 1, dtype=torch.bool)) == 0
    batched = torch.ones(3, 4, 4, dtype=torch.bool)
    assert torch.equal(dl_max(batched, lmin=10), torch.zeros(3, dtype=torch.long))


def test_dl_average_max_entropy_batched():
    torch.manual_seed(6)
    trajectory = torch.randn(3, 20, 1, dtype=torch.float64)
    recurrence = hard_recurrence_matrix(trajectory, epsilon=0.6)
    assert dl_average(recurrence).shape == (3,)
    assert dl_max(recurrence).shape == (3,)
    assert dl_entropy(recurrence).shape == (3,)


def test_dl_average_max_entropy_match_recurrenceanalysis_jl():
    torch.manual_seed(2)
    trajectory = torch.randn(30, dtype=torch.float64)
    epsilon = 0.75
    recurrence = hard_recurrence_matrix(trajectory, epsilon=epsilon)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    cases = [(0, 1, 2), (2, 3, 4)]
    for theiler_window, julia_theiler, lmin in cases:
        julia_code = f"""
        using RecurrenceAnalysis
        x = [{values}]
        R = RecurrenceMatrix(x, {epsilon})
        println(dl_average(R; theiler={julia_theiler}, lmin={lmin}))
        println(dl_max(R; theiler={julia_theiler}, lmin={lmin}))
        println(dl_entropy(R; theiler={julia_theiler}, lmin={lmin}))
        """
        result = subprocess.run(
            ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        julia_avg, julia_max, julia_entr = (float(v) for v in result.stdout.strip().splitlines())
        assert torch.isclose(dl_average(recurrence, lmin=lmin, theiler_window=theiler_window), torch.tensor(julia_avg), atol=1e-5)
        assert dl_max(recurrence, lmin=lmin, theiler_window=theiler_window).item() == julia_max
        assert torch.isclose(dl_entropy(recurrence, lmin=lmin, theiler_window=theiler_window), torch.tensor(julia_entr), atol=1e-5)
