"""Tests for torchnlts.recurrence.rqa.

Dual ground truth, for each of
recurrence_rate (eq. 11), determinism (eq. 13), dl_average (eq. 12),
dl_max, and dl_entropy:
  1. Paper: Marwan et al. (2007) -- checked via a hand-derived example.
  2. Julia: RecurrenceAnalysis.jl's `recurrencerate`/`determinism`/
     `dl_average`/`dl_max`/`dl_entropy` (and `rqa`'s LAM/TT/Vmax/VENTR/RTE
     for the vertical-line and recurrence-time measures), whose `theiler` keyword excludes
     `theiler` diagonals (`theiler=1` excludes only the LOI); it's
     related to this module's `theiler_window` by
     `theiler = theiler_window + 1`.
"""

import math

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix
from torchnlts.recurrence.rqa import (
    _recurrence_time_histogram,
    determinism,
    divergence,
    dl_average,
    dl_entropy,
    dl_max,
    laminarity,
    ratio,
    recurrence_rate,
    rt_entropy,
    trapping_time,
    vl_entropy,
    vl_max,
    windowed_rqa,
)


def test_recurrence_rate_hand_derived():
    recurrence = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ]
    )
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
        stdout = run_julia(julia_code)
        julia_rr = float(stdout.strip())
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
    with pytest.raises(ValueError):
        recurrence_rate(torch.zeros(3, 5, dtype=torch.bool))


def test_determinism_no_recurrence_is_zero_not_nan():
    # Matches RecurrenceAnalysis.jl: DET is 0.0, not NaN, when no line reaches lmin (even with zero recurrent points).
    assert torch.isclose(determinism(torch.eye(4, dtype=torch.bool)), torch.tensor(0.0))
    assert torch.isclose(determinism(torch.eye(2, dtype=torch.bool), theiler_window=1), torch.tensor(0.0))


def test_determinism_hand_derived():
    # Offsets +2 and -2 each have length 4 - 2 = 2 >= lmin; every off-diagonal recurrent point is in a line -> DET = 1.
    recurrence = torch.zeros(4, 4, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    for offset in (2, -2):
        diag = torch.diagonal(recurrence, offset=offset)
        diag.fill_(True)
    assert torch.isclose(determinism(recurrence, lmin=2), torch.tensor(1.0))


def test_determinism_short_lines_excluded_by_lmin():
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
    with pytest.raises(ValueError):
        determinism(torch.ones(4, 4))


def test_diagonal_line_measures_reject_negative_theiler_window():
    # Regression: a negative theiler_window counted the main diagonal as one giant line; now rejected.
    recurrence = torch.eye(4, dtype=torch.bool)
    for fn in (determinism, dl_average, dl_max, dl_entropy):
        with pytest.raises(ValueError):
            fn(recurrence, theiler_window=-1)


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
        stdout = run_julia(julia_code)
        julia_det = float(stdout.strip())
        torch_det = determinism(recurrence, lmin=lmin, theiler_window=theiler_window).item()
        assert torch.isclose(torch.tensor(torch_det), torch.tensor(julia_det), atol=1e-5), (
            theiler_window,
            lmin,
            torch_det,
            julia_det,
        )


def _no_qualifying_lines_recurrence():
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
    assert torch.isclose(dl_average(recurrence), torch.tensor(3.0))
    assert dl_max(recurrence) == 3


def test_dl_entropy_hand_derived():
    # Lines of length 2 and 3, one each (diagonal excluded): a fair coin, entropy = log(2).
    recurrence = torch.zeros(6, 6, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    recurrence[0, 2] = True
    recurrence[1, 3] = True
    recurrence[3, 0] = True
    recurrence[4, 1] = True
    recurrence[5, 2] = True
    assert torch.isclose(dl_entropy(recurrence), torch.log(torch.tensor(2.0)))


def test_dl_entropy_single_line_length_is_zero():
    # Offsets +-2 of a 5x5 matrix are two lines of the same length: a single occupied bin has zero entropy.
    recurrence = torch.zeros(5, 5, dtype=torch.bool)
    recurrence.fill_diagonal_(True)
    for offset in (2, -2):
        torch.diagonal(recurrence, offset=offset).fill_(True)
    assert dl_entropy(recurrence) == 0
    entropy = dl_entropy(torch.stack([recurrence, torch.zeros_like(recurrence)]))
    assert entropy[0] == 0 and torch.isnan(entropy[1])


def test_dl_average_max_entropy_empty_histogram_convention():
    # Matches RecurrenceAnalysis.jl: entropy of an empty distribution is NaN, not 0.
    recurrence = _no_qualifying_lines_recurrence()
    assert torch.isclose(dl_average(recurrence), torch.tensor(0.0))
    assert dl_max(recurrence) == 0
    assert torch.isnan(dl_entropy(recurrence))


def test_dl_max_lmin_exceeds_trajectory_length_no_crash():
    # Regression: dl_max's .amax() crashed on an empty histogram when lmin exceeds the longest possible line.
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
        stdout = run_julia(julia_code)
        julia_avg, julia_max, julia_entr = (float(v) for v in stdout.strip().splitlines())
        assert torch.isclose(
            dl_average(recurrence, lmin=lmin, theiler_window=theiler_window), torch.tensor(julia_avg), atol=1e-5
        )
        assert dl_max(recurrence, lmin=lmin, theiler_window=theiler_window).item() == julia_max
        assert torch.isclose(
            dl_entropy(recurrence, lmin=lmin, theiler_window=theiler_window), torch.tensor(julia_entr), atol=1e-5
        )


def _vertical_example():
    # Vertical lines per column after removing row j (theiler_window=0):
    #   0: [1-3], [5]   1: [0], [2-3]   2: [1], [3-4]   3: [0-2]   4: [0], [2]   5: [1-4]
    columns = [[1, 2, 3, 5], [0, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3], [0, 2], [1, 2, 3, 4, 5]]
    recurrence = torch.zeros(6, 6, dtype=torch.bool)
    for column, rows in enumerate(columns):
        recurrence[rows, column] = True
    return recurrence


def _entropy(counts):
    total = sum(counts)
    return -sum(count / total * math.log(count / total) for count in counts)


def test_vertical_line_measures_hand_derived():
    recurrence = _vertical_example()
    # Line lengths outside the window: five of length 1, two of 2, two of 3, one of 4 (19 points).
    assert torch.isclose(laminarity(recurrence), torch.tensor(14 / 19))
    assert torch.isclose(trapping_time(recurrence), torch.tensor(14 / 5))
    assert vl_max(recurrence).item() == 4
    assert torch.isclose(vl_entropy(recurrence), torch.tensor(_entropy([2, 2, 1])))
    assert torch.isclose(laminarity(recurrence, lmin=3), torch.tensor(10 / 19))
    assert torch.isclose(trapping_time(recurrence, lmin=3), torch.tensor(10 / 3))
    assert torch.isclose(laminarity(recurrence, lmin=1), torch.tensor(1.0))


def test_recurrence_time_entropy_hand_derived():
    # Times per column: gap + floor(previous length / 2) + floor(next length / 2)
    # gives 2, 2, 2, 1 (columns 0, 1, 2, 4); columns 3 and 5 have one line.
    assert torch.isclose(rt_entropy(_vertical_example()), torch.tensor(_entropy([3, 1])))
    # Entropy is symmetric under swapping the counts, so pin the histogram too.
    histogram = _recurrence_time_histogram(_vertical_example(), 0)
    assert histogram.tolist() == [0, 1, 3, 0, 0, 0, 0]


def test_vertical_measures_cut_lines_at_theiler_window():
    recurrence = torch.zeros(4, 4, dtype=torch.bool)
    recurrence[:, 1] = True
    assert vl_max(recurrence, lmin=1, theiler_window=0).item() == 2  # rows 2-3, row 0 alone
    assert vl_max(recurrence, lmin=1, theiler_window=1).item() == 1  # row 3 only
    assert vl_max(recurrence, lmin=1, theiler_window=2).item() == 0


def test_vertical_measures_empty_conventions():
    recurrence = torch.eye(5, dtype=torch.bool)
    assert laminarity(recurrence).item() == 0.0
    assert trapping_time(recurrence).item() == 0.0
    assert vl_max(recurrence).item() == 0
    assert torch.isnan(vl_entropy(recurrence))
    assert torch.isnan(rt_entropy(recurrence))
    assert vl_max(recurrence, lmin=9).item() == 0
    assert vl_max(torch.ones(1, 1, dtype=torch.bool)).item() == 0
    assert torch.isnan(rt_entropy(torch.ones(1, 1, dtype=torch.bool)))


def test_vertical_measures_batched():
    example = _vertical_example()
    batch = torch.stack([example, torch.zeros_like(example), example.T])
    for function in (laminarity, trapping_time, vl_max, vl_entropy, rt_entropy):
        values = function(batch)
        assert values.shape == (3,)
        for index in range(3):
            single = function(batch[index])
            assert torch.equal(values[index].isnan(), single.isnan())
            if not single.isnan():
                assert torch.isclose(values[index].double(), single.double())


def test_vertical_measures_reject_invalid_arguments():
    recurrence = _vertical_example()
    for function in (laminarity, trapping_time, vl_max, vl_entropy):
        with pytest.raises(ValueError):
            function(recurrence, lmin=0)
        with pytest.raises(ValueError):
            function(recurrence, lmin=1.5)
        with pytest.raises(ValueError):
            function(recurrence, theiler_window=-1)
        with pytest.raises(ValueError):
            function(recurrence.float())
    with pytest.raises(ValueError):
        rt_entropy(recurrence, theiler_window=-1)
    with pytest.raises(ValueError):
        laminarity(torch.zeros(3, 4, dtype=torch.bool))


def test_vertical_measures_match_recurrenceanalysis_jl():
    # Julia's standalone laminarity() ignores theiler in its numerator for
    # theiler != 1, so LAM is compared against rqa()[:LAM], which is consistent.
    torch.manual_seed(2)
    trajectory = torch.randn(40, dtype=torch.float64)
    epsilon = 0.6
    recurrence = hard_recurrence_matrix(trajectory, epsilon=epsilon)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    for theiler_window, julia_theiler, lmin in [(0, 1, 2), (2, 3, 2), (0, 1, 3), (1, 2, 1)]:
        julia_code = f"""
        using RecurrenceAnalysis
        x = [{values}]
        R = RecurrenceMatrix(x, {epsilon})
        r = rqa(R; theiler={julia_theiler}, lmin={lmin})
        for key in (:LAM, :TT, :Vmax, :VENTR, :RTE)
            println(r[key])
        end
        """
        stdout = run_julia(julia_code)
        lam, tt, vmax, ventr, rte = (float(v) for v in stdout.strip().splitlines())
        kwargs = dict(lmin=lmin, theiler_window=theiler_window)
        assert torch.isclose(laminarity(recurrence, **kwargs), torch.tensor(lam), atol=1e-9)
        assert torch.isclose(trapping_time(recurrence, **kwargs), torch.tensor(tt), atol=1e-9)
        assert vl_max(recurrence, **kwargs).item() == vmax
        assert torch.isclose(vl_entropy(recurrence, **kwargs), torch.tensor(ventr), atol=1e-9)
        assert torch.isclose(rt_entropy(recurrence, theiler_window=theiler_window), torch.tensor(rte), atol=1e-9)


def test_vertical_measures_vmap_and_default_dtype():
    batch = torch.stack([_vertical_example(), _vertical_example().T])
    for function in (determinism, laminarity, trapping_time, vl_max, vl_entropy, rt_entropy):
        mapped = torch.vmap(function)(batch)
        for index in range(2):
            assert torch.equal(mapped[index], function(batch[index]))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for function in (laminarity, trapping_time, vl_entropy, rt_entropy):
            assert function(batch).dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_line_measures_reject_fractional_arguments():
    recurrence = _vertical_example()
    for function in (determinism, laminarity, trapping_time, vl_max, vl_entropy, dl_average):
        with pytest.raises(ValueError):
            function(recurrence, theiler_window=1.5)
        with pytest.raises(ValueError):
            function(recurrence, lmin=1.5)
    with pytest.raises(ValueError):
        rt_entropy(recurrence, theiler_window=1.5)


def test_vertical_measures_longest_possible_line():
    # Regression guard: a vertical line of length T - 1 (the longest outside the identity line) must be counted.
    recurrence = torch.zeros(6, 6, dtype=torch.bool)
    recurrence[1:, 0] = True
    assert laminarity(recurrence).item() == pytest.approx(1.0)
    assert trapping_time(recurrence).item() == pytest.approx(5.0)
    assert vl_max(recurrence).item() == 5


def _banded_recurrence(num_points=5):
    return (
        torch.eye(num_points, dtype=torch.bool)
        | torch.diag(torch.ones(num_points - 1, dtype=torch.bool), 1)
        | torch.diag(torch.ones(num_points - 1, dtype=torch.bool), -1)
    )


def _julia_values(recurrence, expressions, julia_theiler=1, lmin=2):
    rows = ";".join(" ".join("1" if value else "0" for value in row) for row in recurrence.tolist())
    julia_code = f"""
    using RecurrenceAnalysis, SparseArrays
    R = RecurrenceMatrix(sparse(Bool[{rows}]))
    kw = (theiler={julia_theiler}, lmin={lmin})
    {expressions}
    """
    return [float(line) for line in run_julia(julia_code).split()]


def _random_recurrence(num_points=30, seed=0):
    torch.manual_seed(seed)
    trajectory = torch.cumsum(torch.randn(num_points, 1, dtype=torch.float64), dim=0)
    return hard_recurrence_matrix(trajectory, epsilon=1.5)


def test_ratio_divergence_hand_derived():
    recurrence = _banded_recurrence()
    # 8 off-diagonal points of 20 pairs; all in lines (length 4): DET = 1, RR = 0.4, Lmax = 4.
    assert ratio(recurrence).item() == pytest.approx(2.5)
    assert divergence(recurrence).item() == pytest.approx(0.25)
    assert torch.isnan(ratio(recurrence, theiler_window=1))
    assert divergence(recurrence, theiler_window=1).item() == math.inf


def test_ratio_divergence_degenerate_and_batched():
    empty = torch.eye(4, dtype=torch.bool)
    assert torch.isnan(ratio(empty))
    assert divergence(empty).item() == math.inf  # Lmax = 0 (Julia: 1.0 / 0 = Inf)
    isolated = _no_qualifying_lines_recurrence()
    assert ratio(isolated).item() == 0.0
    batch = torch.stack([_banded_recurrence(), empty.new_zeros(5, 5) | torch.eye(5, dtype=torch.bool)])
    assert ratio(batch).shape == (2,) and divergence(batch).shape == (2,)
    assert divergence(batch)[0].item() == pytest.approx(0.25) and divergence(batch)[1].item() == math.inf


def test_ratio_divergence_match_julia():
    recurrence = _random_recurrence()
    for theiler_window, lmin in ((0, 2), (2, 2), (0, 3)):
        kwargs = dict(lmin=lmin, theiler_window=theiler_window)
        julia_ratio, julia_div, julia_rqa_div = _julia_values(
            recurrence,
            "println(determinism(R; kw...) / recurrencerate(R; theiler=kw.theiler)); println(divergence(R; kw...)); println(rqa(R; kw...)[:DIV])",
            julia_theiler=theiler_window + 1,
            lmin=lmin,
        )
        assert ratio(recurrence, **kwargs).item() == pytest.approx(julia_ratio, rel=1e-5)
        assert divergence(recurrence, **kwargs).item() == pytest.approx(julia_div, rel=1e-6)
        assert julia_div == julia_rqa_div


def test_windowed_rqa_hand_derived():
    recurrence = _banded_recurrence()
    # width 3, step 2: windows start at 0 and 2; each is a 3x3 band (4 of 6 pairs, DET 1, Lmax 2).
    assert torch.allclose(windowed_rqa(recurrence, recurrence_rate, 3, step=2), torch.full((2,), 4 / 6))
    assert torch.equal(windowed_rqa(recurrence, dl_max, 3, step=2), torch.tensor([2, 2]))
    # step 1 gives 3 windows; width 4 step 2 drops the trailing partial window (1 window).
    assert windowed_rqa(recurrence, determinism, 3).shape == (3,)
    assert windowed_rqa(recurrence, determinism, 4, step=2).shape == (1,)
    assert windowed_rqa(recurrence, dl_max, 5).item() == dl_max(recurrence).item()


def test_windowed_rqa_places_windows_on_diagonal_blocks():
    # Regression: windows must be diagonal sub-blocks, not rows/columns of different offsets.
    recurrence = torch.zeros(6, 6, dtype=torch.bool)
    recurrence[3:, 3:] = True
    values = windowed_rqa(recurrence, recurrence_rate, 3, step=3, theiler_window=-1)
    assert torch.equal(values, torch.tensor([0.0, 1.0]))


def test_windowed_rqa_batched_and_kwargs():
    batch = torch.stack([_random_recurrence(seed=0), _random_recurrence(seed=1)])
    values = windowed_rqa(batch, determinism, 10, step=4, lmin=3, theiler_window=1)
    assert values.shape == (2, 6)
    for index in range(2):
        assert torch.equal(values[index], windowed_rqa(batch[index], determinism, 10, step=4, lmin=3, theiler_window=1))
    assert torch.equal(
        torch.vmap(lambda matrix: windowed_rqa(matrix, laminarity, 10, step=4))(batch),
        windowed_rqa(batch, laminarity, 10, step=4),
    )
    assert windowed_rqa(batch, dl_max, 10, step=4).dtype == torch.long


def test_windowed_rqa_validation():
    recurrence = _banded_recurrence()
    for window_size, step in ((1, 1), (6, 1), (3, 0), (3.5, 1), (3, 1.5)):
        with pytest.raises(ValueError):
            windowed_rqa(recurrence, determinism, window_size, step=step)
    with pytest.raises(ValueError):
        windowed_rqa(recurrence, "determinism", 3)
    with pytest.raises(ValueError):
        windowed_rqa(torch.zeros(3, 4, dtype=torch.bool), determinism, 2)


def test_windowed_rqa_matches_julia():
    recurrence = _random_recurrence(num_points=40)
    width, step = 12, 5
    expressions = (
        f"for f in (determinism, laminarity, dl_max); foreach(println, windowed(R, f, {width}, {step}; kw...)); end"
    )
    values = _julia_values(recurrence, expressions)
    num_windows = (40 - width) // step + 1
    for index, measure in enumerate((determinism, laminarity, dl_max)):
        expected = values[index * num_windows : (index + 1) * num_windows]
        got = windowed_rqa(recurrence, measure, width, step=step, lmin=2, theiler_window=0)
        assert got.shape == (num_windows,)
        assert torch.allclose(got.double(), torch.tensor(expected, dtype=torch.float64), atol=1e-9), measure


def test_windowed_rqa_forwards_each_kwarg():
    recurrence = torch.eye(5, dtype=torch.bool) | torch.diag(torch.ones(4, dtype=torch.bool), 1)
    assert torch.equal(windowed_rqa(recurrence, determinism, 3), torch.ones(3))
    assert torch.equal(windowed_rqa(recurrence, determinism, 3, lmin=3), torch.zeros(3))
    assert torch.equal(windowed_rqa(recurrence, determinism, 3, theiler_window=1), torch.zeros(3))


def test_windowed_rqa_block_orientation():
    # Regression: blocks must keep [row, column] order (a transposed block turns a vertical line into a row).
    recurrence = torch.zeros(5, 5, dtype=torch.bool)
    recurrence[1:, 0] = True
    assert windowed_rqa(recurrence, laminarity, 5).item() == 1.0
    assert windowed_rqa(recurrence.T, laminarity, 5).item() == 0.0


def test_divergence_follows_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        assert divergence(_banded_recurrence()).dtype == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_windowed_rqa_chunking_matches_single_call(monkeypatch):
    import torchnlts.recurrence.rqa as module

    batch = torch.stack([_random_recurrence(seed=0), _random_recurrence(seed=1)])
    expected = windowed_rqa(batch, determinism, 10, step=3)
    for budget in (1, 2 * 100 * 3, 2 * 100 * 4 + 1):  # one window, three, four per chunk
        monkeypatch.setattr(module, "_WINDOW_ELEMENT_BUDGET", budget)
        assert torch.equal(windowed_rqa(batch, determinism, 10, step=3), expected)
        assert torch.equal(torch.vmap(lambda matrix: windowed_rqa(matrix, determinism, 10, step=3))(batch), expected)


def test_windowed_rqa_empty_batch_and_chunk_calls(monkeypatch):
    import torchnlts.recurrence.rqa as rqa_module

    empty = windowed_rqa(torch.zeros(0, 5, 5, dtype=torch.bool), recurrence_rate, 3)
    assert empty.shape == (0, 3)
    monkeypatch.setattr(rqa_module, "_WINDOW_ELEMENT_BUDGET", 100)
    calls = []

    def recording(blocks):
        calls.append(blocks.shape[0])
        return blocks.sum(dim=(-2, -1))

    recurrence = torch.ones(2, 10, 10)
    windowed_rqa(recurrence, recording, 3)
    # 100 // (2 * 9) = 5 windows per call over 8 windows, times batch 2.
    assert calls == [10, 6]


def test_windowed_rqa_keeps_measure_gradient():
    matrix = torch.rand(7, 7, dtype=torch.float64, requires_grad=True)
    windowed_rqa(matrix, recurrence_rate, 3, step=2).sum().backward()
    expected = torch.zeros_like(matrix)
    for start in range(0, 5, 2):
        block = matrix.detach()[start : start + 3, start : start + 3].clone().requires_grad_()
        recurrence_rate(block).backward()
        expected[start : start + 3, start : start + 3] += block.grad
    torch.testing.assert_close(matrix.grad, expected)
