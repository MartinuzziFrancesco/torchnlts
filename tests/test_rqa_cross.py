"""RQA measures on cross-recurrence matrices: ``theiler_window=-1`` and rectangular ``[T, T']`` input.

Ground truth: hand-derived matrices, a brute-force loop implementation, and RecurrenceAnalysis.jl's
``CrossRecurrenceMatrix`` with ``rqa(...; theiler=0)`` (rows index the first series, columns the second, in both
libraries; vertical lines run along rows within a column).
"""

import math
from collections import Counter

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.recurrence import (
    determinism,
    divergence,
    dl_average,
    dl_entropy,
    dl_max,
    hard_cross_recurrence_matrix,
    laminarity,
    ratio,
    recurrence_rate,
    rt_entropy,
    trapping_time,
    vl_entropy,
    vl_max,
    windowed_rqa,
)

LINE_MEASURES = (determinism, dl_average, dl_max, dl_entropy, laminarity, trapping_time, vl_max, vl_entropy)
WINDOW_MEASURES = (*LINE_MEASURES, rt_entropy, ratio, divergence)


def _hand_matrix():
    return torch.tensor(
        [
            [1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )


def _runs(values):
    lengths, current = [], 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    return [*lengths, current] if current else lengths


def _entropy(lengths):
    if not lengths:
        return math.nan
    counts = Counter(lengths)
    return -sum(count / len(lengths) * math.log(count / len(lengths)) for count in counts.values())


def _brute_force(matrix, lmin=2):
    rows = matrix.tolist()
    num_rows, num_columns = len(rows), len(rows[0])
    points = sum(map(sum, rows))
    diagonal, vertical, times = [], [], []
    for offset in range(-(num_rows - 1), num_columns):
        diagonal += _runs(rows[i][i + offset] for i in range(num_rows) if 0 <= i + offset < num_columns)
    for column in range(num_columns):
        values = [rows[i][column] for i in range(num_rows)]
        vertical += _runs(values)
        lines, start = [], None
        for index, value in enumerate([*values, False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                lines.append((start, index - 1))
                start = None
        for (first_start, first_end), (second_start, second_end) in zip(lines, lines[1:]):
            half_first = (first_end - first_start + 1) // 2
            half_second = (second_end - second_start + 1) // 2
            times.append(second_start - first_end - 1 + half_first + half_second)
    diagonal = [length for length in diagonal if length >= lmin]
    vertical = [length for length in vertical if length >= lmin]
    return {
        "determinism": sum(diagonal) / max(points, 1),
        "dl_average": sum(diagonal) / max(len(diagonal), 1),
        "dl_max": max(diagonal, default=0),
        "dl_entropy": _entropy(diagonal),
        "laminarity": sum(vertical) / max(points, 1),
        "trapping_time": sum(vertical) / max(len(vertical), 1),
        "vl_max": max(vertical, default=0),
        "vl_entropy": _entropy(vertical),
        "rt_entropy": _entropy(times),
        "recurrence_rate": points / (num_rows * num_columns),
    }


def _library(matrix, lmin=2):
    values = {function.__name__: function(matrix, lmin=lmin, theiler_window=-1) for function in LINE_MEASURES}
    values["rt_entropy"] = rt_entropy(matrix, theiler_window=-1)
    values["recurrence_rate"] = recurrence_rate(matrix, theiler_window=-1)
    return values


def test_full_main_diagonal_is_one_line():
    recurrence = torch.eye(6, dtype=torch.bool)
    assert dl_max(recurrence, theiler_window=-1).item() == 6
    assert determinism(recurrence, theiler_window=-1).item() == 1.0
    assert dl_max(recurrence, theiler_window=0).item() == 0


def test_rectangular_hand_derived():
    matrix = _hand_matrix()
    kwargs = dict(theiler_window=-1)
    # 10 recurrent points; diagonals: lengths 3 (offset 0), 4 (offset 1), and three singletons.
    assert recurrence_rate(matrix, **kwargs).item() == pytest.approx(10 / 24)
    assert determinism(matrix, **kwargs).item() == pytest.approx(0.7)
    assert dl_average(matrix, **kwargs).item() == pytest.approx(3.5)
    assert dl_max(matrix, **kwargs).item() == 4
    assert dl_entropy(matrix, **kwargs).item() == pytest.approx(math.log(2))
    # Vertical lines (along rows within a column): lengths 2, 2, 3 for columns 1, 2, 5.
    assert laminarity(matrix, **kwargs).item() == pytest.approx(0.7)
    assert trapping_time(matrix, **kwargs).item() == pytest.approx(7 / 3)
    assert vl_max(matrix, **kwargs).item() == 3
    assert vl_entropy(matrix, **kwargs).item() == pytest.approx(-(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
    assert divergence(matrix, **kwargs).item() == pytest.approx(0.25)
    assert ratio(matrix, **kwargs).item() == pytest.approx(0.7 / (10 / 24))
    assert dl_max(matrix.T, **kwargs).item() == 4


@pytest.mark.parametrize("shape", [(7, 11), (11, 7), (9, 9), (1, 5), (5, 1)])
@pytest.mark.parametrize("lmin", [1, 2, 3])
def test_matches_brute_force(shape, lmin):
    generator = torch.Generator().manual_seed(sum(shape) * 10 + lmin)
    batch = torch.rand(3, *shape, generator=generator) < 0.45
    single = _library(batch[0], lmin)
    for name, value in _brute_force(batch[0], lmin).items():
        assert single[name].item() == pytest.approx(value, nan_ok=True), (name, shape, lmin)
    batched = _library(batch, lmin)
    for index in range(3):
        for name, value in _brute_force(batch[index], lmin).items():
            assert batched[name][index].item() == pytest.approx(value, nan_ok=True), (name, shape, lmin)


def test_negative_windows_are_equivalent():
    matrix = torch.rand(6, 8, generator=torch.Generator().manual_seed(0)) < 0.5
    for function in LINE_MEASURES:
        torch.testing.assert_close(
            function(matrix, theiler_window=-1), function(matrix, theiler_window=-7), equal_nan=True
        )


def test_rectangular_matrix_with_theiler_band_raises():
    matrix = _hand_matrix()
    for function in (*WINDOW_MEASURES, recurrence_rate):
        for window in (0, 2):
            with pytest.raises(ValueError, match="theiler_window=-1"):
                function(matrix, theiler_window=window)
        with pytest.raises(ValueError, match="theiler_window=-1"):
            function(matrix)


def test_windowed_rqa_forwards_negative_window():
    matrix = torch.eye(8, dtype=torch.bool)
    assert torch.equal(windowed_rqa(matrix, dl_max, 4, step=2, theiler_window=-1), torch.full((3,), 4))


def test_vmap_on_rectangular_input():
    batch = torch.rand(3, 6, 9, generator=torch.Generator().manual_seed(1)) < 0.5
    for function in (determinism, dl_max, laminarity, vl_max, rt_entropy):
        mapped = torch.vmap(lambda matrix, function=function: function(matrix, theiler_window=-1))(batch)
        for index in range(3):
            torch.testing.assert_close(mapped[index], function(batch[index], theiler_window=-1), equal_nan=True)


JULIA_NAMES = {
    "RR": "recurrence_rate",
    "DET": "determinism",
    "L": "dl_average",
    "Lmax": "dl_max",
    "ENTR": "dl_entropy",
    "LAM": "laminarity",
    "TT": "trapping_time",
    "Vmax": "vl_max",
    "VENTR": "vl_entropy",
    "RTE": "rt_entropy",
}


def _julia_cross_values(x, y, epsilon, lmin):
    x_values = ", ".join(f"{value:.17g}" for value in x.tolist())
    y_values = ", ".join(f"{value:.17g}" for value in y.tolist())
    keys = ", ".join(f":{key}" for key in JULIA_NAMES)
    code = f"""
    using RecurrenceAnalysis
    R = CrossRecurrenceMatrix([{x_values}], [{y_values}], {epsilon})
    r = rqa(R; theiler=0, lmin={lmin})
    for key in ({keys},)
        println(r[key])
    end
    """
    return dict(zip(JULIA_NAMES, (float(line) for line in run_julia(code).split()), strict=True))


@pytest.mark.parametrize(("num_x", "num_y"), [(25, 25), (30, 18), (18, 30)])
def test_matches_recurrenceanalysis_jl_cross_recurrence(num_x, num_y):
    generator = torch.Generator().manual_seed(num_x * 100 + num_y)
    x = torch.cumsum(torch.randn(num_x, dtype=torch.float64, generator=generator), dim=0) * 0.4
    y = torch.cumsum(torch.randn(num_y, dtype=torch.float64, generator=generator), dim=0) * 0.4
    epsilon, lmin = 0.8, 2
    library = _library(hard_cross_recurrence_matrix(x, y, epsilon=epsilon), lmin)
    julia = _julia_cross_values(x, y, epsilon, lmin)
    for key, name in JULIA_NAMES.items():
        assert library[name].double().item() == pytest.approx(julia[key], rel=1e-5, abs=1e-6, nan_ok=True), key
