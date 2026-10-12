"""Tests for torchnlts.embedding.cao.cao_e1 / cao_e2.

Ground truth:
  1. Paper: Cao (1997), Physica D 110, eq. 1-3 (E1) and eq. 4-5 (E2), against an independent brute-force
     transcription, plus Cao's property that E2 is about 1 for every d on random data and differs from 1 for some d
     on deterministic data.
  2. Julia: DelayEmbeddings.jl's delay_afnn (E1) and stochastic_indicator (E2, within the tolerance explained in its
     test).
"""

import math

import numpy as np
import pytest
import torch

from tests.conftest import logistic_map_trajectory, pseudo_random_series, run_julia
from torchnlts.embedding.cao import cao_e1, cao_e2
from torchnlts.embedding.delay import delay_embedding


def _brute_force_cao(trajectory, embedding_dim, delay, theiler_window, new_coordinate_only):
    """Independent, unoptimized transcription of Cao (1997) eq. 1-5, including delay_afnn's coincident-neighbor
    fallback (see cao.py). ``new_coordinate_only`` selects E*(d) of eq. 4 (the plain distance between the added
    coordinates) instead of E(d) of eq. 1 (the ratio of the d + 1 to the d dimensional distance).
    """
    points = trajectory.reshape(trajectory.shape[0], -1)

    def mean_of(dim):
        embedded_low = delay_embedding(trajectory, dim, delay=delay)
        embedded_high = delay_embedding(trajectory, dim + 1, delay=delay)
        num_common = embedded_high.shape[0]
        embedded_low = embedded_low[:num_common]
        total, count = 0.0, 0
        for i in range(num_common):
            candidates = []
            for j in range(num_common):
                if abs(i - j) <= theiler_window:
                    continue
                d = torch.linalg.vector_norm(embedded_low[i] - embedded_low[j]).item()
                candidates.append((d, j))
            if not candidates:
                continue
            candidates.sort(key=lambda pair: pair[0])
            best_d, best_j = candidates[0]
            if best_d == 0.0 and len(candidates) > 1:
                best_d, best_j = candidates[1]
            if best_d == 0.0:
                continue
            if new_coordinate_only:
                total += torch.linalg.vector_norm(points[i + dim * delay] - points[best_j + dim * delay]).item()
            else:
                total += torch.linalg.vector_norm(embedded_high[i] - embedded_high[best_j]).item() / best_d
            count += 1
        return total / (count if new_coordinate_only else num_common - 1)

    return mean_of(embedding_dim + 1) / mean_of(embedding_dim)


def _julia_stochastic_indicator(cases):
    """E2 of DelayEmbeddings.jl's stochastic_indicator for d = 1..4, one list per ``(series, delay)`` in ``cases``."""
    lines = ["using DelayEmbeddings"]
    for series, delay in cases:
        values = ", ".join(f"{v:.17g}" for v in series.tolist())
        lines.append(f'println("E2|", join(stochastic_indicator([{values}], {delay}, 1:4), ","))')
    stdout = run_julia("\n".join(lines))
    rows = [line.split("|", 1)[1] for line in stdout.splitlines() if line.startswith("E2|")]
    return [torch.tensor([float(value) for value in row.split(",")], dtype=torch.float64) for row in rows]


def test_cao_e1_matches_brute_force_reference():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    for embedding_dim in (1, 2, 3):
        expected = _brute_force_cao(trajectory, embedding_dim, delay=1, theiler_window=1, new_coordinate_only=False)
        actual = cao_e1(trajectory, embedding_dim, delay=1, theiler_window=1).item()
        assert math.isclose(actual, expected), (embedding_dim, actual, expected)


def test_cao_e2_matches_brute_force_reference():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    for embedding_dim in (1, 2, 3):
        expected = _brute_force_cao(trajectory, embedding_dim, delay=1, theiler_window=1, new_coordinate_only=True)
        actual = cao_e2(trajectory, embedding_dim, delay=1, theiler_window=1).item()
        assert math.isclose(actual, expected), (embedding_dim, actual, expected)


@pytest.mark.parametrize("function, new_coordinate_only", [(cao_e1, False), (cao_e2, True)])
def test_cao_multivariate_matches_brute_force_reference(function, new_coordinate_only):
    torch.manual_seed(5)
    trajectory = torch.randn(50, 2, dtype=torch.float64)
    for embedding_dim in (1, 2):
        expected = _brute_force_cao(trajectory, embedding_dim, 2, 1, new_coordinate_only)
        actual = function(trajectory, embedding_dim, delay=2, theiler_window=1).item()
        assert math.isclose(actual, expected), (embedding_dim, actual, expected)


def test_cao_e2_is_near_one_on_noise():
    # Cao's property: E2 is about 1 for every d on random data. At N = 2000 it scatters around 1 with a standard
    # deviation of about 0.024 (the mean of about 2000 half-normal terms), so 0.1 is about four standard deviations.
    torch.manual_seed(0)
    noise = torch.randn(2000, dtype=torch.float64)
    for embedding_dim in range(1, 6):
        assert abs(cao_e2(noise, embedding_dim, delay=1).item() - 1.0) < 0.1, embedding_dim


def test_cao_e2_differs_from_one_on_logistic_map():
    # Cao's property: E2 differs from 1 for some d on deterministic data. The r = 4 logistic map gives about 2 at every d.
    trajectory = logistic_map_trajectory(2000)
    for embedding_dim in range(1, 6):
        assert abs(cao_e2(trajectory, embedding_dim, delay=1).item() - 1.0) > 0.5, embedding_dim


def test_cao_e1_saturates_for_low_dimensional_system():
    # A 2D limit cycle observed through one coordinate: E1 should be
    # markedly below 1 (still unfolding) at d=1, and near 1 by d=2-3.
    t = torch.linspace(0, 60 * math.pi, 600, dtype=torch.float64)
    trajectory = torch.sin(t)
    e1_low = cao_e1(trajectory, embedding_dim=1, delay=5, theiler_window=5).item()
    e1_high = cao_e1(trajectory, embedding_dim=3, delay=5, theiler_window=5).item()
    assert abs(e1_high - 1.0) < abs(e1_low - 1.0)


def test_cao_e1_matches_delayembeddings_jl():
    torch.manual_seed(1)
    trajectory = torch.randn(100, dtype=torch.float64)
    delay, embedding_dim = 2, 2

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using DelayEmbeddings
    s = [{values}]
    e1s = delay_afnn(s, {delay}, {embedding_dim}:{embedding_dim})
    println(e1s[1])
    """
    stdout = run_julia(julia_code)
    julia_e1 = float(stdout.strip())
    torch_e1 = cao_e1(trajectory, embedding_dim, delay=delay, theiler_window=0).item()
    assert math.isclose(torch_e1, julia_e1, rel_tol=1e-6), (torch_e1, julia_e1)


def test_cao_e2_matches_delayembeddings_jl():
    # rtol 2%: Julia divides each E* by the embedded length (not the number of summed terms) and leaves the last
    # tau + 1 points out of its neighbor tree; both effects are below 0.5% at 500 points and shrink as 1/N.
    torch.manual_seed(1)
    cases = [(torch.randn(500, dtype=torch.float64), 2), (logistic_map_trajectory(500), 1)]
    for (series, delay), julia_e2 in zip(cases, _julia_stochastic_indicator(cases), strict=True):
        torch_e2 = torch.stack([cao_e2(series, embedding_dim, delay=delay) for embedding_dim in range(1, 5)])
        assert torch.allclose(torch_e2, julia_e2, rtol=2e-2), (torch_e2, julia_e2)


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
def test_cao_shapes_and_batch_matches_rows(function):
    torch.manual_seed(3)
    batch = torch.randn(3, 80, 1, dtype=torch.float64)
    result = function(batch, 2)
    assert result.shape == (3,)
    assert function(batch[:1], 2).shape == (1,)
    assert function(batch[0], 2).shape == ()
    assert torch.allclose(result, torch.stack([function(row, 2) for row in batch]))


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
def test_cao_integer_input_matches_default_float(function):
    # Regression: linalg.vector_norm raised on int64 input.
    torch.manual_seed(0)
    integers = torch.randint(0, 1000, (120,))
    result = function(integers, 2)
    assert result.dtype == torch.get_default_dtype() and torch.isfinite(result)
    assert torch.equal(result, function(integers.to(torch.get_default_dtype()), 2))


@pytest.mark.parametrize(
    "function, dtype, embedding_dim, series",
    [
        # Amplitudes near the float16 limit make differences of points overflow in half arithmetic.
        (cao_e1, torch.float16, 2, (300, 12345, 4e4)),
        (cao_e2, torch.float16, 2, (300, 12345, 4e4)),
        (cao_e1, torch.bfloat16, 3, (100, 2, 1.0)),
        (cao_e2, torch.bfloat16, 2, (300, 12345, 3e4)),
    ],
)
def test_cao_half_precision_is_computed_in_float32(function, dtype, embedding_dim, series):
    values = pseudo_random_series(*series).to(dtype)
    result = function(values, embedding_dim)
    assert result.dtype == dtype and torch.isfinite(result)
    assert torch.equal(result, function(values.float(), embedding_dim).to(dtype))


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
@pytest.mark.parametrize("position, value", [(0, float("nan")), (60, float("nan")), (119, float("inf"))])
def test_cao_nonfinite_row_is_nan_and_other_rows_are_unaffected(function, position, value):
    # Regression: a non-finite sample near the start of the series left the result finite.
    torch.manual_seed(0)
    clean = torch.randn(120, dtype=torch.float64)
    bad = clean.clone()
    bad[position] = value
    result = function(torch.stack([clean, bad])[..., None], 2)
    assert torch.isnan(result[1])
    assert torch.equal(result[0], function(clean, 2))
    assert torch.isnan(function(bad, 2))


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
def test_cao_nonfinite_row_does_not_leak_gradient(function):
    torch.manual_seed(0)
    clean = torch.randn(80, dtype=torch.float64)
    bad = clean.clone()
    bad[0] = float("nan")
    rows = torch.stack([clean, bad])[..., None].requires_grad_()
    torch.nan_to_num(function(rows, 2)).sum().backward()
    standalone = clean.clone().requires_grad_()
    function(standalone, 2).backward()
    assert torch.isfinite(rows.grad).all()
    assert standalone.grad.abs().sum() > 0
    assert torch.equal(rows.grad[0, :, 0], standalone.grad)
    assert (rows.grad[1] == 0).all()


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
def test_cao_vmap_matches_batched_call(function):
    torch.manual_seed(0)
    rows = torch.randn(3, 100, dtype=torch.float64)
    rows[1, 0] = float("nan")
    mapped = torch.vmap(lambda row: function(row, 2))(rows)
    assert torch.allclose(mapped, function(rows[..., None], 2), equal_nan=True)
    assert torch.isnan(mapped[1]) and torch.isfinite(mapped[[0, 2]]).all()


@pytest.mark.parametrize("function", [cao_e1, cao_e2])
def test_cao_integer_arguments_are_validated(function):
    series = torch.randn(60)
    for embedding_dim in (2.0, "2", None):
        with pytest.raises(ValueError):
            function(series, embedding_dim)
    with pytest.raises(ValueError):
        function(series, 2, delay=1.5)
    assert torch.equal(function(series, np.int64(2), delay=np.int32(1)), function(series, 2, delay=1))
