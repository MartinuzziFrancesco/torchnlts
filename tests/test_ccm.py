"""Tests for torchnlts.causality.ccm.

Ground truth: a hand-derived 5-point example of the neighbor weighting; a unidirectionally coupled logistic-map pair
(the driven series' manifold recovers the driver, the reverse does not); CausalityTools.jl's
``ConvergentCrossMapping``/``predict`` run through ``juliaenv_causality`` (its Theiler parameter ``w`` is
``theiler_window = w + 1`` here); and a plain-torch loop for values and gradients.
"""

import math

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.causality.ccm import ccm_skill, cross_map_estimate

JULIA_PROJECT = "juliaenv_causality"


def test_cross_map_estimate_hand_derived():
    # Hand-computed with math.exp: manifold [0,1,5,6,20], estimated [10,20,50,60,200], embedding_dim=1,
    # 2 neighbors weighted by exp(-d_k/d_1)/sum(exp(-d_j/d_1)).
    manifold = torch.tensor([0.0, 1.0, 5.0, 6.0, 20.0], dtype=torch.float64)
    estimated = torch.tensor([10.0, 20.0, 50.0, 60.0, 200.0], dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=0)
    expected = torch.tensor(
        [20.539586298862744, 11.897034927102672, 58.10296507289733, 49.460413701137256, 55.17849554406135],
        dtype=torch.float64,
    )
    assert torch.allclose(result, expected, atol=1e-10, rtol=0)


def test_cross_map_estimate_zero_distance_duplicate_no_crash():
    # manifold[0] == manifold[2]: a zero nearest distance makes exp(-d_1/d_1) 0/0; must not NaN, duplicate dominates.
    manifold = torch.tensor([0.0, 5.0, 0.0, 8.0, 9.0], dtype=torch.float64)
    estimated = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=0)
    assert torch.isfinite(result).all()
    assert torch.isclose(result[0], torch.tensor(3.0, dtype=torch.float64), atol=1e-6)


def test_cross_map_estimate_nan_on_insufficient_neighbors_library_too_small():
    # library_size=1 gives at most 1 candidate per row, fewer than the 2 neighbors needed: all NaN.
    torch.manual_seed(4)
    manifold = torch.randn(6, dtype=torch.float64)
    estimated = torch.randn(6, dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, library_size=1, theiler_window=0)
    assert torch.isnan(result).all()


def test_cross_map_estimate_nan_on_insufficient_neighbors_short_trajectory():
    # 4 neighbors are needed but only 2 embedded points exist: all NaN.
    manifold = torch.arange(4.0, dtype=torch.float64)
    estimated = 10 * manifold
    result = cross_map_estimate(manifold, estimated, embedding_dim=3, delay=1, theiler_window=0)
    assert torch.isnan(result).all()


def test_cross_map_estimate_nan_only_on_theiler_starved_rows():
    # A large Theiler window starves only the middle rows: the NaN mask is per row.
    torch.manual_seed(5)
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=15)
    assert torch.isfinite(result[:3]).all()
    assert torch.isnan(result[3:17]).all()
    assert torch.isfinite(result[17:]).all()


def test_ccm_skill_hand_derived():
    estimate = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    perfect = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64)
    inverted = torch.tensor([8.0, 6.0, 4.0, 2.0], dtype=torch.float64)
    assert torch.isclose(ccm_skill(estimate, perfect), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)
    assert torch.isclose(ccm_skill(estimate, inverted), torch.tensor(-1.0, dtype=torch.float64), atol=1e-10)


def test_ccm_skill_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        ccm_skill(torch.zeros(5, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="matching shapes"):
        ccm_skill(torch.zeros(5, dtype=torch.float64), torch.zeros(5, 1, dtype=torch.float64))


def test_ccm_skill_nan_on_constant_estimate():
    estimate = torch.full((10,), 3.0, dtype=torch.float64)
    actual = torch.randn(10, dtype=torch.float64)
    assert torch.isnan(ccm_skill(estimate, actual))


def test_ccm_skill_degenerate_columns_are_nan_and_keep_gradients_finite():
    nan, inf = float("nan"), float("inf")
    parameter = torch.tensor(0.5, requires_grad=True)
    # Rows: constant estimate, NaN estimate, constant actual, inf actual, valid. The non-finite values are
    # additive, so the gradient into the shared parameter is finite apart from the library's own handling.
    base = torch.tensor([[3.0, 3.0, 3.0], [0.0, nan, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0]])
    slope = torch.tensor([[0.0, 0.0, 0.0]] + [[1.0, -1.0, 2.0]] * 4)
    actual = torch.tensor([[0.0, 1.0, 5.0], [0.0, 1.0, 5.0], [7.0, 7.0, 7.0], [0.0, inf, 5.0], [0.0, 1.0, 5.0]])
    skill = ccm_skill((base + parameter * slope).unsqueeze(-1), actual.unsqueeze(-1))
    assert torch.isnan(skill).flatten().tolist() == [True, True, True, True, False]
    (gradient,) = torch.autograd.grad(skill[4, 0], parameter)
    (expected,) = torch.autograd.grad(ccm_skill(base[4] + parameter * slope[4], actual[4]), parameter)
    assert gradient != 0
    torch.testing.assert_close(gradient, expected)


def test_ccm_skill_nonfinite_actual_column_keeps_gradients_finite():
    torch.manual_seed(5)
    base, slope = torch.randn(12, 2, dtype=torch.float64), torch.randn(12, 2, dtype=torch.float64)
    actual = torch.randn(12, 2, dtype=torch.float64)
    actual[4, 1] = float("nan")
    actual.requires_grad_()
    shift = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    skill = ccm_skill(base + shift * slope, actual)
    assert torch.isfinite(skill[0]) and torch.isnan(skill[1])
    gradients = torch.autograd.grad(skill[0], (shift, actual))

    expected_skill = ccm_skill(base[:, :1] + shift * slope[:, :1], actual[:, :1])
    expected = torch.autograd.grad(expected_skill.sum(), (shift, actual))
    assert gradients[0] != 0
    assert torch.isfinite(gradients[0]) and torch.isfinite(gradients[1]).all()
    torch.testing.assert_close(gradients[0], expected[0])
    torch.testing.assert_close(gradients[1][:, 0], expected[1][:, 0])
    assert (gradients[1][:, 1] == 0).all()


def _coupled_logistic_maps(num_points: int, seed_x: float = 0.4, seed_y: float = 0.3):
    rx, ry, coupling = 3.8, 3.5, 0.3
    x = torch.zeros(num_points, dtype=torch.float64)
    y = torch.zeros(num_points, dtype=torch.float64)
    x[0], y[0] = seed_x, seed_y
    for t in range(num_points - 1):
        x[t + 1] = x[t] * rx * (1 - x[t])
        y[t + 1] = (y[t] * ry * (1 - y[t]) + coupling * x[t]) % 1.0
    return x, y


def test_cross_map_estimate_recovers_correct_causal_direction():
    # X drives Y: cross-mapping the driver X from the driven Y's manifold should show higher skill than the reverse.
    x, y = _coupled_logistic_maps(1000)
    embedding_dim, delay, theiler_window = 3, 1, 5

    estimate_x_from_y = cross_map_estimate(y, x, embedding_dim, delay=delay, theiler_window=theiler_window)
    n = estimate_x_from_y.shape[0]
    offset = (embedding_dim - 1) * delay
    skill_x_from_y = ccm_skill(estimate_x_from_y, x[offset : offset + n])

    estimate_y_from_x = cross_map_estimate(x, y, embedding_dim, delay=delay, theiler_window=theiler_window)
    skill_y_from_x = ccm_skill(estimate_y_from_x, y[offset : offset + n])

    assert skill_x_from_y.item() > 0.8
    assert skill_x_from_y.item() > skill_y_from_x.item() + 0.3


def test_cross_map_estimate_matches_causalitytools_jl():
    # CCM(d=3, tau=-1, w=5) in Julia excludes |i-j|<=6, so theiler_window=6 here (w -> theiler_window+1).
    x, y = _coupled_logistic_maps(60)
    embedding_dim, delay, w = 3, 1, 5

    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    measure = CCM(d={embedding_dim}, τ={-delay}, w={w})
    t_hat, t_bar, rho = predict(measure, x, y)
    println(join(t_hat, " "))
    println(rho)
    """
    stdout = run_julia(julia_code, JULIA_PROJECT)
    lines = stdout.strip().splitlines()
    julia_estimate = torch.tensor([float(v) for v in lines[0].split()], dtype=torch.float64)
    julia_skill = float(lines[1])

    estimate = cross_map_estimate(y, x, embedding_dim, delay=delay, theiler_window=w + 1)
    n = estimate.shape[0]
    offset = (embedding_dim - 1) * delay
    skill = ccm_skill(estimate, x[offset : offset + n])

    assert estimate.shape == julia_estimate.shape
    assert torch.allclose(estimate, julia_estimate, atol=1e-9, rtol=0)
    assert math.isclose(skill.item(), julia_skill, abs_tol=1e-9)


def test_cross_map_estimate_batched_matches_independent_loop():
    torch.manual_seed(0)
    manifold = torch.randn(3, 40, 1, dtype=torch.float64)
    estimated = torch.randn(3, 40, 2, dtype=torch.float64)
    batched = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, theiler_window=1)
    assert batched.shape[0] == 3
    for i in range(3):
        expected = cross_map_estimate(manifold[i], estimated[i], embedding_dim=2, delay=1, theiler_window=1)
        assert torch.allclose(batched[i], expected)


def test_cross_map_estimate_output_rank_follows_estimated_series():
    torch.manual_seed(7)
    manifold = torch.randn(30, dtype=torch.float64)
    estimated = torch.randn(30, 3, dtype=torch.float64)
    first = cross_map_estimate(manifold, estimated[:, 0], embedding_dim=2)
    assert first.shape == (29,)
    assert cross_map_estimate(manifold, estimated[:, :1], embedding_dim=2).shape == (29, 1)
    assert cross_map_estimate(manifold, estimated, embedding_dim=2).shape == (29, 3)
    assert cross_map_estimate(manifold[None, :, None], estimated[None], embedding_dim=2).shape == (1, 29, 3)
    assert cross_map_estimate(manifold[None, :, None], estimated[None, :, :1], embedding_dim=2).shape == (1, 29, 1)
    # The manifold's rank does not matter: a [T, D] manifold with a [T] estimated series gives [N].
    wide = torch.randn(30, 2, dtype=torch.float64)
    assert cross_map_estimate(wide, estimated[:, 0], embedding_dim=2).shape == (29,)
    torch.testing.assert_close(first, cross_map_estimate(manifold, estimated[:, :1], embedding_dim=2)[:, 0])


def test_cross_map_estimate_single_estimate_keeps_its_axis():
    manifold = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    estimated = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert cross_map_estimate(manifold, estimated, embedding_dim=3, theiler_window=0).shape == (1,)


def test_ccm_skill_reduces_the_time_axis():
    torch.manual_seed(8)
    estimate = torch.randn(2, 30, 3, dtype=torch.float64)
    actual = torch.randn(2, 30, 3, dtype=torch.float64)
    batched = ccm_skill(estimate, actual)
    assert batched.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            expected = torch.corrcoef(torch.stack([estimate[i, :, j], actual[i, :, j]]))[0, 1]
            assert torch.isclose(batched[i, j], expected)
    assert ccm_skill(estimate[0], actual[0]).shape == (3,)
    assert ccm_skill(estimate[:1], actual[:1]).shape == (1, 3)
    scalar = ccm_skill(estimate[0, :, 0], actual[0, :, 0])
    assert scalar.shape == ()
    assert torch.isclose(scalar, batched[0, 0])
    assert ccm_skill(estimate[0, :1, 0], actual[0, :1, 0]).shape == ()


def test_ccm_skill_dtypes():
    estimate = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=torch.float64)
    actual = torch.tensor([2, 7, 1, 8, 2, 8, 1, 8])
    expected = ccm_skill(estimate, actual.to(torch.float64))
    assert ccm_skill(estimate, actual).dtype == torch.float64
    torch.testing.assert_close(ccm_skill(estimate, actual), expected)
    both_integer = ccm_skill(estimate.long(), actual)
    assert both_integer.dtype == torch.get_default_dtype()
    torch.testing.assert_close(both_integer.double(), ccm_skill(estimate.long().double(), actual.double()))
    for dtype in (torch.float16, torch.bfloat16):
        half = ccm_skill(estimate.to(dtype), actual.to(dtype))
        assert half.dtype == dtype
        assert torch.equal(half, ccm_skill(estimate.to(dtype).float(), actual.to(dtype).float()).to(dtype))
    with pytest.raises(ValueError, match="complex"):
        ccm_skill(estimate.to(torch.complex128), actual.to(torch.complex128))


def test_cross_map_estimate_library_size_restricts_neighbors():
    torch.manual_seed(1)
    manifold = torch.randn(50, dtype=torch.float64)
    estimated = torch.randn(50, dtype=torch.float64)
    full = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, library_size=None, theiler_window=0)
    partial = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, library_size=10, theiler_window=0)
    assert full.shape == partial.shape
    assert not torch.allclose(full, partial)


def test_cross_map_estimate_dtype_preserved_float32():
    torch.manual_seed(2)
    manifold = torch.randn(20, dtype=torch.float32)
    estimated = torch.randn(20, dtype=torch.float32)
    result = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1)
    assert result.dtype == torch.float32


def test_cross_map_estimate_promotes_dtypes_and_accepts_integers():
    torch.manual_seed(22)
    manifold, estimated = torch.randn(40, dtype=torch.float64), torch.randn(40, dtype=torch.float64)
    assert cross_map_estimate(manifold, estimated.float(), embedding_dim=2).dtype == torch.float64
    assert cross_map_estimate(manifold.float(), estimated, embedding_dim=2).dtype == torch.float64
    for dtype in (torch.float16, torch.bfloat16):
        result = cross_map_estimate(manifold.to(dtype), estimated.to(dtype), embedding_dim=2)
        assert result.dtype == torch.float32
        expected = cross_map_estimate(manifold.to(dtype).float(), estimated.to(dtype).float(), embedding_dim=2)
        assert torch.equal(result, expected)
    integers = (manifold * 10).round().long()
    result = cross_map_estimate(integers, integers.flip(0), embedding_dim=2)
    assert result.dtype == torch.get_default_dtype()
    floats = integers.to(torch.get_default_dtype())
    assert torch.equal(result, cross_map_estimate(floats, floats.flip(0), embedding_dim=2))


def test_cross_map_estimate_truncates_to_shorter_series():
    manifold = torch.randn(30, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1)
    assert result.shape[0] == 19


def test_cross_map_estimate_rejects_negative_theiler_window():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError):
        cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, theiler_window=-1)


def test_cross_map_estimate_rejects_library_size_out_of_range():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    for bad in (0, 100):
        with pytest.raises(ValueError):
            cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, library_size=bad)


def test_cross_map_estimate_rejects_mismatched_batchedness():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(2, 20, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1)


@pytest.mark.parametrize("name", ["embedding_dim", "delay", "theiler_window", "library_size"])
def test_cross_map_estimate_integer_arguments_raise_value_error(name):
    series = torch.randn(30, dtype=torch.float64)
    arguments = {"embedding_dim": 2, "delay": 1, "theiler_window": 0, "library_size": 10}
    for bad in (2.0, 1.5, "2", None, torch.tensor([1, 2])):
        if name == "library_size" and bad is None:
            continue
        with pytest.raises(ValueError, match=name):
            cross_map_estimate(series, series, **{**arguments, name: bad})


@pytest.mark.parametrize("which", ["manifold", "estimated", "both"])
def test_cross_map_estimate_rejects_complex_series(which):
    real = torch.randn(30, dtype=torch.float64)
    complex_series = torch.complex(real, real)
    manifold = complex_series if which in ("manifold", "both") else real
    estimated = complex_series if which in ("estimated", "both") else real
    with pytest.raises(ValueError, match="complex"):
        cross_map_estimate(manifold, estimated, embedding_dim=2)


def test_cross_map_estimate_accepts_numpy_integers():
    torch.manual_seed(9)
    manifold, estimated = torch.randn(30, dtype=torch.float64), torch.randn(30, dtype=torch.float64)
    result = cross_map_estimate(
        manifold,
        estimated,
        numpy.int64(2),
        delay=numpy.int32(1),
        library_size=numpy.int64(20),
        theiler_window=numpy.int8(1),
    )
    expected = cross_map_estimate(manifold, estimated, 2, delay=1, library_size=20, theiler_window=1)
    assert torch.equal(result, expected)


def test_cross_map_estimate_gradcheck():
    # Well-separated manifold values: finite-difference steps do not flip neighbor picks.
    torch.manual_seed(3)
    manifold = torch.tensor([0.0, 10.0, 25.0, 42.0, 60.0], dtype=torch.float64)
    estimated = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)

    def f(estimated):
        return cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=0)

    assert torch.autograd.gradcheck(f, (estimated,))


def test_cross_map_estimate_gradcheck_w_r_t_manifold_series():
    # Gradient through the distances that feed the weights, not just the weighted sum.
    torch.manual_seed(6)
    manifold = torch.tensor([0.0, 10.0, 25.0, 42.0, 60.0], dtype=torch.float64, requires_grad=True)
    estimated = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

    def f(manifold):
        return cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=0)

    assert torch.autograd.gradcheck(f, (manifold,))


def _cross_map_rows(manifold, estimated, embedding_dim, library_size, rows):
    """Plain-torch cross-map estimate (delay 1, Theiler window 0) of the given embedded rows only."""
    embedded = torch.stack(
        [manifold[start : start + embedding_dim] for start in range(len(manifold) - embedding_dim + 1)]
    )
    estimates = []
    for row in rows:
        candidates = [other for other in range(library_size) if other != row]
        distances = torch.stack([(embedded[row] - embedded[other]).norm() for other in candidates])
        order = distances.argsort()[: embedding_dim + 1]
        nearest = distances[order]
        weights = torch.exp(-nearest / nearest[0])
        weights = weights / weights.sum()
        targets = torch.stack([estimated[candidates[position] + embedding_dim - 1] for position in order.tolist()])
        estimates.append((weights * targets).sum())
    return torch.stack(estimates)


def _short_library_inputs():
    # library_size=3 leaves 2 candidates for each of the first 3 rows, fewer than the 3 neighbors needed.
    torch.manual_seed(0)
    manifold = torch.randn(40, dtype=torch.float64, requires_grad=True)
    estimated = torch.randn(40, dtype=torch.float64, requires_grad=True)
    return manifold, estimated


def test_cross_map_estimate_rows_without_neighbors_do_not_put_nan_into_gradients():
    manifold, estimated = _short_library_inputs()
    estimate = cross_map_estimate(manifold, estimated, embedding_dim=2, library_size=3)
    valid = torch.isfinite(estimate)
    assert valid.sum() == 36 and not valid[:3].any()

    gradients = torch.autograd.grad(estimate[valid].sum(), (manifold, estimated))
    rows = valid.nonzero().flatten().tolist()
    expected_estimate = _cross_map_rows(manifold, estimated, embedding_dim=2, library_size=3, rows=rows)
    expected_gradients = torch.autograd.grad(expected_estimate.sum(), (manifold, estimated))
    torch.testing.assert_close(estimate[valid], expected_estimate)
    for gradient, expected in zip(gradients, expected_gradients):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected)


def test_cross_map_estimate_gradcheck_restricted_to_valid_rows():
    manifold, estimated = _short_library_inputs()
    valid = torch.isfinite(cross_map_estimate(manifold, estimated, embedding_dim=2, library_size=3))

    def restricted(manifold, estimated):
        return cross_map_estimate(manifold, estimated, embedding_dim=2, library_size=3)[valid]

    assert torch.autograd.gradcheck(restricted, (manifold, estimated))


def test_cross_map_estimate_batched_invalid_rows_keep_other_batch_rows_clean():
    torch.manual_seed(10)
    manifold = torch.randn(2, 30, 1, dtype=torch.float64, requires_grad=True)
    estimated = torch.randn(2, 30, 2, dtype=torch.float64, requires_grad=True)
    estimate = cross_map_estimate(manifold, estimated, embedding_dim=2, library_size=3, theiler_window=1)
    valid = torch.isfinite(estimate).all(dim=-1)
    assert valid.any() and not valid.all()
    gradients = torch.autograd.grad(estimate[valid].sum(), (manifold, estimated))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cross_map_estimate_nonfinite_target_gives_nan_row_and_finite_gradients():
    # Embedding_dim 1 uses the 2 nearest values: the NaN target at index 2 (value 25) is a neighbor of rows 0, 1 and 3.
    values = torch.tensor([0.0, 10.0, 25.0, 42.0, 60.0, 83.0, 110.0], dtype=torch.float64)
    for bad_value in (float("nan"), float("inf")):
        manifold = values.clone().requires_grad_()
        targets = torch.arange(1.0, 8.0, dtype=torch.float64)
        targets[2] = bad_value
        targets.requires_grad_()
        estimate = cross_map_estimate(manifold, targets, embedding_dim=1, theiler_window=0)
        assert torch.isnan(estimate).tolist() == [True, True, False, True, False, False, False]
        valid = torch.isfinite(estimate)
        gradients = torch.autograd.grad(estimate[valid].sum(), (manifold, targets))
        rows = valid.nonzero().flatten().tolist()
        expected_gradients = torch.autograd.grad(
            _cross_map_rows(manifold, targets, embedding_dim=1, library_size=7, rows=rows).sum(), (manifold, targets)
        )
        for gradient, expected in zip(gradients, expected_gradients):
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient, expected)


def test_cross_map_estimate_nonfinite_manifold_sample_gives_nan_rows_and_finite_gradients():
    torch.manual_seed(11)
    values = torch.randn(40, dtype=torch.float64)
    values[20] = float("nan")
    manifold = values.requires_grad_()
    estimated = torch.randn(40, dtype=torch.float64, requires_grad=True)
    estimate = cross_map_estimate(manifold, estimated, embedding_dim=2)
    # The two embedded points containing sample 20 are the NaN rows.
    assert torch.isnan(estimate).nonzero().flatten().tolist() == [19, 20]
    gradients = torch.autograd.grad(estimate[torch.isfinite(estimate)].sum(), (manifold, estimated))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ccm_skill_gradcheck():
    torch.manual_seed(7)
    estimate = torch.randn(20, dtype=torch.float64, requires_grad=True)
    actual = torch.randn(20, dtype=torch.float64)

    def f(estimate):
        return ccm_skill(estimate, actual)

    assert torch.autograd.gradcheck(f, (estimate,))


def test_cross_map_estimate_huge_theiler_window_excludes_every_neighbor():
    torch.manual_seed(19)
    manifold, estimated = torch.randn(30, dtype=torch.float64), torch.randn(30, dtype=torch.float64)
    for window in (29, 2**63 - 1, 2**63, 2**64, 10**30):
        assert torch.isnan(cross_map_estimate(manifold, estimated, embedding_dim=2, theiler_window=window)).all()


def test_cross_map_estimate_and_ccm_skill_work_under_vmap():
    torch.manual_seed(18)
    manifold = torch.randn(3, 30, 1, dtype=torch.float64)
    estimated = torch.randn(3, 30, 2, dtype=torch.float64)

    def estimate(manifold, estimated):
        return cross_map_estimate(manifold, estimated, embedding_dim=2, library_size=5, theiler_window=1)

    mapped = torch.vmap(estimate)(manifold, estimated)
    torch.testing.assert_close(mapped, estimate(manifold, estimated), equal_nan=True)
    assert torch.isnan(mapped).any() and torch.isfinite(mapped).any()
    actual = torch.randn(3, 20, 2, dtype=torch.float64)
    torch.testing.assert_close(torch.vmap(ccm_skill)(actual, actual.flip(1)), ccm_skill(actual, actual.flip(1)))
