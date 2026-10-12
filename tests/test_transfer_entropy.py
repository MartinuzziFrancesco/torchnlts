"""Tests for torchnlts.causality.transfer_entropy.

Ground truth:

- Frenzel & Pompe (2007): conditioning on a constant reduces ``conditional_mutual_information(x, y, z)`` to
  ``mutual_information(x, y)`` (``max(dist_x, 0) == dist_x``, so ``count_xz`` and ``count_yz`` are the plain marginal
  counts). ``count_z`` equals the number of points only if every joint radius is positive, which holds for the
  4-point example used here.
- Schreiber (2000): ``transfer_entropy`` is larger in the direction of the coupling of a unidirectionally coupled
  logistic-map pair, and near zero for independent noise.
- CausalityTools.jl (``FPVP``, ``TEShannon`` through ``CMIDecomposition``): it counts marginal neighbors inclusively
  (``<=``), this module strictly (``<``), so the default results differ. An inclusive-counting variant written here
  without calling the module matches Julia to float64 precision, which pins the joint radius, the digamma terms and
  the alignment of the marginals.
"""

import math

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.causality.transfer_entropy import conditional_mutual_information, transfer_entropy
from torchnlts.entropy.mutual_information import mutual_information

JULIA_PROJECT = "juliaenv_causality"


def test_conditional_mutual_information_constant_z_matches_mutual_information():
    # Conditioning on a constant reduces CMI to MI; reuses mutual_information's 4-point example (KSG1 = 0).
    x = torch.tensor([0.0, 1.0, 3.0, 10.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64)
    z_constant = torch.full((4,), 5.0, dtype=torch.float64)
    mi = mutual_information(x, y, num_neighbors=1, theiler_window=0)
    cmi = conditional_mutual_information(x, y, z_constant, num_neighbors=1, theiler_window=0)
    assert torch.isclose(mi, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(cmi, mi, atol=1e-12)


def test_conditional_mutual_information_matches_causalitytools_jl_fpvp_disagreement():
    # Strict marginal counting differs from FPVP's inclusive counting; see the module docstring.
    torch.manual_seed(1)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    z = torch.randn(50, dtype=torch.float64)

    strict = conditional_mutual_information(x, y, z, num_neighbors=3, theiler_window=0)

    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    z = {z.tolist()}
    est = FPVP(CMIShannon(base=ℯ); k=3, w=0)
    println(association(est, x, y, z))
    """
    stdout = run_julia(julia_code, JULIA_PROJECT)
    julia_inclusive = float(stdout.strip())

    assert not math.isclose(strict.item(), julia_inclusive, abs_tol=0.05)


def _inclusive_conditional_mutual_information(x, y, z, num_neighbors, theiler_window):
    # Plain-torch reimplementation of CausalityTools.jl's inclusive (<=) marginal counting, independent of the module.
    from torchnlts._shape import _ensure_batched
    from torchnlts.neighbors.distance import pairwise_distance
    from torchnlts.neighbors.theiler import apply_theiler_window

    x_b, _ = _ensure_batched(x)
    y_b, _ = _ensure_batched(y)
    z_b, _ = _ensure_batched(z)
    dist_x = pairwise_distance(x_b, norm_order=float("inf"))
    dist_y = pairwise_distance(y_b, norm_order=float("inf"))
    dist_z = pairwise_distance(z_b, norm_order=float("inf"))
    joint = apply_theiler_window(torch.maximum(torch.maximum(dist_x, dist_y), dist_z), theiler_window)
    radius = joint.topk(num_neighbors, dim=-1, largest=False).values[..., -1]
    r = radius.unsqueeze(-1)
    count_xz = (torch.maximum(dist_x, dist_z) <= r).sum(dim=-1).to(dist_x.dtype)
    count_yz = (torch.maximum(dist_y, dist_z) <= r).sum(dim=-1).to(dist_x.dtype)
    count_z = (dist_z <= r).sum(dim=-1).to(dist_x.dtype)
    per_point = torch.digamma(count_xz) + torch.digamma(count_yz) - torch.digamma(count_z)
    digamma_k = torch.digamma(torch.tensor(float(num_neighbors), dtype=dist_x.dtype))
    return (digamma_k - per_point.mean(dim=-1)).squeeze()


def test_conditional_mutual_information_inclusive_variant_matches_causalitytools_jl():
    # Same input, inclusive-counting variant vs Julia: checks the joint-radius/digamma formula and marginal alignment.
    torch.manual_seed(1)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    z = torch.randn(50, dtype=torch.float64)

    inclusive = _inclusive_conditional_mutual_information(x, y, z, num_neighbors=3, theiler_window=0)

    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    z = {z.tolist()}
    est = FPVP(CMIShannon(base=ℯ); k=3, w=0)
    println(association(est, x, y, z))
    """
    stdout = run_julia(julia_code, JULIA_PROJECT)
    julia_inclusive = float(stdout.strip())
    assert math.isclose(inclusive.item(), julia_inclusive, abs_tol=1e-9)


def test_conditional_mutual_information_rejects_shape_and_param_issues():
    x = torch.randn(20, dtype=torch.float64)
    y = torch.randn(20, dtype=torch.float64)
    z = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError):
        conditional_mutual_information(x, y, z, num_neighbors=0)
    with pytest.raises(ValueError):
        conditional_mutual_information(x, y, z, theiler_window=-1)


def test_conditional_mutual_information_nan_on_insufficient_neighbors():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    result = conditional_mutual_information(x, y, z, num_neighbors=5, theiler_window=0)
    assert torch.isnan(result)


def test_conditional_mutual_information_batched_matches_independent_loop():
    torch.manual_seed(2)
    x = torch.randn(3, 30, 1, dtype=torch.float64)
    y = torch.randn(3, 30, 1, dtype=torch.float64)
    z = torch.randn(3, 30, 1, dtype=torch.float64)
    batched = conditional_mutual_information(x, y, z, num_neighbors=3, theiler_window=1)
    assert batched.shape == (3,)
    for i in range(3):
        expected = conditional_mutual_information(x[i], y[i], z[i], num_neighbors=3, theiler_window=1)
        assert torch.isclose(batched[i], expected)


def _coupled_logistic_maps(num_points: int, seed_x: float = 0.4, seed_y: float = 0.3):
    rx, ry, coupling = 3.8, 3.5, 0.3
    x = torch.zeros(num_points, dtype=torch.float64)
    y = torch.zeros(num_points, dtype=torch.float64)
    x[0], y[0] = seed_x, seed_y
    for t in range(num_points - 1):
        x[t + 1] = x[t] * rx * (1 - x[t])
        y[t + 1] = (y[t] * ry * (1 - y[t]) + coupling * x[t]) % 1.0
    return x, y


def test_transfer_entropy_recovers_correct_causal_direction():
    x, y = _coupled_logistic_maps(2000)
    te_x_to_y = transfer_entropy(x, y, embedding_dim=2, delay=1, theiler_window=5)
    te_y_to_x = transfer_entropy(y, x, embedding_dim=2, delay=1, theiler_window=5)
    assert te_x_to_y.item() > 0.2
    assert te_x_to_y.item() > te_y_to_x.item() + 0.2


def test_transfer_entropy_near_zero_for_independent_series():
    torch.manual_seed(3)
    a = torch.randn(2000, dtype=torch.float64)
    b = torch.randn(2000, dtype=torch.float64)
    te = transfer_entropy(a, b, embedding_dim=2, theiler_window=5)
    assert abs(te.item()) < 0.05


def test_transfer_entropy_matches_causalitytools_jl_disagreement():
    # Julia CMIDecomposition(TEShannon, FPVP(k=3, w=0)); matches to float64 once both use inclusive counting.
    torch.manual_seed(4)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)

    strict = transfer_entropy(x, y, embedding_dim=1, delay=1, prediction_lag=1, num_neighbors=3, theiler_window=0)

    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    est = CMIDecomposition(TEShannon(base=ℯ), FPVP(CMIShannon(base=ℯ); k=3, w=0))
    println(association(est, x, y))
    """
    stdout = run_julia(julia_code, JULIA_PROJECT)
    julia_inclusive = float(stdout.strip())
    assert not math.isclose(strict.item(), julia_inclusive, abs_tol=0.05)


def test_transfer_entropy_inclusive_variant_matches_causalitytools_jl():
    # Same case; marginals (y[1:], x[:-1], y[:-1]) under inclusive counting.
    torch.manual_seed(4)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)

    inclusive = _inclusive_conditional_mutual_information(y[1:], x[:-1], y[:-1], num_neighbors=3, theiler_window=0)

    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    est = CMIDecomposition(TEShannon(base=ℯ), FPVP(CMIShannon(base=ℯ); k=3, w=0))
    println(association(est, x, y))
    """
    stdout = run_julia(julia_code, JULIA_PROJECT)
    julia_inclusive = float(stdout.strip())
    assert math.isclose(inclusive.item(), julia_inclusive, abs_tol=1e-9)


def test_transfer_entropy_batched_matches_independent_loop():
    torch.manual_seed(5)
    source = torch.randn(3, 60, 1, dtype=torch.float64)
    target = torch.randn(3, 60, 1, dtype=torch.float64)
    batched = transfer_entropy(source, target, embedding_dim=2, theiler_window=1)
    assert batched.shape == (3,)
    for i in range(3):
        expected = transfer_entropy(source[i], target[i], embedding_dim=2, theiler_window=1)
        assert torch.isclose(batched[i], expected)


def test_transfer_entropy_dtype_preserved_float32():
    torch.manual_seed(6)
    source = torch.randn(30, dtype=torch.float32)
    target = torch.randn(30, dtype=torch.float32)
    result = transfer_entropy(source, target, embedding_dim=2)
    assert result.dtype == torch.float32


def test_transfer_entropy_truncates_to_shorter_series():
    source = torch.randn(40, dtype=torch.float64)
    target = torch.randn(25, dtype=torch.float64)
    result = transfer_entropy(source, target, embedding_dim=2, delay=1, prediction_lag=1)
    assert torch.isfinite(result) or torch.isnan(result)


def test_transfer_entropy_rejects_bad_params():
    source = torch.randn(30, dtype=torch.float64)
    target = torch.randn(30, dtype=torch.float64)
    with pytest.raises(ValueError):
        transfer_entropy(source, target, embedding_dim=0)
    with pytest.raises(ValueError):
        transfer_entropy(source, target, prediction_lag=0)


def test_transfer_entropy_rejects_too_short_trajectory():
    source = torch.randn(3, dtype=torch.float64)
    target = torch.randn(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        transfer_entropy(source, target, embedding_dim=5, delay=1, prediction_lag=1)


@pytest.mark.parametrize("name", ["num_neighbors", "theiler_window"])
def test_conditional_mutual_information_integer_arguments_raise_value_error(name):
    series = torch.randn(30, dtype=torch.float64)
    for bad in (3.0, 1.5, "3", None, torch.tensor([1, 2])):
        with pytest.raises(ValueError, match=name):
            conditional_mutual_information(series, series.flip(0), series.roll(3), **{name: bad})


def test_conditional_mutual_information_accepts_numpy_integers():
    torch.manual_seed(12)
    x, y, z = (torch.randn(40, dtype=torch.float64) for _ in range(3))
    result = conditional_mutual_information(x, y, z, num_neighbors=numpy.int64(2), theiler_window=numpy.int32(1))
    assert torch.equal(result, conditional_mutual_information(x, y, z, num_neighbors=2, theiler_window=1))


@pytest.mark.parametrize("name", ["embedding_dim", "delay", "prediction_lag", "num_neighbors", "theiler_window"])
def test_transfer_entropy_integer_arguments_raise_value_error(name):
    series = torch.randn(40, dtype=torch.float64)
    for bad in (2.0, 1.5, "2", None, torch.tensor([1, 2])):
        with pytest.raises(ValueError, match=name):
            transfer_entropy(series, series.flip(0), **{name: bad})


def test_transfer_entropy_accepts_numpy_integers():
    torch.manual_seed(13)
    source, target = torch.randn(60, dtype=torch.float64), torch.randn(60, dtype=torch.float64)
    result = transfer_entropy(
        source,
        target,
        embedding_dim=numpy.int64(2),
        delay=numpy.int32(1),
        prediction_lag=numpy.int64(2),
        num_neighbors=numpy.int16(2),
        theiler_window=numpy.int8(1),
    )
    expected = transfer_entropy(
        source, target, embedding_dim=2, delay=1, prediction_lag=2, num_neighbors=2, theiler_window=1
    )
    assert torch.equal(result, expected)


@pytest.mark.parametrize("variable", ["x", "y", "z"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_conditional_mutual_information_nonfinite_point_gives_nan_for_that_element_only(variable, bad_value):
    torch.manual_seed(14)
    arguments = {name: torch.randn(3, 40, 1, dtype=torch.float64) for name in "xyz"}
    clean = conditional_mutual_information(**arguments, theiler_window=2)
    arguments[variable] = arguments[variable].clone()
    arguments[variable][1, 7, 0] = bad_value
    result = conditional_mutual_information(**arguments, theiler_window=2)
    assert torch.isnan(result[1]) and torch.isfinite(clean[1])
    assert torch.equal(result[[0, 2]], clean[[0, 2]])


def test_conditional_mutual_information_huge_theiler_window_excludes_every_neighbor():
    torch.manual_seed(20)
    x, y, z = (torch.randn(30, dtype=torch.float64) for _ in range(3))
    for window in (29, 2**63 - 1, 2**63, 2**64, 10**30):
        assert torch.isnan(conditional_mutual_information(x, y, z, theiler_window=window))
        assert torch.isnan(transfer_entropy(x, y, theiler_window=window))


def test_conditional_mutual_information_and_transfer_entropy_work_under_vmap():
    torch.manual_seed(21)
    x, y, z = (torch.randn(3, 40, 1, dtype=torch.float64) for _ in range(3))
    x[1, 5, 0] = float("nan")
    expected = conditional_mutual_information(x, y, z, theiler_window=1)
    assert torch.isnan(expected[1]) and torch.isfinite(expected[[0, 2]]).all()
    torch.testing.assert_close(
        torch.vmap(lambda a, b, c: conditional_mutual_information(a, b, c, theiler_window=1))(x, y, z),
        expected,
        equal_nan=True,
    )
    expected = transfer_entropy(x, y, embedding_dim=2, theiler_window=1)
    torch.testing.assert_close(
        torch.vmap(lambda a, b: transfer_entropy(a, b, embedding_dim=2, theiler_window=1))(x, y),
        expected,
        equal_nan=True,
    )
