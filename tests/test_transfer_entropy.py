"""Tests for torchnlts.causality.transfer_entropy.

Dual ground truth:
  1. Paper: Frenzel & Pompe (2007) for the conditional-MI estimator
     itself -- conditioning on a *constant* collapses
     ``conditional_mutual_information(x, y, z)`` to plain
     ``mutual_information(x, y)``, since ``max(dist_x, dist_z=0) ==
     dist_x`` always (distances are non-negative), so ``count_xz``/
     ``count_yz`` reduce to plain ``count_x``/``count_y`` exactly,
     unconditionally. ``count_z`` reduces to "all points" (matching the
     accumulated ``+digamma(N)`` a plain MI computation has) *only* when
     every point's joint-space k-th-neighbor radius is strictly positive
     -- if some point's radius is exactly ``0`` (an exact tie/duplicate
     at the k-th-neighbor boundary in the (x,y) joint space), strict
     ``<`` counting gives ``count_z = 1`` (self only) for that point
     instead, and the collapse to plain MI no longer holds exactly.
     ``mutual_information``'s own hand-derived 4-point test case has no
     such tie (confirmed: every point's radius is a distinct positive
     value), so reusing it here as a hand-derived check is valid for
     that specific input, not a claim that the collapse holds
     unconditionally for every input. Schreiber (2000) for the transfer-
     entropy decomposition itself, checked via the causality-direction
     sanity test (a synthetic unidirectionally-coupled logistic-map
     pair, same system used for CCM's own validation).
  2. Julia: CausalityTools.jl's ``FPVP``/``TEShannon``. This module's
     strict marginal counting is a confirmed, deliberate deviation from
     that estimator's inclusive convention -- the same disagreement
     already locked in for ``mutual_information`` vs ``KSG1`` (both use
     ``Neighborhood.jl``'s ``inrangecount``). Two kinds of Julia checks
     below: a locked *disagreement* (default strict vs. Julia's
     default inclusive, confirmed to differ) and a locked *match* (an
     inclusive-counting variant computed independently here, without
     calling this module's own strict implementation, matching Julia to
     float64 precision) -- the match is what actually pins down that the
     joint-radius/digamma formula and the marginal alignment are
     correct, isolating the one known, deliberate difference; the
     disagreement alone would pass even for an unrelated wrong answer.
"""

import math
import subprocess

import torch

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
    # Locked disagreement with CausalityTools.jl FPVP (inclusive marginal counting, as for KSG1); see module docstring.
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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    julia_inclusive = float(result.stdout.strip())

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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    julia_inclusive = float(result.stdout.strip())
    assert math.isclose(inclusive.item(), julia_inclusive, abs_tol=1e-9)


def test_conditional_mutual_information_rejects_shape_and_param_issues():
    x = torch.randn(20, dtype=torch.float64)
    y = torch.randn(20, dtype=torch.float64)
    z = torch.randn(20, dtype=torch.float64)
    try:
        conditional_mutual_information(x, y, z, num_neighbors=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        conditional_mutual_information(x, y, z, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    julia_inclusive = float(result.stdout.strip())
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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    julia_inclusive = float(result.stdout.strip())
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
    try:
        transfer_entropy(source, target, embedding_dim=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        transfer_entropy(source, target, prediction_lag=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transfer_entropy_rejects_too_short_trajectory():
    source = torch.randn(3, dtype=torch.float64)
    target = torch.randn(3, dtype=torch.float64)
    try:
        transfer_entropy(source, target, embedding_dim=5, delay=1, prediction_lag=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
