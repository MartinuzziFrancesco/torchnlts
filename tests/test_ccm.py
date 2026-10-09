"""Tests for torchnlts.causality.ccm.

Dual ground truth:
  1. Paper: Sugihara et al. (2012) -- a hand-derived example (5-point
     scalar series, ``embedding_dim=1`` so each embedded point is just
     the raw value, weights computed independently via ``math.exp`` in
     a standalone script and hardcoded below) pins down the exact
     nearest-neighbor weighting formula; a synthetic unidirectionally-
     coupled logistic-map pair checks the causality-direction logic
     itself (the paper's actual point): cross-mapping from the driven
     series' manifold recovers the driver well, the reverse doesn't.
  2. Julia: CausalityTools.jl's ``ConvergentCrossMapping``/``predict``,
     run live via subprocess on the literal same coupled-logistic-map
     input, matches this module's implementation to float64 precision
     (~1e-16 max per-point difference) for the default full-library
     case -- but only once a confirmed discrepancy in its own ``w``
     (Theiler window) parameter is accounted for: its docstring claims
     ``w=0`` excludes only the query point itself, but it actually
     passes ``Theiler(w+1)`` to Neighborhood.jl (whose own documented
     semantics are ``|i-j| <= w`` excluded), so its *effective*
     exclusion radius is one more than what it claims. Confirmed by
     directly inspecting the neighbor sets Julia's own search returns
     for a mismatching row before and after applying this correction,
     not assumed. See the module docstring for the full explanation.
"""

import math
import subprocess

import torch

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
    assert torch.allclose(result.squeeze(-1), expected, atol=1e-10, rtol=0)


def test_cross_map_estimate_zero_distance_duplicate_no_crash():
    # manifold[0] == manifold[2]: a zero nearest distance makes exp(-d_1/d_1) 0/0; must not NaN, duplicate dominates.
    manifold = torch.tensor([0.0, 5.0, 0.0, 8.0, 9.0], dtype=torch.float64)
    estimated = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=0)
    assert torch.isfinite(result).all()
    assert torch.isclose(result[0, 0], torch.tensor(3.0, dtype=torch.float64), atol=1e-6)


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
    result = cross_map_estimate(manifold, estimated, embedding_dim=1, delay=1, theiler_window=15).squeeze(-1)
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
    try:
        ccm_skill(torch.zeros(5, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ccm_skill_nan_on_constant_estimate():
    estimate = torch.full((10,), 3.0, dtype=torch.float64)
    actual = torch.randn(10, dtype=torch.float64)
    assert torch.isnan(ccm_skill(estimate, actual))


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

    estimate_x_from_y = cross_map_estimate(y, x, embedding_dim, delay, theiler_window=theiler_window)
    n = estimate_x_from_y.shape[0]
    offset = (embedding_dim - 1) * delay
    skill_x_from_y = ccm_skill(estimate_x_from_y, x[offset : offset + n].unsqueeze(-1))

    estimate_y_from_x = cross_map_estimate(x, y, embedding_dim, delay, theiler_window=theiler_window)
    skill_y_from_x = ccm_skill(estimate_y_from_x, y[offset : offset + n].unsqueeze(-1))

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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    julia_estimate = torch.tensor([float(v) for v in lines[0].split()], dtype=torch.float64)
    julia_skill = float(lines[1])

    estimate = cross_map_estimate(y, x, embedding_dim, delay, theiler_window=w + 1)
    n = estimate.shape[0]
    offset = (embedding_dim - 1) * delay
    skill = ccm_skill(estimate, x[offset : offset + n].unsqueeze(-1))

    assert estimate.shape == (julia_estimate.shape[0], 1)
    assert torch.allclose(estimate.squeeze(-1), julia_estimate, atol=1e-9, rtol=0)
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


def test_cross_map_estimate_truncates_to_shorter_series():
    manifold = torch.randn(30, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    result = cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1)
    assert result.shape[0] == 19


def test_cross_map_estimate_rejects_negative_theiler_window():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    try:
        cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cross_map_estimate_rejects_library_size_out_of_range():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(20, dtype=torch.float64)
    for bad in (0, 100):
        try:
            cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1, library_size=bad)
            assert False, f"expected ValueError for library_size={bad}"
        except ValueError:
            pass


def test_cross_map_estimate_rejects_mismatched_batchedness():
    manifold = torch.randn(20, dtype=torch.float64)
    estimated = torch.randn(2, 20, 1, dtype=torch.float64)
    try:
        cross_map_estimate(manifold, estimated, embedding_dim=2, delay=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


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


def test_ccm_skill_gradcheck():
    torch.manual_seed(7)
    estimate = torch.randn(20, dtype=torch.float64, requires_grad=True)
    actual = torch.randn(20, dtype=torch.float64)

    def f(estimate):
        return ccm_skill(estimate, actual)

    assert torch.autograd.gradcheck(f, (estimate,))
