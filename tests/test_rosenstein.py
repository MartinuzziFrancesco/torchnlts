"""Tests for torchnlts.lyapunov.rosenstein.

Dual ground truth:
  1. Paper: Rosenstein, Collins & De Luca (1993) -- a hand-derived
     5-point divergence example, a synthetic exact-linear-E(k) test for
     the slope fit alone (mirroring
     torchnlts.dimension.correlation.correlation_dimension's own split
     test design), and the paper's own natural analytic benchmark: the
     logistic map at r=4, whose largest Lyapunov exponent is exactly
     ln(2) (the standard
     hand-derivable case for this kind of estimator).
  2. Julia: ChaosTools.jl's `lyapunov_from_data`, compared at tight
     (float64-precision) tolerance -- matching it required discovering
     and replicating a real, confirmed convention (a reference-point set
     fixed by the sweep's largest k, not maximized independently per k;
     see rosenstein_divergence's docstring), not a loose statistical
     comparison.
"""

import math
import subprocess

import pytest
import torch

from tests.conftest import logistic_map_trajectory
from torchnlts.lyapunov.rosenstein import rosenstein_divergence, rosenstein_lyapunov

JULIA_PROJECT = "juliaenv"


def test_rosenstein_divergence_hand_derived():
    # 5 points, embedding_dim=1 (windows are the raw values), k=1,
    # theiler_window=0. Nearest neighbor (excluding self) per point:
    #   i=0 (0)  -> i=2 (1), dist 1     i=1 (5)  -> i=3 (6), dist 1
    #   i=2 (1)  -> i=0 (0), dist 1     i=3 (6)  -> i=1 (5), dist 1
    #   i=4 (12) -> i=3 (6), dist 6 (excluded below: i=4 has no k=1 step)
    # Evolved (k=1) divergences: |traj[1]-traj[3]|=|5-6|=1 (i=0),
    # |traj[2]-traj[4]|=|1-12|=11 (i=1), |traj[3]-traj[1]|=|6-5|=1 (i=2),
    # |traj[4]-traj[2]|=|12-1|=11 (i=3). E(1) = mean(log(1,11,1,11)).
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, delay=1, theiler_window=0)
    expected = (math.log(1) + math.log(11) + math.log(1) + math.log(11)) / 4
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_rosenstein_divergence_sweep_uses_max_k_reference_set_throughout():
    # Same trajectory as the hand-derived test, k_values=[1, 2]: the
    # reference-point set is fixed by max_k=2 for BOTH entries (see the
    # function's Returns docstring, matching ChaosTools.jl), not
    # independently maximized per k. At max_k=2, only i=0 and i=2 have
    # both a valid reference index (i+2<5) and a neighbor whose own
    # index+2<5 (neighbor_index=[2,3,0,1,3]; i=1's neighbor is 3, and
    # 3+2=5 is not <5, so i=1 -- despite being valid at k=1 alone -- is
    # excluded here too):
    #   k=1: i=0 -> |traj[1]-traj[3]|=|5-6|=1;  i=2 -> |traj[3]-traj[1]|=|6-5|=1
    #        -> E(1) = mean(log(1), log(1)) = 0.0
    #   k=2: i=0 -> |traj[2]-traj[4]|=|1-12|=11; i=2 -> |traj[4]-traj[2]|=|12-1|=11
    #        -> E(2) = mean(log(11), log(11)) = log(11)
    # This deliberately differs from an independent per-k computation,
    # which would use i=0,1,2 at k=1 (see test_rosenstein_divergence_hand_derived's
    # 4-point average) -- a test asserting only the last (max-k) entry
    # would pass even without this fixed-population behavior, so both
    # entries are checked here.
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    k_values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    swept = rosenstein_divergence(trajectory, k_values, embedding_dim=1, delay=1, theiler_window=0)
    expected = torch.tensor([0.0, math.log(11)], dtype=torch.float64)
    assert torch.allclose(swept, expected)

    expected_at_max_k = rosenstein_divergence(trajectory, k_values=2.0, embedding_dim=1, delay=1, theiler_window=0)
    assert torch.isclose(swept[-1], expected_at_max_k)


def test_rosenstein_divergence_batched_and_sweep_shapes():
    # Batching must keep trajectories independent: each batch row's swept
    # curve must match calling that trajectory alone with the same sweep
    # (not compared against independent per-k scalar calls -- see
    # test_rosenstein_divergence_sweep_last_entry_matches_scalar_call_at_max_k
    # for why only the sweep's own largest k has that property).
    torch.manual_seed(0)
    trajectory = torch.randn(4, 40, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    swept = rosenstein_divergence(trajectory, k_values, embedding_dim=2, theiler_window=1)
    assert swept.shape == (3, 4)
    for i in range(4):
        expected = rosenstein_divergence(trajectory[i], k_values, embedding_dim=2, theiler_window=1)
        assert expected.shape == (3,)
        assert torch.allclose(swept[:, i], expected, equal_nan=True)
    expected_at_max_k = rosenstein_divergence(trajectory, k_values=5.0, embedding_dim=2, theiler_window=1)
    assert torch.allclose(swept[-1], expected_at_max_k, equal_nan=True)


def test_rosenstein_divergence_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 30, 2, dtype=torch.float64)
    batched = rosenstein_divergence(trajectory, k_values=2, embedding_dim=2)
    unbatched = rosenstein_divergence(trajectory[0], k_values=2, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_rosenstein_divergence_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=2)
    assert torch.isnan(result)


def test_rosenstein_divergence_k_too_large_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=10, embedding_dim=1, theiler_window=0)
    assert torch.isnan(result)


def test_rosenstein_divergence_zero_evolved_distance_is_neg_inf_not_clamped():
    # Trajectory revisits the same value exactly (0.0 at indices 0, 2, 4),
    # so a reference point's nearest neighbor can evolve to an identical
    # point -- log(0) = -inf, matching log(0) in the cited paper's
    # formula and ChaosTools.jl's own (unclamped) `log(delay_distance(...))`,
    # rather than silently flooring to a large finite value.
    trajectory = torch.tensor([0.0, 5.0, 0.0, 5.0, 0.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=0)
    assert torch.isneginf(result)


def test_rosenstein_divergence_rejects_negative_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=-1, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_rejects_fractional_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=1.5, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_rejects_empty_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=torch.tensor([]), embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_rejects_multidim_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=torch.tensor([[1.0, 2.0]]), embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_rejects_infinite_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=float("inf"), embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_k_validation_survives_float32_rounding():
    # A plain Python float argument must not be silently cast to
    # torch's default (often float32) dtype before validation: -1e-50
    # underflows to -0.0 in float32 (which would pass a `>= 0` check),
    # and 1.00000001 rounds to exactly 1.0 in float32 (which would pass
    # an "is this an integer" check).
    trajectory = torch.arange(5.0, dtype=torch.float64)
    for bad_k in (-1e-50, 1.00000001):
        try:
            rosenstein_divergence(trajectory, k_values=bad_k, embedding_dim=1)
            assert False, f"expected ValueError for k_values={bad_k}"
        except ValueError:
            pass


def test_rosenstein_divergence_absurdly_large_k_is_nan_not_wrong_answer():
    # A k many orders of magnitude past int64 range relative to the
    # trajectory length must not silently wrap around in the bounds-check
    # arithmetic (start_index + max_k) and produce a plausible-looking
    # finite answer -- no point can possibly evolve this far forward.
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=9223372036854775807, embedding_dim=1)
    assert torch.isnan(result)


def test_rosenstein_divergence_rejects_negative_theiler_window():
    # A negative window would admit self via theiler_window_mask's "no
    # exclusion" convention, making every point its own nearest
    # neighbor at distance 0 and trivially "evolving with itself".
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rosenstein_divergence_k_zero_is_original_neighbor_distance():
    # k=0: no evolution, E(0) is just the log of each reference point's
    # own nearest-neighbor distance (the paper's intercept term). At
    # max_k=0 every point's own bounds check is trivially satisfied, so
    # -- unlike k>=1 in test_rosenstein_divergence_hand_derived, where
    # i=4 is excluded -- all 5 points are valid here, including i=4
    # (whose nearest neighbor is i=3 at distance 6, not 1 like the rest).
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=0, embedding_dim=1, theiler_window=0)
    expected = (math.log(1) * 4 + math.log(6)) / 5
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_rosenstein_divergence_gradcheck():
    # Gradient class piecewise: away from a nearest-neighbor tie
    # boundary (this seed/size combination verified not to hit one),
    # the evolved-point distances differentiate normally w.r.t. the
    # trajectory once each reference point's neighbor is fixed.
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def divergence(x):
        return rosenstein_divergence(x, k_values=2, embedding_dim=2, delay=1, theiler_window=0)

    assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_rosenstein_lyapunov_hand_derived_exact_linear():
    # E(k) = 3*k + 1 exactly -> slope 3, exact regression fit (mirrors
    # correlation_dimension's own exact-power-law slope-only test).
    k_values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    divergence = 3.0 * k_values + 1.0
    assert torch.isclose(rosenstein_lyapunov(k_values, divergence), torch.tensor(3.0, dtype=torch.float64))


def test_rosenstein_lyapunov_two_points_is_exact_slope():
    k_values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    divergence = torch.tensor([0.5, 1.3], dtype=torch.float64)
    expected = (divergence[1] - divergence[0]) / (k_values[1] - k_values[0])
    assert torch.isclose(rosenstein_lyapunov(k_values, divergence), expected)


def test_rosenstein_lyapunov_batched_shape():
    torch.manual_seed(2)
    trajectory = torch.randn(3, 40, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, theiler_window=1)
    lyapunov = rosenstein_lyapunov(k_values, divergence)
    assert lyapunov.shape == (3,)
    for i in range(3):
        expected = rosenstein_lyapunov(
            k_values, rosenstein_divergence(trajectory[i], k_values, embedding_dim=2, theiler_window=1)
        )
        assert torch.isclose(lyapunov[i], expected, equal_nan=True)


def test_rosenstein_lyapunov_single_k_is_nan_not_crash():
    result = rosenstein_lyapunov(torch.tensor([1.0]), torch.tensor([0.5]))
    assert torch.isnan(result)


def test_rosenstein_lyapunov_identical_k_is_nan_not_crash():
    result = rosenstein_lyapunov(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.2, 0.5, 0.3]))
    assert torch.isnan(result)


def test_rosenstein_lyapunov_gradcheck_direct_k_and_divergence():
    k_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    divergence = torch.tensor([0.1, 0.5, 0.9, 1.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rosenstein_lyapunov, (k_values, divergence))


def test_rosenstein_matches_analytic_logistic_map_lyapunov_exponent():
    # Rosenstein, Collins & De Luca (1993)'s own natural benchmark:
    # the logistic map x_{n+1} = 4 x_n (1 - x_n) has largest Lyapunov
    # exponent exactly ln(2). A finite-sample kNN estimator won't hit
    # this exactly, so this uses a tolerance appropriate for that,
    # not an exact-value check.
    trajectory = logistic_map_trajectory(2000)
    k_values = torch.arange(1, 12, dtype=torch.float64)
    divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, delay=1, theiler_window=10)
    # Fit range [1, 7]: visually confirmed to be the linear-growth
    # region before E(k) saturates as k grows past it.
    estimate = rosenstein_lyapunov(k_values[:7], divergence[:7])
    assert abs(estimate.item() - math.log(2)) < 0.05


def test_rosenstein_matches_chaostools_jl():
    # Both this module and ChaosTools.jl's lyapunov_from_data use an
    # exact (not approximate) nearest-neighbor search and the same
    # fixed-reference-set-sized-by-max(k) convention (see
    # rosenstein_divergence's Returns docstring, confirmed by reading
    # ChaosTools.jl's source directly and independently reproducing its
    # internal per-point loop in Julia to isolate where a difference
    # would come from before writing this test), so the two curves are
    # expected to agree closely, not just in the same ballpark.
    trajectory = logistic_map_trajectory(300)
    k_values = torch.arange(1, 9, dtype=torch.float64)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ChaosTools
    using DelayEmbeddings
    x = [{values}]
    R = embed(x, 2, 1)
    E = lyapunov_from_data(R, 1:8; w=10, ntype=NeighborNumber(1), distance=Euclidean())
    println(join(E, ","))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_divergence = torch.tensor([float(v) for v in result.stdout.strip().split(",")], dtype=torch.float64)
    torch_divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, delay=1, theiler_window=10)
    # atol=1e-9, rtol=0 (torch.allclose's default rtol=1e-5 would dwarf
    # atol here and hide a real mismatch): both sides do the same
    # brute-force/tree Euclidean nearest-neighbor search and log/mean
    # reduction in float64 on identical data, so the residual is
    # float64 parsed-string round-trip noise, not model mismatch.
    assert torch.allclose(torch_divergence, julia_divergence, atol=1e-9, rtol=0)

    julia_slope = rosenstein_lyapunov(k_values, julia_divergence)
    torch_slope = rosenstein_lyapunov(k_values, torch_divergence)
    assert torch.isclose(julia_slope, torch_slope, atol=1e-9, rtol=0)


def test_rosenstein_divergence_out_of_range_preserves_shape_and_dtype():
    for dtype in (torch.float32, torch.float64):
        for shape in ((5,), (5, 1), (1, 5, 1), (2, 5, 1)):
            trajectory = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
            for large_k in (5, 2**63 - 1, 2**64, 10**30, 1e300):
                for k_values, normal_k in ((large_k, 0), ([0, large_k], [0, 1])):
                    result = rosenstein_divergence(trajectory, k_values, embedding_dim=1)
                    normal = rosenstein_divergence(trajectory, normal_k, embedding_dim=1)
                    assert result.shape == normal.shape
                    assert result.dtype == normal.dtype == dtype
                    assert result.device == normal.device
                    assert torch.isnan(result).all()


def test_rosenstein_divergence_large_theiler_window_excludes_self():
    trajectory = torch.arange(5.0, dtype=torch.float64).reshape(1, 5, 1)
    for window in (2**63 - 1, 2**63, 2**64, 10**30):
        result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1, theiler_window=window)
        assert result.shape == (2, 1)
        assert torch.isnan(result).all()


def test_rosenstein_divergence_one_copy_embedding_ignores_large_delay():
    trajectory = torch.tensor([0., 5., 1., 6., 12.], dtype=torch.float64)
    expected = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1)
    result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1, delay=10**30)
    assert torch.equal(result, expected)


def test_rosenstein_divergence_embedding_and_validation_edges():
    trajectory = torch.arange(5.0, dtype=torch.float32).reshape(1, 5, 1)
    for options in ({'embedding_dim': 0}, {'delay': 0}, {'embedding_dim': 4, 'delay': 2},
                    {'embedding_dim': 2, 'delay': 10**30}):
        with pytest.raises(ValueError):
            rosenstein_divergence(trajectory, 0, **options)
    for bad_k in (-1e-50, 1.00000001, float('inf'), float('-inf'), float('nan')):
        with pytest.raises(ValueError):
            rosenstein_divergence(trajectory, [0, bad_k])
    result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=3, delay=2)
    assert result.shape == (2, 1)
    assert result.dtype == trajectory.dtype
    assert torch.isnan(result).all()


def test_rosenstein_divergence_matches_independent_loop_for_delays_and_norms():
    # Build embeddings and choose neighbors directly, without the production
    # embedding, kNN, gathering, or reduction helpers.
    torch.manual_seed(31)
    trajectory = torch.randn(2, 18, 2, dtype=torch.float64)
    k_values = [2, 0, 1, 2]
    for embedding_dim, delay in ((1, 1), (2, 3), (3, 2)):
        for norm_order in (1., 2., float('inf')):
            expected = []
            for batch in trajectory:
                embedded = [torch.cat([batch[start + offset * delay] for offset in range(embedding_dim)])
                            for start in range(len(batch) - (embedding_dim - 1) * delay)]
                logs = [[] for _ in k_values]
                for start, point in enumerate(embedded):
                    candidates = [index for index in range(len(embedded)) if abs(index - start) > 4]
                    neighbor = min(candidates, key=lambda index: (point - embedded[index]).norm(p=norm_order).item())
                    nearest_distance = (point - embedded[neighbor]).norm(p=norm_order)
                    assert sum((point - embedded[index]).norm(p=norm_order) == nearest_distance for index in candidates) == 1
                    if max(start, neighbor) + max(k_values) >= len(embedded):
                        continue
                    for position, step in enumerate(k_values):
                        logs[position].append((embedded[start + step] - embedded[neighbor + step]).norm(p=norm_order).log())
                expected.append(torch.stack([torch.stack(values).mean() for values in logs]))
            result = rosenstein_divergence(trajectory, k_values, embedding_dim, delay, 4, norm_order)
            assert torch.allclose(result, torch.stack(expected, dim=1), atol=1e-12, rtol=0)


def test_rosenstein_divergence_batched_sweep_gradcheck():
    torch.manual_seed(32)
    trajectory = torch.randn(1, 12, 2, dtype=torch.float64, requires_grad=True)
    for norm_order in (1., 2., float('inf')):
        def divergence(values):
            return rosenstein_divergence(values, [0, 2], embedding_dim=2, delay=2, norm_order=norm_order)
        assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_rosenstein_divergence_small_separation_amplifies_gradient():
    trajectory = torch.tensor([0., 5., 1., 6., 12.], dtype=torch.float64, requires_grad=True)
    result = rosenstein_divergence(trajectory, 0, embedding_dim=1)
    gradient = torch.autograd.grad(result, trajectory)[0]
    small = (trajectory.detach() * 1e-12).requires_grad_()
    small_result = rosenstein_divergence(small, 0, embedding_dim=1)
    small_gradient = torch.autograd.grad(small_result, small)[0]
    assert torch.allclose(small_gradient * 1e-12, gradient, atol=1e-12, rtol=0)
    assert small_gradient.abs().max() > 1e11


def test_rosenstein_divergence_one_duplicate_pair_is_neg_inf():
    trajectory = torch.tensor([0., 0., 4., 10., 17.], dtype=torch.float64)
    assert torch.isneginf(rosenstein_divergence(trajectory, 0, embedding_dim=1))


def test_rosenstein_lyapunov_rejects_mismatched_sweep_shapes():
    for k_values, divergence in (([0, 1, 2], torch.tensor([7.])),
                                 ([[0, 1]], torch.tensor([1., 2.])),
                                 ([0, 1], torch.ones(2, 1, 1)),
                                 (0, torch.tensor(1.))):
        with pytest.raises(ValueError, match='matching'):
            rosenstein_lyapunov(k_values, divergence)


def test_rosenstein_zero_distance_row_does_not_poison_shared_gradient():
    # Row 0 is constant: every nearest-neighbor distance is exactly 0, so its
    # curve is -inf (documented) and its slope NaN. log(0)'s infinite
    # gradient must not reach a parameter shared with row 1.
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    trajectories = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 1.0, 6.0, 12.0]], dtype=torch.float64).unsqueeze(-1)
    curve = rosenstein_divergence(parameter * trajectories, [0, 1], embedding_dim=1)
    assert torch.isinf(curve[:, 0]).all()
    result = rosenstein_lyapunov([0, 1], curve)
    assert torch.isnan(result[0]) and torch.isfinite(result[1])
    (gradient,) = torch.autograd.grad(result[1], parameter)
    # The slope of log-distances is invariant to rescaling the trajectory.
    assert torch.isclose(gradient, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_rosenstein_divergence_rejects_complex_k_values():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0])
    for k_values in (torch.tensor([0.0 + 2j, 1.0 + 3j]), [0, 1j]):
        with pytest.raises(ValueError, match="real"):
            rosenstein_divergence(trajectory, k_values)


def test_rosenstein_divergence_high_norm_order_small_separations():
    # (1e-15 / 10) ** 40 underflows float64 unless each pair is scaled by itself.
    ulp = math.ulp(10.0)
    trajectory = torch.tensor([[0.0, 0.0], [1e-15, 1e-15], [10.0, 10.0], [10.0 + 2 * ulp, 10.0 + 2 * ulp]], dtype=torch.float64)
    expected = (math.log(1e-15) + math.log(2 * ulp)) / 2 + math.log(2.0) / 40
    assert math.isclose(rosenstein_divergence(trajectory, 0, embedding_dim=1, norm_order=40).item(), expected, rel_tol=1e-12)
