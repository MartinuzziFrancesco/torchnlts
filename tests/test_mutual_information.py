"""Tests for torchnlts.entropy.mutual_information.

Dual ground truth:
  1. Paper: Kraskov, Stogbauer & Grassberger (2004) -- hand-derived
     4-point (Theiler window 0) and 5-point (Theiler window 1) examples
     worked directly from eq. 6-8's strict-inequality definition, plus
     their own Sec. III validation approach (bivariate correlated
     Gaussians, whose true mutual information -0.5*ln(1-rho^2) is known
     analytically).
  2. Julia: CausalityTools.jl's ``KSG1`` estimator. This module's strict
     marginal counting is a confirmed, deliberate deviation from that
     estimator's inclusive convention (see the module docstring), so
     "Julia ground truth" here means a locked-in disagreement test with
     independently hand-verified values on both sides, not a numerical
     match -- picking an arbitrarily loose tolerance to paper over a
     confirmed finite-sample difference would be worse than
     admitting it directly.
"""

import math
import subprocess

import torch

from torchnlts.entropy.mutual_information import mutual_information

# CausalityTools.jl's dependency constraints are incompatible with the
# shared juliaenv's pinned FractalDimensions.jl version (a real conflict,
# not just an unresolved upgrade -- confirmed by attempting to force both
# into one environment), so it lives in its own project instead of
# juliaenv, unlike every other Julia dual-validation test in this repo.
JULIA_PROJECT = "juliaenv_causality"


def test_mutual_information_hand_derived():
    # 4 points, k=1, w=0. Joint (Chebyshev-max-of-marginals) distances to
    # each point's nearest neighbor (excluding self): ds = [2, 2, 2, 18].
    # Strict-inequality marginal counts (eq. 6-7's n_x(i)/n_y(i), self
    # trivially included since its own distance 0 is always < a positive
    # radius): nx = [2, 2, 1, 4], ny = [2, 2, 3, 1] -- e.g. row 1's
    # dx = [1, 0, 2, 9] against radius 2 excludes the boundary neighbor
    # at distance exactly 2 (strict "<", unlike an inclusive "<=").
    # KSG1 (eq. 8): mi = psi(1) + psi(4) - mean(psi(nx) + psi(ny)) = 0
    # exactly. See test_mutual_information_disagrees_with_causalitytools_jl
    # for the same input under CausalityTools.jl's inclusive convention.
    x = torch.tensor([0.0, 1.0, 3.0, 10.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=1, theiler_window=0)
    assert torch.isclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_mutual_information_hand_derived_theiler_window():
    # 5 points, k=1, w=1 (excludes self and immediate index neighbors
    # from the joint-space search only). Joint distances:
    #   row0 -> ds=10 (nearest eligible is index 2)
    #   row1..4 -> ds=1 (each other's nearest eligible index)
    # Strict marginal counts at those radii: nx = ny = [3, 1, 1, 1, 1]
    # (row 0's radius is wide enough to admit 3 points per marginal;
    # rows 1-4 admit only themselves). mi = psi(1) + psi(5) -
    # mean(psi(nx) + psi(ny)) = H_4 - 2*H_2/5 = 89/60,
    # using psi(m) = H_(m-1) - Euler's constant.
    x = torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 10.0, 2.0, 11.0], dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=1, theiler_window=1)
    assert torch.isclose(result, torch.tensor(1.483333333333333, dtype=torch.float64))


def test_mutual_information_batched_matches_unbatched():
    torch.manual_seed(0)
    x = torch.randn(3, 50, 1, dtype=torch.float64)
    y = torch.randn(3, 50, 1, dtype=torch.float64)
    batched = mutual_information(x, y, num_neighbors=3)
    assert batched.shape == (3,)
    for i in range(3):
        expected = mutual_information(x[i], y[i], num_neighbors=3)
        assert torch.isclose(batched[i], expected)


def test_mutual_information_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    x = torch.randn(1, 30, 1, dtype=torch.float64)
    y = torch.randn(1, 30, 1, dtype=torch.float64)
    batched = mutual_information(x, y, num_neighbors=3)
    unbatched = mutual_information(x[0], y[0], num_neighbors=3)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_mutual_information_truncates_to_shorter_length():
    torch.manual_seed(2)
    x = torch.randn(30, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    assert torch.isclose(
        mutual_information(x, y, num_neighbors=3),
        mutual_information(x, y[:30], num_neighbors=3),
    )


def test_mutual_information_rejects_invalid_num_neighbors():
    x = torch.randn(10, dtype=torch.float64)
    y = torch.randn(10, dtype=torch.float64)
    try:
        mutual_information(x, y, num_neighbors=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mutual_information_rejects_negative_theiler_window():
    # A negative window would admit self via theiler_window_mask's
    # "no exclusion" convention, making every point its own nearest
    # neighbor at distance 0 and collapsing the whole estimator.
    x = torch.randn(10, dtype=torch.float64)
    y = torch.randn(10, dtype=torch.float64)
    try:
        mutual_information(x, y, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mutual_information_nan_input_gives_nan_output():
    # A raise on NaN would need a data-dependent Python `if` on the
    # tensor's contents, which rejects torch.vmap's BatchedTensor inputs
    # outright -- so NaN input degrades to the same documented NaN
    # output as an insufficient neighbor count instead (see
    # test_mutual_information_vmap_compatible).
    x = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert torch.isnan(mutual_information(x, y, num_neighbors=1))


def test_mutual_information_nan_in_discarded_tail_is_harmless():
    # NaN validation runs after truncation to the shorter length, so a
    # NaN beyond that point -- never actually used -- doesn't affect
    # the result, matching permutation_entropy's convention of
    # validating the data that's actually used rather than the raw input.
    clean = torch.arange(4, dtype=torch.float64)
    longer = torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan")], dtype=torch.float64)
    for x, y in ((clean, longer), (longer, clean)):
        assert torch.isclose(
            mutual_information(x, y, num_neighbors=1),
            mutual_information(clean, clean, num_neighbors=1),
        )


def test_mutual_information_vmap_compatible():
    # NaN detection is a plain tensor op (torch.where), not a Python
    # `if` branching on tensor content, so it survives being called
    # under vmap (a data-dependent `if torch.isnan(...).any(): raise`
    # would reject vmap's BatchedTensor inputs outright).
    z = torch.zeros((2, 4, 1), dtype=torch.float64)
    direct = mutual_information(z, z, num_neighbors=1)
    vmapped = torch.vmap(lambda a, b: mutual_information(a, b, num_neighbors=1))(z, z)
    assert torch.allclose(direct, vmapped)


def test_mutual_information_zero_radius_uses_formula_self_correction():
    # All four joint points coincide: n_x = n_y = 0 under strict <.
    # Eq. 8 therefore gives psi(1) + psi(4) - 2*psi(1) = H_3 = 11/6.
    # This is a formal formula value, not a meaningful continuous MI.
    z = torch.zeros(4, dtype=torch.float64)
    expected = torch.tensor(11 / 6, dtype=torch.float64)
    assert torch.isclose(mutual_information(z, z, num_neighbors=1), expected, atol=1e-12, rtol=0)
    # Mixed zero/positive radii: [0, 0, 2, 3]; self-inclusive counts
    # are [1, 1, 1, 1] in both marginals, giving the same formula value.
    mixed = torch.tensor([0.0, 0.0, 2.0, 5.0], dtype=torch.float64)
    batch = torch.stack((z, mixed)).unsqueeze(-1)
    assert torch.allclose(mutual_information(batch, batch, num_neighbors=1), expected.expand(2), atol=1e-12, rtol=0)


def test_mutual_information_insufficient_neighbors_is_nan():
    # N=4, k=4: excluding self leaves only 3 candidates, one short of
    # the requested 4th-nearest-neighbor radius.
    z = torch.arange(4, dtype=torch.float64)
    assert torch.isnan(mutual_information(z, z, num_neighbors=4))
    assert torch.isnan(mutual_information(z, z, num_neighbors=5))
    # k=3 (exactly enough) stays finite.
    assert torch.isfinite(mutual_information(z, z, num_neighbors=3))


def test_mutual_information_theiler_window_starves_interior_rows_to_nan():
    # N=4, w=1, k=2: interior rows (index 1, 2) have only one eligible
    # joint-space neighbor left after excluding self and both immediate
    # index neighbors, one short of k=2 -- even though edge rows (0, 3)
    # have two. A single starved row must NaN the whole average rather
    # than silently average in a wrong, treat-inf-as-a-real-radius value
    # for that row.
    z = torch.arange(4, dtype=torch.float64)
    assert torch.isnan(mutual_information(z, z, num_neighbors=2, theiler_window=1))
    # k=1 leaves every row with at least one eligible neighbor.
    assert torch.isfinite(mutual_information(z, z, num_neighbors=1, theiler_window=1))


def test_mutual_information_norm_order_changes_the_estimate():
    # norm_order sets a single Minkowski order shared by both marginals
    # and (through their elementwise max) the joint metric -- changing
    # it from the default Chebyshev should change the estimate.
    torch.manual_seed(0)
    x = torch.randn(60, 2, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    chebyshev = mutual_information(x, y, num_neighbors=4, norm_order=float("inf"))
    euclidean = mutual_information(x, y, num_neighbors=4, norm_order=2.0)
    assert not torch.isclose(chebyshev, euclidean)


def test_mutual_information_matches_analytic_gaussian_correlation():
    # Kraskov et al. (2004), Sec. III: for bivariate Gaussian (X, Y)
    # with correlation rho, the true mutual information is
    # -0.5*ln(1 - rho^2). KSG1 is only asymptotically unbiased, so this
    # uses a loose tolerance appropriate for a finite-sample estimator,
    # not an exact-value check.
    torch.manual_seed(3)
    num_samples = 4000
    rho = 0.7
    z1 = torch.randn(num_samples, dtype=torch.float64)
    z2 = torch.randn(num_samples, dtype=torch.float64)
    x = z1
    y = rho * z1 + math.sqrt(1 - rho**2) * z2
    result = mutual_information(x, y, num_neighbors=5)
    true_mi = -0.5 * math.log(1 - rho**2)
    assert abs(result.item() - true_mi) < 0.05


def test_mutual_information_near_zero_for_independent_variables():
    torch.manual_seed(4)
    x = torch.randn(4000, dtype=torch.float64)
    y = torch.randn(4000, dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=5)
    assert abs(result.item()) < 0.03


def test_mutual_information_disagrees_with_causalitytools_jl():
    # Same input as test_mutual_information_hand_derived. Locks in the
    # module docstring's claimed discrepancy: CausalityTools.jl's KSG1
    # counts marginal neighbors within an *inclusive* (<=) radius
    # (confirmed by directly probing Neighborhood.jl's `inrangecount` at
    # an exact boundary distance), giving nx = [2, 3, 2, 4], ny =
    # [3, 3, 3, 2] on this input (each 1 higher than this module's
    # strict nx/ny wherever a neighbor sits exactly at the joint
    # radius) and mi = -0.875, versus this module's paper-faithful 0.0.
    # If a future CausalityTools.jl version switches to strict counting,
    # this test will start failing, which is the point -- it means the
    # documented discrepancy has gone stale and should be revisited.
    x = torch.tensor([0.0, 1.0, 3.0, 10.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64)
    julia_code = """
    using CausalityTools
    x = [0.0, 1.0, 3.0, 10.0]
    y = [0.0, 2.0, 1.0, 20.0]
    println(association(KSG1(MIShannon(base=MathConstants.e); k=1, w=0), x, y))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_mi = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_mi = mutual_information(x, y, num_neighbors=1, theiler_window=0)
    assert torch.isclose(julia_mi, torch.tensor(-0.875, dtype=torch.float64))
    assert torch.isclose(torch_mi, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert not torch.isclose(torch_mi, julia_mi, atol=0.1)
