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

# CausalityTools.jl conflicts with the shared juliaenv's FractalDimensions.jl pin (checked by forcing both into one
# environment), so it has its own project, unlike the other Julia dual-validation tests.
JULIA_PROJECT = "juliaenv_causality"


def test_mutual_information_hand_derived():
    # k=1, w=0. Joint nn distances ds = [2, 2, 2, 18]; strict counts nx = [2, 2, 1, 4], ny = [2, 2, 3, 1].
    # KSG1 (eq. 8): mi = psi(1) + psi(4) - mean(psi(nx) + psi(ny)) = 0.
    x = torch.tensor([0.0, 1.0, 3.0, 10.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=1, theiler_window=0)
    assert torch.isclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_mutual_information_hand_derived_theiler_window():
    # k=1, w=1. Joint distances ds = [10, 1, 1, 1, 1]; strict counts nx = ny = [3, 1, 1, 1, 1].
    # mi = psi(1) + psi(5) - mean(psi(nx) + psi(ny)) = H_4 - 2*H_2/5 = 89/60, using psi(m) = H_(m-1) - Euler's constant.
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
    x = torch.randn(10, dtype=torch.float64)
    y = torch.randn(10, dtype=torch.float64)
    try:
        mutual_information(x, y, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mutual_information_nan_input_gives_nan_output():
    x = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert torch.isnan(mutual_information(x, y, num_neighbors=1))


def test_mutual_information_nan_in_discarded_tail_is_harmless():
    clean = torch.arange(4, dtype=torch.float64)
    longer = torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan")], dtype=torch.float64)
    for x, y in ((clean, longer), (longer, clean)):
        assert torch.isclose(
            mutual_information(x, y, num_neighbors=1),
            mutual_information(clean, clean, num_neighbors=1),
        )


def test_mutual_information_vmap_compatible():
    z = torch.zeros((2, 4, 1), dtype=torch.float64)
    direct = mutual_information(z, z, num_neighbors=1)
    vmapped = torch.vmap(lambda a, b: mutual_information(a, b, num_neighbors=1))(z, z)
    assert torch.allclose(direct, vmapped)


def test_mutual_information_zero_radius_uses_formula_self_correction():
    # All joint points coincide: nx = ny = 0, so mi = psi(1) + psi(4) - 2*psi(1) = H_3 = 11/6 (a formula value, not a
    # real MI).
    z = torch.zeros(4, dtype=torch.float64)
    expected = torch.tensor(11 / 6, dtype=torch.float64)
    assert torch.isclose(mutual_information(z, z, num_neighbors=1), expected, atol=1e-12, rtol=0)
    # Radii [0, 0, 2, 3]: self-inclusive counts are [1, 1, 1, 1] in both marginals, giving the same formula value.
    mixed = torch.tensor([0.0, 0.0, 2.0, 5.0], dtype=torch.float64)
    batch = torch.stack((z, mixed)).unsqueeze(-1)
    assert torch.allclose(mutual_information(batch, batch, num_neighbors=1), expected.expand(2), atol=1e-12, rtol=0)


def test_mutual_information_insufficient_neighbors_is_nan():
    z = torch.arange(4, dtype=torch.float64)
    assert torch.isnan(mutual_information(z, z, num_neighbors=4))
    assert torch.isnan(mutual_information(z, z, num_neighbors=5))
    assert torch.isfinite(mutual_information(z, z, num_neighbors=3))


def test_mutual_information_theiler_window_starves_interior_rows_to_nan():
    # Interior rows 1, 2 have one eligible neighbor (fewer than k=2); a single starved row must NaN the whole average.
    z = torch.arange(4, dtype=torch.float64)
    assert torch.isnan(mutual_information(z, z, num_neighbors=2, theiler_window=1))
    assert torch.isfinite(mutual_information(z, z, num_neighbors=1, theiler_window=1))


def test_mutual_information_norm_order_changes_the_estimate():
    torch.manual_seed(0)
    x = torch.randn(60, 2, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    chebyshev = mutual_information(x, y, num_neighbors=4, norm_order=float("inf"))
    euclidean = mutual_information(x, y, num_neighbors=4, norm_order=2.0)
    assert not torch.isclose(chebyshev, euclidean)


def test_mutual_information_matches_analytic_gaussian_correlation():
    # Kraskov et al. (2004), Sec. III: bivariate Gaussian MI is -0.5*ln(1 - rho^2); KSG1 is only asymptotically
    # unbiased, hence the loose tolerance.
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
    # CausalityTools.jl counts marginals inclusively (<=): nx = [2, 3, 2, 4], ny = [3, 3, 3, 2], mi = -0.875 (vs 0.0
    # here). If it ever switches to strict counting this fails, meaning the documented discrepancy is stale.
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


def _naive_ksg2(x, y, num_neighbors, julia_variant=False, theiler_window=0):
    # Loop reference for KSG2 (Chebyshev, 1-D marginals; Theiler window on the joint search only).
    # julia_variant: marginal radius from the k-th joint neighbor only, and "- 1" in place of "- 1/k".
    num = len(x)
    total = 0.0
    for i in range(num):
        joint = sorted((max(abs(x[i] - x[j]), abs(y[i] - y[j])), j) for j in range(num) if abs(i - j) > theiler_window)
        neighbors = [j for _, j in joint[:num_neighbors]]
        if julia_variant:
            neighbors = neighbors[-1:]
        for series in (x, y):
            radius = max(abs(series[i] - series[j]) for j in neighbors)
            count = sum(1 for j in range(num) if j != i and abs(series[i] - series[j]) <= radius)
            total += float(torch.digamma(torch.tensor(float(count), dtype=torch.float64)))
    digamma = lambda value: float(torch.digamma(torch.tensor(float(value), dtype=torch.float64)))
    correction = 1.0 if julia_variant else 1.0 / num_neighbors
    return digamma(num_neighbors) - correction + digamma(num) - total / num


def test_mutual_information_ksg2_hand_derived():
    # k=1. Joint nn: 0->1, 1->0, 2->1, 3->2, so |dx| = [1, 1, 3, 6], |dy| = [2, 2, 1, 19]; inclusive nx = [1, 1, 1, 1],
    # ny = [2, 2, 2, 2]. I2 = psi(1) - 1 - (psi(1) + psi(2)) + psi(4) = psi(4) - psi(2) - 1 = -1/6.
    x = torch.tensor([0.0, 1.0, 4.0, 10.0], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=1, estimator="ksg2")
    assert torch.isclose(result, torch.tensor(-1 / 6, dtype=torch.float64), atol=1e-12, rtol=0)


def test_mutual_information_ksg2_matches_naive_reference():
    # k=3 exercises the max over the k neighbors' marginal extents.
    torch.manual_seed(5)
    x = torch.randn(40, dtype=torch.float64)
    y = 0.6 * x + torch.randn(40, dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=3, estimator="ksg2")
    assert abs(result.item() - _naive_ksg2(x.tolist(), y.tolist(), 3)) < 1e-10


def test_mutual_information_ksg2_theiler_window_excludes_joint_neighbors_only():
    # w=1: joint neighbors skip i-1, i, i+1; marginal counts still see them (only self is excluded).
    torch.manual_seed(6)
    x = torch.cumsum(torch.randn(30, dtype=torch.float64), dim=0)
    y = torch.cumsum(torch.randn(30, dtype=torch.float64), dim=0)
    with_window = mutual_information(x, y, num_neighbors=2, theiler_window=3, estimator="ksg2")
    without = mutual_information(x, y, num_neighbors=2, theiler_window=0, estimator="ksg2")
    assert torch.isfinite(with_window) and not torch.isclose(with_window, without)


def test_mutual_information_default_estimator_is_ksg1_and_invalid_raises():
    torch.manual_seed(0)
    x = torch.randn(30, dtype=torch.float64)
    y = torch.randn(30, dtype=torch.float64)
    assert torch.equal(mutual_information(x, y), mutual_information(x, y, estimator="ksg1"))
    try:
        mutual_information(x, y, estimator="ksg3")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mutual_information_ksg2_batched_nan_and_starvation():
    torch.manual_seed(7)
    x = torch.randn(3, 40, 1, dtype=torch.float64)
    y = torch.randn(3, 40, 1, dtype=torch.float64)
    batched = mutual_information(x, y, num_neighbors=3, estimator="ksg2")
    assert batched.shape == (3,)
    for i in range(3):
        assert torch.isclose(batched[i], mutual_information(x[i], y[i], num_neighbors=3, estimator="ksg2"))
    x[1, 5, 0] = float("nan")
    x[2, 5, 0] = float("inf")
    out = mutual_information(x, y, num_neighbors=3, estimator="ksg2")
    assert torch.isfinite(out[0]) and torch.isnan(out[1]) and torch.isnan(out[2])
    z = torch.arange(4, dtype=torch.float64)
    assert torch.isnan(mutual_information(z, z, num_neighbors=4, estimator="ksg2"))
    assert torch.isnan(mutual_information(z, z, num_neighbors=5, estimator="ksg2"))
    assert torch.isfinite(mutual_information(z, z, num_neighbors=3, estimator="ksg2"))
    assert torch.isnan(mutual_information(z, z, num_neighbors=2, theiler_window=1, estimator="ksg2"))
    vmapped = torch.vmap(lambda a, b: mutual_information(a, b, num_neighbors=3, estimator="ksg2"))(
        torch.randn(2, 20, 1, dtype=torch.float64), torch.randn(2, 20, 1, dtype=torch.float64)
    )
    assert vmapped.shape == (2,)


def test_mutual_information_both_estimators_match_analytic_gaussian():
    torch.manual_seed(3)
    rho = 0.7
    z1 = torch.randn(4000, dtype=torch.float64)
    z2 = torch.randn(4000, dtype=torch.float64)
    y = rho * z1 + math.sqrt(1 - rho**2) * z2
    true_mi = -0.5 * math.log(1 - rho**2)
    for estimator in ("ksg1", "ksg2"):
        result = mutual_information(z1, y, num_neighbors=5, estimator=estimator)
        assert abs(result.item() - true_mi) < 0.05, estimator


def _julia_ksg2(x, y, num_neighbors):
    julia_code = f"""
    using CausalityTools
    x = {x.tolist()}
    y = {y.tolist()}
    println(association(KSG2(MIShannon(base=MathConstants.e); k={num_neighbors}, w=0), x, y))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip())


def test_mutual_information_ksg2_matches_causalitytools_jl_at_k1_and_deviates_for_larger_k():
    # KSG2 matches CausalityTools.jl at k=1; for k>1 the package uses only the k-th neighbor and "- 1" (paper: "- 1/k").
    # This module follows the paper; the Julia value is locked to the loop reference of that variant.
    torch.manual_seed(8)
    x = torch.randn(40, dtype=torch.float64)
    y = 0.6 * x + torch.randn(40, dtype=torch.float64)
    assert abs(_julia_ksg2(x, y, 1) - mutual_information(x, y, num_neighbors=1, estimator="ksg2").item()) < 1e-10
    julia_k3 = _julia_ksg2(x, y, 3)
    assert abs(julia_k3 - _naive_ksg2(x.tolist(), y.tolist(), 3, julia_variant=True)) < 1e-10
    assert abs(julia_k3 - mutual_information(x, y, num_neighbors=3, estimator="ksg2").item()) > 0.01


def test_mutual_information_ksg2_theiler_matches_loop_reference():
    torch.manual_seed(9)
    x = torch.cumsum(torch.randn(40, dtype=torch.float64), dim=0)
    y = x + torch.randn(40, dtype=torch.float64)
    result = mutual_information(x, y, num_neighbors=2, theiler_window=3, estimator="ksg2")
    assert abs(result.item() - _naive_ksg2(x.tolist(), y.tolist(), 2, theiler_window=3)) < 1e-10


def test_mutual_information_ksg2_preserves_float32():
    torch.manual_seed(10)
    x = torch.randn(40, dtype=torch.float32)
    y = 0.6 * x + torch.randn(40, dtype=torch.float32)
    result = mutual_information(x, y, num_neighbors=3, estimator="ksg2")
    assert result.dtype == torch.float32
    assert abs(result.item() - _naive_ksg2(x.double().tolist(), y.double().tolist(), 3)) < 1e-4


def test_mutual_information_ksg2_vmap_matches_direct_with_nan_row():
    torch.manual_seed(11)
    x = torch.randn(3, 20, 1, dtype=torch.float64)
    y = torch.randn(3, 20, 1, dtype=torch.float64)
    x[1, 3, 0] = float("nan")
    direct = mutual_information(x, y, num_neighbors=3, estimator="ksg2")
    vmapped = torch.vmap(lambda a, b: mutual_information(a, b, num_neighbors=3, estimator="ksg2"))(x, y)
    assert torch.allclose(direct, vmapped, equal_nan=True)
    assert torch.isnan(vmapped[1]) and torch.isfinite(vmapped[0])


def test_mutual_information_rejects_non_integer_num_neighbors():
    x = torch.randn(10, dtype=torch.float64)
    for estimator in ("ksg1", "ksg2"):
        for bad in (1.5, float("nan"), float("inf")):
            try:
                mutual_information(x, x, num_neighbors=bad, estimator=estimator)
                assert False, "expected ValueError"
            except ValueError:
                pass


def test_mutual_information_ksg2_small_scale_deviates_from_causalitytools_jl():
    # CausalityTools.jl adds 10*eps() to each marginal radius, which changes counts for data near 1e-16 even at k=1;
    # this module is scale-free.
    scale = 1e-16
    x = torch.tensor([0.0, 1.0, 4.0, 10.0], dtype=torch.float64) * scale
    y = torch.tensor([0.0, 2.0, 1.0, 20.0], dtype=torch.float64) * scale
    assert torch.isclose(
        mutual_information(x, y, num_neighbors=1, estimator="ksg2"), torch.tensor(-1 / 6, dtype=torch.float64)
    )
    assert abs(_julia_ksg2(x, y, 1) - (-13 / 6)) < 1e-9
