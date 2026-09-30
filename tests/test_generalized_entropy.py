"""Tests for torchnlts.entropy.generalized.

Dual ground truth:
  1. Paper/math: exact closed-form special cases (Shannon limit at
     order 1, Hartley entropy at Renyi's alpha=0, collision entropy at
     alpha=2, min-entropy at alpha=inf) plus a general theorem (Renyi
     entropy is non-increasing in alpha) checked directly, not just
     against one worked example.
  2. Julia: ComplexityMeasures.jl's `Renyi`/`Tsallis`. Exact match
     across the full parameter range tested, including negative q for
     Tsallis and an explicit zero-probability outcome -- except one
     confirmed deviation, a genuine discontinuity across Julia's own
     `q ≈ 1` tolerance window when k != 1 (see module docstring), locked
     in below.

Also locks in regressions for numerical-stability bugs in earlier versions
of this module (exact-equality-only order-1 special casing, and direct
power-sum computation instead of `logsumexp`), plus a composition gap (NaN
silently read as zero probability).
"""

import math
import subprocess

import torch

from torchnlts.entropy.generalized import renyi_entropy, tsallis_entropy

JULIA_PROJECT = "juliaenv"


def _shannon(probability):
    nonzero = probability > 0
    safe = torch.where(nonzero, probability, torch.ones_like(probability))
    return -(probability * torch.log(safe)).sum(dim=-1)


def test_renyi_entropy_alpha_1_is_shannon():
    torch.manual_seed(0)
    raw = torch.rand(6, dtype=torch.float64)
    probability = raw / raw.sum()
    assert torch.isclose(renyi_entropy(probability, 1.0), _shannon(probability))


def test_tsallis_entropy_q_1_is_shannon():
    torch.manual_seed(1)
    raw = torch.rand(6, dtype=torch.float64)
    probability = raw / raw.sum()
    assert torch.isclose(tsallis_entropy(probability, 1.0), _shannon(probability))
    # k scales the true q->1 limit (unlike ComplexityMeasures.jl's own
    # q=1 branch, which silently drops k -- see module docstring).
    assert torch.isclose(tsallis_entropy(probability, 1.0, k=2.5), 2.5 * _shannon(probability))


def test_renyi_entropy_alpha_0_is_hartley():
    probability = torch.tensor([0.5, 0.3, 0.0, 0.2], dtype=torch.float64)
    assert torch.isclose(renyi_entropy(probability, 0.0), torch.log(torch.tensor(3.0, dtype=torch.float64)))


def test_renyi_entropy_alpha_2_is_collision_entropy():
    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    expected = -torch.log((probability**2).sum())
    assert torch.isclose(renyi_entropy(probability, 2.0), expected)


def test_renyi_entropy_alpha_inf_is_min_entropy():
    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    expected = -torch.log(probability.max())
    assert torch.isclose(renyi_entropy(probability, float("inf")), expected)


def test_renyi_entropy_is_nonincreasing_in_alpha():
    # Renyi's own theorem: H_alpha is non-increasing in alpha, for any
    # probability distribution -- checked here on several random ones,
    # not just the one worked example above.
    torch.manual_seed(2)
    alphas = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 4.0, 10.0, float("inf")]
    for _ in range(5):
        raw = torch.rand(8, dtype=torch.float64)
        probability = raw / raw.sum()
        values = [renyi_entropy(probability, a).item() for a in alphas]
        assert all(values[i] >= values[i + 1] - 1e-10 for i in range(len(values) - 1))


def test_renyi_entropy_rejects_negative_alpha():
    probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    try:
        renyi_entropy(probability, -1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_tsallis_entropy_allows_negative_q():
    # Unlike Renyi, Tsallis has no sign restriction here (matching
    # ComplexityMeasures.jl) -- exercised with an explicit
    # zero-probability outcome, since that's exactly the case the
    # safe-base-then-mask construction (module docstring) exists for:
    # naively substituting 0 as the base before a negative exponent
    # would give inf, not a masked-out 0.
    probability = torch.tensor([0.5, 0.3, 0.2, 0.0], dtype=torch.float64)
    result = tsallis_entropy(probability, -2.0)
    assert torch.isfinite(result)


def test_renyi_gradcheck():
    torch.manual_seed(3)
    raw = torch.rand(5, dtype=torch.float64, requires_grad=True)
    for alpha in (0.5, 1.0, 2.0, 5.0, float("inf")):

        def entropy(r, alpha=alpha):
            return renyi_entropy(r / r.sum(), alpha)

        assert torch.autograd.gradcheck(entropy, (raw,))


def test_renyi_alpha_0_has_no_gradient_path():
    # Confirmed directly (module docstring): Hartley entropy is a
    # purely combinatorial function of probability, not a continuous
    # one -- its output has no autograd connection to the input at
    # all, distinct from a genuinely zero-valued gradient.
    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64, requires_grad=True)
    result = renyi_entropy(probability, 0.0)
    assert not result.requires_grad
    assert result.grad_fn is None


def test_tsallis_gradcheck():
    torch.manual_seed(4)
    raw = torch.rand(5, dtype=torch.float64, requires_grad=True)
    for q in (-1.0, 0.0, 0.5, 1.0, 2.0):

        def entropy(r, q=q):
            return tsallis_entropy(r / r.sum(), q, k=1.5)

        assert torch.autograd.gradcheck(entropy, (raw,))


def test_renyi_entropy_batched():
    torch.manual_seed(5)
    raw = torch.rand(4, 6, dtype=torch.float64)
    probability = raw / raw.sum(dim=-1, keepdim=True)
    batched = renyi_entropy(probability, 2.0)
    assert batched.shape == (4,)
    for i in range(4):
        assert torch.isclose(batched[i], renyi_entropy(probability[i], 2.0))


def test_renyi_entropy_stable_near_order_1_in_float32():
    # Regression: an earlier version special-cased alpha=1 only at *exact*
    # equality, leaving a floating-point cancellation gap nearby --
    # alpha=1.00000001 in float32 returned 0 instead of ~0.693147.
    probability = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
    result = renyi_entropy(probability, 1.00000001)
    assert torch.isclose(result, torch.tensor(math.log(2.0), dtype=torch.float32), atol=1e-3)


def test_tsallis_entropy_stable_near_order_1_in_float32():
    probability = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
    result = tsallis_entropy(probability, 1.00000001)
    assert torch.isclose(result, torch.tensor(math.log(2.0), dtype=torch.float32), atol=1e-3)


def test_generalized_entropy_stable_near_order_1_outside_shannon_window_skewed_distribution():
    # A *skewed* float32 distribution at an order outside the alpha=1
    # tolerance window (0.9996/1.0004, not 1.00000001) still suffers severe
    # cancellation in the general formula itself; a logsumexp-only version
    # was off by ~93% here. The near-1 expm1/log1p reformulation (see
    # module docstring) fixes it. Reference values are computed with plain
    # Python floats (double precision), not this module.
    probability = torch.tensor([1 - 2**-20, 2**-20, 0.0], dtype=torch.float32)
    reference = {0.9996: 1.4211130565233943e-05, 1.0004: 1.4137819011059937e-05}
    for order, expected in reference.items():
        renyi_result = renyi_entropy(probability, order)
        tsallis_result = tsallis_entropy(probability, order)
        assert torch.isclose(renyi_result, torch.tensor(expected, dtype=torch.float32), rtol=0.01)
        assert torch.isclose(tsallis_result, torch.tensor(expected, dtype=torch.float32), rtol=0.01)


def test_renyi_entropy_isclose_tolerance_does_not_scale_with_probability_dtype():
    # Regression: an earlier version scaled the alpha ≈ 0/1 tolerance by
    # probability.dtype's own eps, matching Julia's convention for a
    # *Float32* order but not for a plain Python float order (always double
    # precision), wrongly routing alpha=0.0003 on float32 data into the
    # alpha=0 Hartley branch (losing both the value and the gradient path).
    # The order parameter's own (double) precision determines the tolerance,
    # not probability's dtype.
    probability = torch.tensor([1 - 2**-23, 2**-23], dtype=torch.float32, requires_grad=True)
    result = renyi_entropy(probability, 0.0003)
    assert result.requires_grad
    assert torch.isclose(result, torch.tensor(0.6909657716751099, dtype=torch.float32), rtol=1e-3)


def test_renyi_entropy_stable_at_extreme_alpha():
    # Regression: an earlier version formed probability**alpha directly,
    # overflowing to inf at alpha=200 (float32) or alpha=2000 (float64)
    # even though the true entropy (log(2), the min-entropy of a uniform
    # 2-outcome distribution) is far from either dtype's representable
    # range. The logsumexp reformulation never forms probability**alpha in
    # linear space.
    expected = math.log(2.0)
    p32 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    assert torch.isclose(renyi_entropy(p32, 200.0), torch.tensor(expected, dtype=torch.float32), atol=1e-3)
    p64 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert torch.isclose(renyi_entropy(p64, 2000.0), torch.tensor(expected, dtype=torch.float64), atol=1e-6)


def test_tsallis_entropy_stable_at_extreme_q():
    # A log-space shift of the 1/(order-1) scaling (applied *before*
    # exponentiating, not after -- see module docstring) rescues q=-128
    # (float32) and q=-1024 (float64), which overflow to inf with the
    # extreme-alpha fix above alone. Reference values are from the module
    # docstring's derivation, checked by hand via
    # exp(log(2) - log(|q-1|)) scaling.
    p32 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    assert torch.isclose(tsallis_entropy(p32, -128.0), torch.tensor(5.2757e36, dtype=torch.float32), rtol=1e-3)
    p64 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert torch.isclose(tsallis_entropy(p64, -1024.0), torch.tensor(3.5077e305, dtype=torch.float64), rtol=1e-3)


def test_tsallis_entropy_stable_at_extreme_q_and_k_together():
    # The shift above still overflows once k is also extreme if k is
    # multiplied in after exponentiating, so k is folded into the same
    # log-space shift. Reference value from the module docstring's
    # derivation.
    probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    result = tsallis_entropy(probability, -1040.0, k=1e-300)
    assert torch.isclose(result, torch.tensor(2.2634700727e10, dtype=torch.float64), rtol=1e-3)


def test_generalized_entropy_correct_value_for_extreme_order_and_near_subnormal_probability():
    # Regression: clamping the near-1 formula's internal exponent before
    # exp/expm1 (to keep its local gradient finite when discarded) broke
    # the *forward value* in the cases it was meant to protect. The clamp
    # also changed u itself, which the saturation check uses to decide
    # whether to trust this branch, so a saturated case could look
    # well-behaved and select the wrong branch. Saturation is detected
    # from the *unclamped* exponent, computed separately, not from the
    # (already safe) u alone.
    probability = torch.softmax(torch.tensor([0.0, -310 * math.log(10.0)], dtype=torch.float64), dim=-1)
    result = renyi_entropy(probability, 0.001)
    assert torch.isclose(result, torch.tensor(0.3990266880503495, dtype=torch.float64), rtol=1e-6)


def test_generalized_entropy_gradcheck_across_extreme_orders():
    # torch.where backprops through BOTH branches of the near-1/extreme-
    # order hybrid above regardless of which is selected, and each
    # branch's local gradient can blow up exactly where the *other* branch
    # is selected, poisoning the total gradient via 0*inf=NaN even though
    # the forward VALUE is correct (same class of bug as in rosenstein.py).
    # The fix substitutes a safe dummy into whichever branch is discarded.
    # Without it, this sweep failed for roughly 1 in 6 (order, seed)
    # combinations. This test keeps a representative slice of a broader
    # 675-combination sweep (which had zero failures after the fix); the
    # full sweep is too slow for routine runs.
    def make_prob(raw):
        return raw / raw.sum()

    torch.manual_seed(7)
    raw = torch.rand(6, dtype=torch.float64, requires_grad=True)
    for alpha in (10.0, 15.0, 20.0, 30.0, 50.0):

        def renyi_at(r, alpha=alpha):
            return renyi_entropy(make_prob(r), alpha)

        assert torch.autograd.gradcheck(renyi_at, (raw,))
    for q in (-30.0, -10.0, 10.0, 30.0):

        def tsallis_at(r, q=q):
            return tsallis_entropy(make_prob(r), q, k=1.5)

        assert torch.autograd.gradcheck(tsallis_at, (raw,))


def test_renyi_entropy_propagates_nan_probability_row():
    # Regression: NaN > 0 is False in IEEE 754, so the zero-probability
    # masking logic treated a NaN row (e.g. ordinal_pattern_probabilities's
    # convention for a degenerate trajectory) as "every outcome has zero
    # probability" instead of propagating NaN (giving inf/-inf/wrong finite
    # values depending on alpha).
    probability = torch.full((5,), float("nan"), dtype=torch.float64)
    for alpha in (0.0, 0.5, 1.0, 2.0, float("inf")):
        assert torch.isnan(renyi_entropy(probability, alpha))


def test_tsallis_entropy_propagates_nan_probability_row():
    probability = torch.full((5,), float("nan"), dtype=torch.float64)
    for q in (-1.0, 0.0, 0.5, 1.0, 2.0):
        assert torch.isnan(tsallis_entropy(probability, q))


def test_renyi_entropy_nan_does_not_poison_other_batch_rows():
    torch.manual_seed(6)
    raw = torch.rand(6, dtype=torch.float64)
    valid = raw / raw.sum()
    invalid = torch.full((6,), float("nan"), dtype=torch.float64)
    batched = torch.stack([valid, invalid])
    result = renyi_entropy(batched, 2.0)
    assert torch.isclose(result[0], renyi_entropy(valid, 2.0))
    assert torch.isnan(result[1])


def test_renyi_masked_zero_gradient_is_not_the_true_limiting_derivative():
    # The masked gradient of 0 at an exact-zero probability entry is a
    # *convention* (this outcome's probability is fixed at zero), not the
    # one-sided limiting derivative as that outcome's probability grows
    # from zero -- for order < 1 that limit is unbounded. Numerically,
    # (H(eps) - H(0)) / eps diverges as eps shrinks rather than converging
    # to the masked gradient's value.
    base = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    h0 = renyi_entropy(base, 0.5)
    slopes = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        perturbed = torch.tensor([0.5 - eps / 2, 0.5 - eps / 2, eps], dtype=torch.float64)
        slopes.append(((renyi_entropy(perturbed, 0.5) - h0) / eps).item())
    # Each successive (smaller-eps) slope estimate is larger, consistent
    # with divergence -- not converging toward the masked gradient of 1.
    assert slopes[0] < slopes[1] < slopes[2] < slopes[3]
    grad_input = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64, requires_grad=True)
    masked_gradient = torch.autograd.grad(renyi_entropy(grad_input, 0.5), grad_input)[0]
    assert masked_gradient[2].item() == 0.0
    assert slopes[-1] > 100 * masked_gradient[2].abs().item() + 1.0  # diverging, nowhere near the masked 0


def test_tsallis_masked_zero_gradient_is_not_the_true_limiting_derivative():
    # Same property, checked independently for Tsallis rather than
    # assumed to carry over from Renyi's own check above.
    base = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    h0 = tsallis_entropy(base, 0.5)
    slopes = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        perturbed = torch.tensor([0.5 - eps / 2, 0.5 - eps / 2, eps], dtype=torch.float64)
        slopes.append(((tsallis_entropy(perturbed, 0.5) - h0) / eps).item())
    assert slopes[0] < slopes[1] < slopes[2] < slopes[3]
    grad_input = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64, requires_grad=True)
    masked_gradient = torch.autograd.grad(tsallis_entropy(grad_input, 0.5), grad_input)[0]
    assert masked_gradient[2].item() == 0.0
    assert slopes[-1] > 100 * masked_gradient[2].abs().item() + 1.0


def test_renyi_entropy_matches_complexitymeasures_jl():
    probability = [0.5, 0.3, 0.2]
    values = ", ".join(f"{v:.17g}" for v in probability)
    julia_code = f"""
    using ComplexityMeasures
    p = Probabilities([{values}])
    for q in (0.0, 0.5, 1.0, 2.0, 5.0, Inf)
        println(information(Renyi(q=q, base=ℯ), p))
    end
    """
    result = subprocess.run(["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    julia_values = [float(v) for v in result.stdout.strip().splitlines()]
    torch_probability = torch.tensor(probability, dtype=torch.float64)
    for julia_value, alpha in zip(julia_values, (0.0, 0.5, 1.0, 2.0, 5.0, float("inf"))):
        torch_value = renyi_entropy(torch_probability, alpha)
        assert torch.isclose(torch_value, torch.tensor(julia_value, dtype=torch.float64), atol=1e-9, rtol=0)


def test_tsallis_entropy_matches_complexitymeasures_jl_including_negative_q():
    probability = [0.5, 0.3, 0.2, 0.0]
    values = ", ".join(f"{v:.17g}" for v in probability)
    julia_code = f"""
    using ComplexityMeasures
    p = Probabilities([{values}])
    for q in (-2.0, -0.5, 0.0, 0.5, 1.0, 2.0)
        println(information(Tsallis(q=q, k=1.0, base=ℯ), p))
    end
    """
    result = subprocess.run(["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    julia_values = [float(v) for v in result.stdout.strip().splitlines()]
    torch_probability = torch.tensor(probability, dtype=torch.float64)
    for julia_value, q in zip(julia_values, (-2.0, -0.5, 0.0, 0.5, 1.0, 2.0)):
        torch_value = tsallis_entropy(torch_probability, q)
        assert torch.isclose(torch_value, torch.tensor(julia_value, dtype=torch.float64), atol=1e-9, rtol=0)


def test_tsallis_entropy_disagrees_with_complexitymeasures_jl_at_q_1_k_not_1():
    # Locks in the confirmed deviation the module docstring describes:
    # Julia's Tsallis at exactly q=1 silently ignores k (its result is
    # identical for k=1 and k=2), while the true q->1 limit of its own
    # general formula (and this module's implementation) scales with
    # k -- confirmed live by evaluating Julia's general-formula branch
    # at q=1.0000001, which *does* scale correctly with k.
    julia_code = """
    using ComplexityMeasures
    p = Probabilities([0.5, 0.3, 0.2])
    println(information(Tsallis(q=1.0, k=2.0, base=ℯ), p))
    println(information(Tsallis(q=1.0000001, k=2.0, base=ℯ), p))
    """
    result = subprocess.run(["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    julia_at_exactly_1, julia_near_1 = (float(v) for v in result.stdout.strip().splitlines())

    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    torch_at_exactly_1 = tsallis_entropy(probability, 1.0, k=2.0).item()

    # Julia's own near-1 evaluation already reflects the true, k-scaled
    # limit; only its *exact*-1 branch disagrees with that limit.
    assert math.isclose(julia_near_1, torch_at_exactly_1, rel_tol=1e-6)
    assert not math.isclose(julia_at_exactly_1, torch_at_exactly_1, rel_tol=1e-3)
