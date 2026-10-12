"""Tests for torchnlts.entropy.generalized.

Dual ground truth:
  1. Paper/math: exact closed-form special cases (Shannon limit at
     order 1, Hartley entropy at Renyi's alpha=0, collision entropy at
     alpha=2, min-entropy at alpha=inf) plus a general theorem (Renyi
     entropy is non-increasing in alpha) checked directly, not just
     against one worked example.
  2. Julia: ComplexityMeasures.jl's `Renyi`/`Tsallis`. Exact match
     across the full parameter range tested, including negative q for
     Tsallis and an explicit zero-probability outcome -- except that
     Julia's `Tsallis` drops k inside its `q ≈ 1` tolerance window when
     k != 1 (see the module docstring); a test below pins the difference.

Also covers numerical stability near order 1 and at extreme orders, and
propagation of NaN rows.
"""

import math

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.entropy.generalized import renyi_entropy, tsallis_entropy


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
    # k scales the true q->1 limit (Julia's own q=1 branch drops k).
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
    torch.manual_seed(2)
    alphas = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 4.0, 10.0, float("inf")]
    for _ in range(5):
        raw = torch.rand(8, dtype=torch.float64)
        probability = raw / raw.sum()
        values = [renyi_entropy(probability, a).item() for a in alphas]
        assert all(values[i] >= values[i + 1] - 1e-10 for i in range(len(values) - 1))


def test_renyi_entropy_rejects_negative_alpha():
    probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        renyi_entropy(probability, -1.0)


def test_tsallis_entropy_allows_negative_q():
    # Zero-probability outcome: a naive 0 base under a negative exponent would give inf, not a masked 0.
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
    # Hartley entropy has no autograd connection to the input (not a zero gradient).
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
    # Regression: alpha=1.00000001 in float32 returned 0 instead of ~0.693147.
    probability = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
    result = renyi_entropy(probability, 1.00000001)
    assert torch.isclose(result, torch.tensor(math.log(2.0), dtype=torch.float32), atol=1e-3)


def test_tsallis_entropy_stable_near_order_1_in_float32():
    probability = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
    result = tsallis_entropy(probability, 1.00000001)
    assert torch.isclose(result, torch.tensor(math.log(2.0), dtype=torch.float32), atol=1e-3)


def test_generalized_entropy_stable_near_order_1_outside_shannon_window_skewed_distribution():
    # Skewed float32 distribution just outside the alpha=1 window; references use plain Python floats.
    probability = torch.tensor([1 - 2**-20, 2**-20, 0.0], dtype=torch.float32)
    reference = {0.9996: 1.4211130565233943e-05, 1.0004: 1.4137819011059937e-05}
    for order, expected in reference.items():
        renyi_result = renyi_entropy(probability, order)
        tsallis_result = tsallis_entropy(probability, order)
        assert torch.isclose(renyi_result, torch.tensor(expected, dtype=torch.float32), rtol=0.01)
        assert torch.isclose(tsallis_result, torch.tensor(expected, dtype=torch.float32), rtol=0.01)


def test_renyi_entropy_isclose_tolerance_does_not_scale_with_probability_dtype():
    # Regression: alpha=0.0003 on float32 data was routed to the alpha=0 Hartley branch;
    # the tolerance follows the order's own (double) precision.
    probability = torch.tensor([1 - 2**-23, 2**-23], dtype=torch.float32, requires_grad=True)
    result = renyi_entropy(probability, 0.0003)
    assert result.requires_grad
    assert torch.isclose(result, torch.tensor(0.6909657716751099, dtype=torch.float32), rtol=1e-3)


def test_renyi_entropy_stable_at_extreme_alpha():
    # Regression: probability**alpha overflowed at alpha=200 (float32) / 2000 (float64); true value is log(2).
    expected = math.log(2.0)
    p32 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    assert torch.isclose(renyi_entropy(p32, 200.0), torch.tensor(expected, dtype=torch.float32), atol=1e-3)
    p64 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert torch.isclose(renyi_entropy(p64, 2000.0), torch.tensor(expected, dtype=torch.float64), atol=1e-6)


def test_tsallis_entropy_stable_at_extreme_q():
    # Reference values by hand: exp(log(2) - log(|q-1|)) scaling.
    p32 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    assert torch.isclose(tsallis_entropy(p32, -128.0), torch.tensor(5.2757e36, dtype=torch.float32), rtol=1e-3)
    p64 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert torch.isclose(tsallis_entropy(p64, -1024.0), torch.tensor(3.5077e305, dtype=torch.float64), rtol=1e-3)


def test_tsallis_entropy_stable_at_extreme_q_and_k_together():
    # k is folded into the same log-space shift.
    probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
    result = tsallis_entropy(probability, -1040.0, k=1e-300)
    assert torch.isclose(result, torch.tensor(2.2634700727e10, dtype=torch.float64), rtol=1e-3)


def test_generalized_entropy_correct_value_for_extreme_order_and_near_subnormal_probability():
    # Regression: clamping the near-1 exponent changed the saturation check and selected the wrong branch.
    probability = torch.softmax(torch.tensor([0.0, -310 * math.log(10.0)], dtype=torch.float64), dim=-1)
    result = renyi_entropy(probability, 0.001)
    assert torch.isclose(result, torch.tensor(0.3990266880503495, dtype=torch.float64), rtol=1e-6)


def test_generalized_entropy_gradcheck_across_extreme_orders():
    # Regression: torch.where backprops through both branches; 0*inf gave NaN gradients (about 1 in 6 combinations).
    # Representative slice of a 675-combination sweep.
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
    # Regression: NaN > 0 is False, so a NaN row was treated as all-zero probability instead of propagating NaN.
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
    # The masked gradient at a zero-probability entry is a convention; for order < 1 the one-sided limit diverges.
    base = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    h0 = renyi_entropy(base, 0.5)
    slopes = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        perturbed = torch.tensor([0.5 - eps / 2, 0.5 - eps / 2, eps], dtype=torch.float64)
        slopes.append(((renyi_entropy(perturbed, 0.5) - h0) / eps).item())
    # Slopes grow as eps shrinks instead of converging to the masked gradient.
    assert slopes[0] < slopes[1] < slopes[2] < slopes[3]
    grad_input = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64, requires_grad=True)
    masked_gradient = torch.autograd.grad(renyi_entropy(grad_input, 0.5), grad_input)[0]
    assert masked_gradient[2].item() == 0.0
    assert slopes[-1] > 100 * masked_gradient[2].abs().item() + 1.0


def test_tsallis_masked_zero_gradient_is_not_the_true_limiting_derivative():
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
    stdout = run_julia(julia_code)
    julia_values = [float(v) for v in stdout.strip().splitlines()]
    torch_probability = torch.tensor(probability, dtype=torch.float64)
    for julia_value, alpha in zip(julia_values, (0.0, 0.5, 1.0, 2.0, 5.0, float("inf")), strict=True):
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
    stdout = run_julia(julia_code)
    julia_values = [float(v) for v in stdout.strip().splitlines()]
    torch_probability = torch.tensor(probability, dtype=torch.float64)
    for julia_value, q in zip(julia_values, (-2.0, -0.5, 0.0, 0.5, 1.0, 2.0), strict=True):
        torch_value = tsallis_entropy(torch_probability, q)
        assert torch.isclose(torch_value, torch.tensor(julia_value, dtype=torch.float64), atol=1e-9, rtol=0)


def test_tsallis_entropy_disagrees_with_complexitymeasures_jl_at_q_1_k_not_1():
    # Julia's Tsallis at exactly q=1 ignores k; the true q->1 limit (and this module) scales with k.
    julia_code = """
    using ComplexityMeasures
    p = Probabilities([0.5, 0.3, 0.2])
    println(information(Tsallis(q=1.0, k=2.0, base=ℯ), p))
    println(information(Tsallis(q=1.0000001, k=2.0, base=ℯ), p))
    """
    stdout = run_julia(julia_code)
    julia_at_exactly_1, julia_near_1 = (float(v) for v in stdout.strip().splitlines())

    probability = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    torch_at_exactly_1 = tsallis_entropy(probability, 1.0, k=2.0).item()

    # Julia's near-1 evaluation scales with k; only its exact-1 branch disagrees.
    assert math.isclose(julia_near_1, torch_at_exactly_1, rel_tol=1e-6)
    assert not math.isclose(julia_at_exactly_1, torch_at_exactly_1, rel_tol=1e-3)


def test_order_one_gradient_is_the_shannon_gradient_with_zero_at_zero_probability():
    probability = torch.tensor([0.5, 0.3, 0.2, 0.0], dtype=torch.float64, requires_grad=True)
    expected = torch.cat([-(probability[:3].detach().log() + 1), torch.zeros(1, dtype=torch.float64)])
    (renyi_gradient,) = torch.autograd.grad(renyi_entropy(probability, 1.0), probability)
    (tsallis_gradient,) = torch.autograd.grad(tsallis_entropy(probability, 1.0, k=2.0), probability)
    assert torch.allclose(renyi_gradient, expected)
    assert torch.allclose(tsallis_gradient, 2.0 * expected)
