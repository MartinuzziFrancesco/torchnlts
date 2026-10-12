"""Tests for torchnlts.entropy.approximate.

Dual ground truth:
  1. Paper: Pincus (1991), eq. 1-4 -- hand-derived hard and soft examples
     (the hard one additionally cross-checked at runtime against
     ComplexityMeasures.jl on the same random-data setup elsewhere in
     this file, with matching parameters; the soft one against a
     from-scratch pure-Python oracle computed once, before writing the
     test, and hardcoded here as the expected value) plus a hand-derived
     self-excluded ("modified ApEn") degenerate case.
  2. Julia: ComplexityMeasures.jl's ``ApproximateEntropy`` (self-inclusive
     by construction, matching this module's default ``theiler_window=-1``)
     -- except at ``embedding_dim=1``, where that package deliberately
     special-cases its own definition (see the module docstring and the
     locked disagreement test below).
"""

import math

import pytest
import torch

from tests.conftest import assert_first_row_gradients_unaffected, run_julia
from torchnlts.entropy.approximate import approximate_entropy, soft_approximate_entropy

NONFINITE = [float("nan"), float("inf"), -float("inf")]
GAP = dict(embedding_dim=1, delay=30)


def test_approximate_entropy_hand_derived():
    # m=2 windows (1,2),(2,1),(1,3), r=1.5: conditional probabilities 1, 2/3, 2/3 (self included).
    # phi(2) = mean(log p); the two m=3 windows are at distance 2 > r, so phi(3) = log(1/2).
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1)
    assert torch.isclose(result, torch.tensor(0.4228371084878357, dtype=torch.float64))


def test_approximate_entropy_delay_2():
    # Lag-2 windows (1,1),(2,3),(1,5),(3,0), r=2.5, self-inclusive: C = [3/4, 3/4, 1/2, 1/2];
    # the two m=3 windows are at distance 5 > r, so phi(3) = log(1/2). An ignored delay would change the value.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 5.0, 0.0], dtype=torch.float64)
    embedded = torch.tensor([[1.0, 1.0], [2.0, 3.0], [1.0, 5.0], [3.0, 0.0]], dtype=torch.float64)
    from torchnlts.embedding.delay import delay_embedding

    assert torch.equal(delay_embedding(trajectory, embedding_dim=2, delay=2), embedded)
    result = approximate_entropy(trajectory, radius=2.5, embedding_dim=2, delay=2)
    phi_2 = (2 * math.log(0.75) + 2 * math.log(0.5)) / 4
    phi_3 = math.log(0.5)
    assert torch.isclose(result, torch.tensor(phi_2 - phi_3, dtype=torch.float64))


def test_approximate_entropy_inclusive_radius_boundary():
    # radius=1.0 equals the smaller nonzero distance, so this pins the inclusive <=.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    at_boundary = approximate_entropy(trajectory, radius=1.0, embedding_dim=2, delay=1)
    above_boundary = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1)
    assert torch.isclose(at_boundary, above_boundary)


def test_log_conditional_probability_positive_theiler_window_unequal_row_denominators():
    # m=1 via the private row helper, theiler_window=1, r=1.5: counts/denominators 1/3, 1/2, 2/2, 1/2, 1/3;
    # the unequal per-row denominators catch a shared denominator.
    from torchnlts.entropy.approximate import _log_conditional_probability

    trajectory = torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0], dtype=torch.float64)
    embedded = trajectory.unsqueeze(-1)
    result = _log_conditional_probability(embedded, radius=1.5, theiler_window=1, norm_order=float("inf"))
    expected = (2 * math.log(1 / 3) + 2 * math.log(1 / 2) + math.log(1.0)) / 5
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_approximate_entropy_self_excluded_degenerate_case():
    # theiler_window=0: the two m=3 windows have no match within r, so phi(3) = log(0) and ApEn = +inf.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=1.5, embedding_dim=2, delay=1, theiler_window=0)
    assert torch.isinf(result) and result > 0


def test_approximate_entropy_radius_sweep():
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float64)
    result = approximate_entropy(trajectory, radius=torch.tensor([1.5, 3.0], dtype=torch.float64))
    expected_at_1_5 = approximate_entropy(trajectory, radius=1.5)
    expected_at_3_0 = approximate_entropy(trajectory, radius=3.0)
    assert torch.allclose(result, torch.stack([expected_at_1_5, expected_at_3_0]))


def test_approximate_entropy_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 30, 2, dtype=torch.float64)
    radii = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    swept = approximate_entropy(trajectory, radius=radii, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, radius in enumerate(radii):
        expected = approximate_entropy(trajectory, radius=radius, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected, equal_nan=True)


def test_approximate_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    unbatched = approximate_entropy(trajectory[0], radius=1.0, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_approximate_entropy_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0])
    result = approximate_entropy(trajectory, radius=0.5, embedding_dim=1, theiler_window=1)
    assert torch.isnan(result)


def test_approximate_entropy_vmap_compatible():
    torch.manual_seed(2)
    trajectory = torch.randn(2, 20, 1, dtype=torch.float64)
    direct = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    vmapped = torch.vmap(lambda t: approximate_entropy(t, radius=1.0, embedding_dim=2))(trajectory)
    assert torch.allclose(direct, vmapped, equal_nan=True)


def test_approximate_entropy_nan_input_gives_nan_output():
    # NaN must propagate, not raise: a data-dependent `if` would break torch.vmap.
    trajectory = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(approximate_entropy(trajectory, radius=1.5, embedding_dim=2))


def test_approximate_entropy_dtype_preserved_float32():
    torch.manual_seed(7)
    trajectory = torch.randn(20, dtype=torch.float32)
    result = approximate_entropy(trajectory, radius=1.0, embedding_dim=2)
    assert result.dtype == torch.float32


@pytest.mark.parametrize("function", [approximate_entropy, soft_approximate_entropy])
def test_approximate_entropies_follow_the_input_dtype(function):
    torch.manual_seed(8)
    reference = torch.randn(60)
    integers = (4 * reference).round().long()
    result = function(integers, 0.5, embedding_dim=2)
    assert result.dtype == torch.get_default_dtype()
    assert torch.equal(result, function(integers.to(torch.get_default_dtype()), 0.5, embedding_dim=2))
    for dtype in (torch.float16, torch.bfloat16):
        half = reference.to(dtype)
        result = function(half, 0.5, embedding_dim=2)
        assert result.dtype == dtype and torch.isfinite(result)
        assert torch.equal(result, function(half.float(), 0.5, embedding_dim=2).to(dtype))
    with pytest.raises(ValueError, match="complex"):
        function(torch.ones(20, dtype=torch.complex64), 0.5)


def test_approximate_entropy_embedding_dim_one_disagrees_with_causalitytools_jl():
    # ComplexityMeasures.jl special-cases m=1 (returns phi(1) alone), so this module's Pincus definition differs.
    trajectory = [0.0, 1.0, 2.0]
    radius = 0.5
    julia_code = f"""
    using ComplexityMeasures
    x = {trajectory}
    est = ApproximateEntropy(m=1, τ=1, r={radius})
    println(complexity(est, x))
    """
    stdout = run_julia(julia_code)
    julia_apen = torch.tensor(float(stdout.strip()), dtype=torch.float64)
    torch_apen = approximate_entropy(torch.tensor(trajectory, dtype=torch.float64), radius=radius, embedding_dim=1)
    # phi(1) alone (Julia) is -ln(3); phi(1) - phi(2) (Pincus, this module) is ln(2/3).
    assert torch.isclose(julia_apen, torch.tensor(-math.log(3), dtype=torch.float64))
    assert torch.isclose(torch_apen, torch.tensor(math.log(2 / 3), dtype=torch.float64))


def test_soft_approximate_entropy_low_temperature_limit_matches_radius_zero_hard():
    trajectory = torch.tensor([0.0, 1.0, 0.3, 1.0], dtype=torch.float64)
    soft = soft_approximate_entropy(trajectory, temperature=1e-6)
    hard = approximate_entropy(trajectory, radius=0.0)
    assert torch.allclose(soft, hard, atol=1e-4)


def test_soft_approximate_entropy_hand_derived_finite_temperature():
    # Gaussian kernel w(d) = exp(-(d/T)^2/2), T=1, on [0, 1, 3]; phi(m) = mean_i log mean_j w, derived in pure Python.
    # m=2 windows are (0,1),(1,3) at Chebyshev distance 2; ApEn = phi(1) - phi(2).
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    result = soft_approximate_entropy(trajectory, temperature=1.0, embedding_dim=1)
    assert torch.isclose(result, torch.tensor(-0.14152971049673713, dtype=torch.float64))


def test_soft_approximate_entropy_nan_input_gives_nan_output():
    # The kernel gives a NaN distance weight 0, so NaN must be propagated explicitly.
    trajectory = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2))


def test_soft_approximate_entropy_batched_and_sweep_shapes():
    torch.manual_seed(8)
    trajectory = torch.randn(4, 20, 2, dtype=torch.float64)
    temperatures = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    swept = soft_approximate_entropy(trajectory, temperature=temperatures, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, temperature in enumerate(temperatures):
        expected = soft_approximate_entropy(trajectory, temperature=temperature, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected)


def test_soft_approximate_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(9)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2)
    unbatched = soft_approximate_entropy(trajectory[0], temperature=0.5, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_soft_approximate_entropy_vmap_compatible():
    torch.manual_seed(10)
    trajectory = torch.randn(2, 20, 1, dtype=torch.float64)
    direct = soft_approximate_entropy(trajectory, temperature=0.5, embedding_dim=2)
    vmapped = torch.vmap(lambda t: soft_approximate_entropy(t, temperature=0.5, embedding_dim=2))(trajectory)
    assert torch.allclose(direct, vmapped)


def test_soft_approximate_entropy_gradcheck():
    torch.manual_seed(3)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_approximate_entropy(x, temperature=0.5, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_approximate_entropy_gradcheck_exponential():
    torch.manual_seed(4)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_approximate_entropy(x, temperature=0.5, embedding_dim=2, kind="exponential")

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_approximate_entropy_gradcheck_temperature():
    torch.manual_seed(5)
    trajectory = torch.randn(15, dtype=torch.float64)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_approximate_entropy(trajectory, temperature=t, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (temperature,))


def test_approximate_entropy_matches_complexitymeasures_jl():
    torch.manual_seed(6)
    trajectory = torch.randn(30, dtype=torch.float64)
    radius = 1.0
    embedding_dim = 2

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = ApproximateEntropy(m={embedding_dim}, τ=1, r={radius})
    println(complexity(est, x))
    """
    stdout = run_julia(julia_code)
    julia_apen = torch.tensor(float(stdout.strip()), dtype=torch.float64)
    torch_apen = approximate_entropy(trajectory, radius=radius, embedding_dim=embedding_dim)
    # rtol=atol=1e-9: both sides compute in float64; the residual is string round-trip noise.
    assert torch.isclose(torch_apen, julia_apen, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("function", [approximate_entropy, soft_approximate_entropy])
@pytest.mark.parametrize("bad", NONFINITE)
def test_approximate_entropies_nan_for_nonfinite_sample_anywhere(function, bad):
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = bad
    values = function(batch, 0.5, **GAP)
    assert torch.isnan(values[1]) and torch.isfinite(values[[0, 2]]).all()
    for row in (0, 2):
        assert torch.equal(values[row], function(batch[row], 0.5, **GAP))
    assert torch.isnan(function(batch[1], 0.5, **GAP)) and torch.isnan(function(batch[1, :, 0], 0.5, **GAP))
    swept = function(batch, torch.tensor([0.5, 0.8], dtype=torch.float64), **GAP)
    assert swept.shape == (2, 3) and torch.isnan(swept[:, 1]).all() and not torch.isnan(swept[:, [0, 2]]).any()


@pytest.mark.parametrize("options", [dict(embedding_dim=2), GAP], ids=["inside_window", "delay_30"])
def test_soft_approximate_entropy_nonfinite_row_leaves_other_gradients(options):
    torch.manual_seed(1)
    clean, bad_row = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    bad_row[25] = float("nan")

    def gradients(rows):
        batch = torch.stack(rows).unsqueeze(-1).requires_grad_()
        temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        soft_approximate_entropy(batch, temperature, **options).nansum().backward()
        return batch.grad, temperature.grad

    batch_grad, temperature_grad = gradients([bad_row, clean])
    single_grad, single_temperature_grad = gradients([clean])
    assert torch.isfinite(batch_grad).all() and (batch_grad[0] == 0).all()
    assert torch.allclose(batch_grad[1], single_grad[0])
    assert torch.isfinite(temperature_grad) and torch.allclose(temperature_grad, single_temperature_grad)


@pytest.mark.parametrize("function", [approximate_entropy, soft_approximate_entropy])
def test_approximate_entropies_vmap_with_nonfinite_row(function):
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = float("nan")
    direct = function(batch, 0.5, **GAP)
    mapped = torch.vmap(lambda series: function(series, 0.5, **GAP))(batch)
    assert torch.allclose(direct, mapped, equal_nan=True) and torch.isnan(mapped[1]) and torch.isfinite(mapped[0])


# Two templates of the quadratic series are at least 10 apart, so every kernel weight at a bandwidth of 0.1 underflows.
SMALL_SERIES = [0.00, 0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.06, -0.01, 0.00, 0.04, -0.03]
QUADRATIC_SERIES = [10.0 * index**2 for index in range(12)]
UNDERFLOW_CASES = [
    # No window of the second row has a match at either length: phi(m) and phi(m+1) are both -inf.
    ([SMALL_SERIES, QUADRATIC_SERIES], dict(theiler_window=0), math.nan),
    # Every window of [0, 0, 5, 5] matches at length 1 but none at length 2: only phi(m+1) is -inf.
    ([[0.0, 0.05, 0.02, 0.07], [0.0, 0.0, 5.0, 5.0]], dict(embedding_dim=1, theiler_window=0), math.inf),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows, options, expected", UNDERFLOW_CASES, ids=["both_phi_infinite", "long_phi_infinite"])
def test_soft_approximate_entropy_underflowed_kernel_sums_keep_other_gradients_finite(rows, options, expected, dtype):
    # log(0) of an underflowed kernel sum must not put NaN into the gradient of the shared bandwidth.
    result = assert_first_row_gradients_unaffected(
        lambda series, bandwidth: soft_approximate_entropy(series, bandwidth, **options), rows, 0.1, dtype
    )
    assert torch.isfinite(result[0])
    torch.testing.assert_close(result[1], torch.tensor(expected, dtype=dtype), equal_nan=True)


@pytest.mark.parametrize("rows, options, expected", UNDERFLOW_CASES, ids=["both_phi_infinite", "long_phi_infinite"])
def test_approximate_entropy_unmatched_windows_give_the_documented_nan_and_inf(rows, options, expected):
    result = approximate_entropy(torch.tensor(rows, dtype=torch.float64).unsqueeze(-1), 0.1, **options)
    assert torch.isfinite(result[0])
    torch.testing.assert_close(result[1], torch.tensor(expected, dtype=torch.float64), equal_nan=True)
