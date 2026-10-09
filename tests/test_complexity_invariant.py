"""Tests for torchnlts.neighbors.complexity_invariant.

Ground truth: Batista et al. (2014) formulas, hand-derived values, and tsfresh's ``cid_ce`` for the
complexity estimate (with and without normalization). No installable package (aeon, tslearn, tsfresh)
implements the full CID, so the distance is checked against ``euclidean * cid_ce`` ratios.
"""

import math

import pytest
import torch

from torchnlts.neighbors import complexity_estimate, complexity_invariant_distance


def _f64(values):
    return torch.tensor(values, dtype=torch.float64)


def test_complexity_estimate_hand_derived():
    # diffs of [1, 3, 2, 6] are 2, -1, 4 -> sqrt(4 + 1 + 16) = sqrt(21)
    assert torch.isclose(complexity_estimate(_f64([1.0, 3.0, 2.0, 6.0])), _f64(math.sqrt(21)))


def test_complexity_estimate_normalized_hand_derived():
    # [0, 2, 0, 2]: mean 1, population std 1 -> [-1, 1, -1, 1], diffs 2, -2, 2 -> sqrt(12)
    assert torch.isclose(complexity_estimate(_f64([0.0, 2.0, 0.0, 2.0]), normalize=True), _f64(math.sqrt(12)))


def test_cid_hand_derived():
    # Q = [0, 0, 0, 3] (CE 3), C = [0, 1, 0, 1] (CE sqrt(3)); Q - C = [0, -1, 0, 2] -> ED = sqrt(5);
    # CF = 3 / sqrt(3) = sqrt(3); CID = sqrt(15)
    first, second = _f64([0.0, 0.0, 0.0, 3.0]), _f64([0.0, 1.0, 0.0, 1.0])
    expected = _f64(math.sqrt(15))
    assert torch.isclose(complexity_invariant_distance(first, second), expected)
    assert torch.isclose(complexity_invariant_distance(second, first), expected)


def test_cid_zero_complexity_conventions():
    constant_a, constant_b = _f64([1.0] * 5), _f64([4.0] * 5)
    wiggly = _f64([0.0, 1.0, 0.0, 1.0, 0.0])
    # both CE zero: CF = 1, CID = ED
    assert torch.isclose(complexity_invariant_distance(constant_a, constant_b), _f64(3.0 * math.sqrt(5)))
    assert complexity_invariant_distance(constant_a, constant_a) == 0
    # one CE zero: infinite
    assert complexity_invariant_distance(constant_a, wiggly) == math.inf
    assert complexity_invariant_distance(wiggly, wiggly) == 0


def test_cid_is_euclidean_for_equal_complexity():
    first = _f64([0.0, 1.0, 0.0, 1.0])
    second = first + 5.0
    assert torch.isclose(complexity_invariant_distance(first, second), _f64(10.0))


@pytest.mark.parametrize("normalize", [False, True])
def test_complexity_estimate_matches_tsfresh(normalize):
    from tsfresh.feature_extraction.feature_calculators import cid_ce

    torch.manual_seed(0)
    batch = torch.randn(4, 50, 1, dtype=torch.float64).cumsum(dim=1)
    result = complexity_estimate(batch, normalize=normalize)
    for row in range(4):
        assert torch.isclose(result[row], _f64(cid_ce(batch[row, :, 0].numpy(), normalize)))
    constant = torch.full((10,), 2.0, dtype=torch.float64)
    assert complexity_estimate(constant, normalize=normalize) == cid_ce(constant.numpy(), normalize) == 0


def test_cid_matches_tsfresh_ratio():
    from tsfresh.feature_extraction.feature_calculators import cid_ce

    torch.manual_seed(1)
    first, second = torch.randn(2, 60, dtype=torch.float64)
    euclidean = torch.linalg.vector_norm(first - second)
    complexities = [cid_ce(series.numpy(), False) for series in (first, second)]
    expected = euclidean * max(complexities) / min(complexities)
    assert torch.isclose(complexity_invariant_distance(first, second), expected)


def test_shapes_and_broadcasting():
    torch.manual_seed(2)
    batch = torch.randn(3, 20, 1, dtype=torch.float64)
    other = torch.randn(3, 20, 1, dtype=torch.float64)
    assert complexity_estimate(batch[0, :, 0]).shape == ()
    assert complexity_estimate(batch[0]).shape == ()
    assert complexity_estimate(batch).shape == (3,)
    paired = complexity_invariant_distance(batch, other)
    assert paired.shape == (3,)
    for row in range(3):
        assert torch.equal(paired[row], complexity_invariant_distance(batch[row], other[row]))
    broadcast = complexity_invariant_distance(batch[0], other)
    assert broadcast.shape == (3,)
    assert torch.equal(broadcast, complexity_invariant_distance(batch[:1], other))
    assert complexity_invariant_distance(batch[0], other[0]).shape == ()


def test_errors():
    with pytest.raises(ValueError, match="equal length"):
        complexity_invariant_distance(torch.randn(10), torch.randn(11))
    with pytest.raises(ValueError, match="batch sizes"):
        complexity_invariant_distance(torch.randn(2, 10, 1), torch.randn(3, 10, 1))
    with pytest.raises(ValueError, match="univariate"):
        complexity_estimate(torch.randn(10, 2))
    with pytest.raises(ValueError, match="complex"):
        complexity_estimate(torch.randn(10, dtype=torch.complex64))


def test_dtype_inferred():
    assert complexity_estimate(torch.randn(10, dtype=torch.float32)).dtype == torch.float32
    assert complexity_estimate(torch.randn(10, dtype=torch.float64)).dtype == torch.float64
    assert complexity_estimate(torch.arange(10)).dtype == torch.get_default_dtype()
    first, second = torch.randn(10, dtype=torch.float32), torch.randn(10, dtype=torch.float64)
    assert complexity_invariant_distance(first, second).dtype == torch.float64
    assert complexity_invariant_distance(first, first).dtype == torch.float32


def test_nan_rows_and_gradient_isolation():
    torch.manual_seed(3)
    batch = torch.randn(4, 30, 1, dtype=torch.float64)
    other = torch.randn(4, 30, 1, dtype=torch.float64)
    batch[1, 5] = float("nan")
    other[2, 0] = float("inf")
    batch.requires_grad_()
    other.requires_grad_()
    distance = complexity_invariant_distance(batch, other)
    assert torch.isnan(distance[1:3]).all() and torch.isfinite(distance[[0, 3]]).all()
    distance[[0, 3]].sum().backward()
    assert not torch.isnan(batch.grad).any() and not torch.isnan(other.grad).any()
    assert (batch.grad[1:3] == 0).all() and (other.grad[1:3] == 0).all()
    assert batch.grad[0].abs().sum() > 0
    estimate = complexity_estimate(batch)
    assert torch.isnan(estimate[1]) and torch.isfinite(estimate[[0, 2, 3]]).all()
    for normalize in (False, True):
        batch.grad = None
        complexity_estimate(batch, normalize=normalize)[[0, 2, 3]].sum().backward()
        assert (batch.grad[1] == 0).all()


def test_gradient_isolation_across_rows():
    torch.manual_seed(4)
    batch = torch.randn(3, 25, 1, dtype=torch.float64, requires_grad=True)
    other = torch.randn(3, 25, 1, dtype=torch.float64)
    complexity_invariant_distance(batch, other)[0].backward()
    assert batch.grad[0].abs().sum() > 0
    assert (batch.grad[1:] == 0).all()


@pytest.mark.parametrize("normalize", [False, True])
def test_gradcheck_complexity_estimate(normalize):
    torch.manual_seed(5)
    batch = torch.randn(2, 12, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: complexity_estimate(value, normalize=normalize), (batch,))


def test_gradcheck_cid():
    torch.manual_seed(6)
    first = torch.randn(2, 12, 1, dtype=torch.float64, requires_grad=True)
    second = (3 * torch.randn(2, 12, 1, dtype=torch.float64)).requires_grad_()  # complexities differ: away from ties
    assert torch.autograd.gradcheck(complexity_invariant_distance, (first, second))


def test_gradient_at_constant_series_is_zero_not_nan():
    constant = torch.ones(8, dtype=torch.float64, requires_grad=True)
    complexity_estimate(constant).backward()
    assert (constant.grad == 0).all()


def test_singular_pair_does_not_contaminate_shared_query_gradient():
    query = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    candidates = torch.tensor([[0.0, 2.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64).reshape(2, 3, 1)
    distance = complexity_invariant_distance(query, candidates)
    assert distance[1] == math.inf
    distance[0].backward(retain_graph=True)
    alone = query.detach().clone().requires_grad_()
    complexity_invariant_distance(alone, candidates[0]).backward()
    assert torch.allclose(query.grad, alone.grad)
    assert torch.allclose(query.grad, _f64([1.0, -4.0, 1.0]))
    query.grad = None
    distance[1].backward()
    assert (query.grad == 0).all()


def test_float16_normalized_small_amplitude():
    series = torch.tensor([0.0, 0.001, 0.0], dtype=torch.float16, requires_grad=True)
    result = complexity_estimate(series, normalize=True)
    result.backward()
    reference = torch.tensor([0.0, 0.001, 0.0], dtype=torch.float64, requires_grad=True)
    expected = complexity_estimate(reference, normalize=True)
    expected.backward()
    assert result.dtype == torch.float16
    assert torch.isclose(result.double(), expected, rtol=1e-2)
    assert torch.isfinite(series.grad).all()
    assert torch.allclose(series.grad.double(), reference.grad, atol=1e-2)
    half_cid = complexity_invariant_distance(series.detach(), series.detach() * 2)
    assert half_cid.dtype == torch.float16 and torch.isfinite(half_cid)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(torch.float32, 1e-25), (torch.float32, 1e20), (torch.float64, 1e-170), (torch.float64, 1e160)],
)
def test_scale_equivariance(dtype, scale):
    base = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    other = torch.tensor([0.0, 2.0, 0.0], dtype=dtype)
    estimate = complexity_estimate(base * scale)
    assert torch.isclose(estimate / scale, torch.tensor(math.sqrt(2), dtype=dtype), rtol=1e-5)
    # ED = 1, ratio 2 -> CID = 2 scale
    distance = complexity_invariant_distance(base * scale, other * scale)
    assert torch.isclose(distance / scale, torch.tensor(2.0, dtype=dtype), rtol=1e-5)
    assert torch.isclose(
        complexity_estimate(base * scale, normalize=True), complexity_estimate(base, normalize=True), rtol=1e-5
    )
    leaf = (base * scale).requires_grad_()
    complexity_estimate(leaf).backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().sum() > 0


def test_short_series_and_singleton_batch_shapes():
    assert complexity_estimate(_f64([0.0, 1.0])) == 1
    assert complexity_estimate(_f64([[0.0], [1.0]])) == 1
    single = torch.zeros(1, 5, 1, dtype=torch.float64)
    assert complexity_estimate(single).shape == (1,)
    assert complexity_invariant_distance(single, single).shape == (1,)
    assert complexity_invariant_distance(single[0], single).shape == (1,)


def test_small_denominator_not_clamped():
    result = complexity_invariant_distance(_f64([0.0, 1e-8, 0.0]), _f64([0.0, 2e-8, 0.0]))
    assert torch.isclose(result, _f64(2e-8))


def test_small_denominator_large_ratio():
    first, second = _f64([0.0, 1e-8, 0.0]), _f64([0.0, 1.0, 0.0])
    expected = (1 - 1e-8) * (math.sqrt(2) / (math.sqrt(2) * 1e-8))
    assert torch.isclose(complexity_invariant_distance(first, second), _f64(expected))


def test_half_precision_distance_computed_in_float32():
    torch.manual_seed(7)
    first, second = torch.randn(2, 40, 1).half(), torch.randn(2, 40, 1).half()
    result = complexity_invariant_distance(first, second)
    expected = complexity_invariant_distance(first.double(), second.double())
    assert result.dtype == torch.float16
    assert torch.allclose(result.double(), expected, rtol=1.5e-3)


def test_mixed_amplitude_shared_query_gradient_float32():
    candidates = torch.tensor([[0.0, 2e-20, 0.0], [0.0, 1.0, 0.0]]).reshape(2, 3, 1)
    query = torch.tensor([0.0, 1e-20, 0.0], requires_grad=True)
    distance = complexity_invariant_distance(query, candidates)
    distance[0].backward(retain_graph=True)
    reference = torch.tensor([0.0, 1e-20, 0.0], dtype=torch.float64, requires_grad=True)
    expected = complexity_invariant_distance(reference, candidates.double())
    expected[0].backward()
    assert torch.isfinite(query.grad).all()
    assert torch.allclose(query.grad.double(), reference.grad, rtol=1e-4)
    assert torch.allclose(distance.double(), expected, rtol=1e-5)


@pytest.mark.parametrize(("dtype", "tiny"), [(torch.float32, 1e-25), (torch.float64, 1e-150)])
def test_mixed_amplitude_pair(dtype, tiny):
    small = torch.tensor([0.0, tiny, 0.0], dtype=dtype, requires_grad=True)
    large = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    distance = complexity_invariant_distance(small, large)
    # ED ~ 1, CF = 1 / tiny
    assert torch.isclose(distance * tiny, torch.tensor(1.0, dtype=dtype), rtol=1e-5)
    if dtype == torch.float64:  # the float32 gradient (about 1e50) is outside the representable range
        distance.backward()
        assert torch.isfinite(small.grad).all()
    assert torch.isclose(
        complexity_invariant_distance(large, small.detach()) * tiny, torch.tensor(1.0, dtype=dtype), rtol=1e-5
    )


@pytest.mark.parametrize(("dtype", "large", "small"), [(torch.float32, 1e10, 1e-30), (torch.float64, 1e150, 1e-300)])
def test_extreme_scale_ratio_does_not_poison_gradient(dtype, large, small):
    candidates = torch.tensor([[0.0, 2 * large, 0.0], [small, small, small]], dtype=dtype).reshape(2, 3, 1)
    query = torch.tensor([0.0, large, 0.0], dtype=dtype, requires_grad=True)
    distance = complexity_invariant_distance(query, candidates)
    assert distance[1] == math.inf
    distance[0].backward()
    # CID is homogeneous of degree 1, so its gradient is scale-free: [1, -4, 1]
    assert torch.allclose(query.grad, torch.tensor([1.0, -4.0, 1.0], dtype=dtype), rtol=1e-4)


@pytest.mark.parametrize("order", [0, 1])
def test_subnormal_against_normal_matches_float64(order):
    tiny = torch.tensor([0.0, 1e-44, 0.0])  # float32 subnormal
    normal = torch.tensor([0.0, 1e-5, 0.0])
    first, second = (tiny, normal) if order == 0 else (normal, tiny)
    result = complexity_invariant_distance(first, second)
    expected = complexity_invariant_distance(first.double(), second.double())
    assert torch.isfinite(result)
    assert torch.isclose(result.double(), expected, rtol=1e-5)
    assert torch.isclose(result, torch.tensor(1.0195e34), rtol=1e-3)


def test_true_overflow_and_underflow_give_inf_and_zero():
    huge = torch.tensor([0.0, 1e38, 0.0])
    tiny = torch.tensor([0.0, 1e-44, 0.0])
    assert complexity_invariant_distance(huge, tiny) == math.inf
    assert complexity_invariant_distance(tiny, tiny.clone()) == 0


@pytest.mark.parametrize("large", [1.0, 1e30])
def test_overflowing_pair_does_not_poison_gradient_of_other_pairs(large):
    query = torch.tensor([0.0, large, 0.0], requires_grad=True)
    candidates = torch.tensor([[0.0, 2 * large, 0.0], [0.0, 1e-44, 0.0]]).reshape(2, 3, 1)
    distance = complexity_invariant_distance(query, candidates)
    assert distance[1] == math.inf
    distance[0].backward()
    assert torch.allclose(query.grad, torch.tensor([1.0, -4.0, 1.0]), rtol=1e-4)
