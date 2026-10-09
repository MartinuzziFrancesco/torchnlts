"""Tests for the ``weighting`` option of torchnlts.entropy.permutation.

Ground truth:
  1. Papers: Fadlallah et al. (2013), Phys. Rev. E 87, 022911 (variance
     weights) and Azami & Escudero (2016), Comput. Methods Programs Biomed.
     128, 40-51 (amplitude-aware weights), hand-derived examples.
  2. Julia: ComplexityMeasures.jl ``WeightedOrdinalPatterns`` and
     ``AmplitudeAwareOrdinalPatterns`` with ``Shannon(base=ℯ)``;
     rtol=atol=1e-9 for float64.
"""

import math
import subprocess

import pytest
import torch

from torchnlts.entropy.permutation import ordinal_pattern_probabilities, permutation_entropy

JULIA_PROJECT = "juliaenv"
TRAJECTORY = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
# Windows (m=3): (4,7,9)->code 0, (7,9,10)->0, (9,10,6)->4, (10,6,11)->2, (6,11,3)->4


def _shannon(probability):
    nonzero = probability[probability > 0]
    return -(nonzero * nonzero.log()).sum()


def test_variance_weighting_hand_derived():
    # Population variance of each window, e.g. (7,9,10): 14/9, (9,10,6): 26/9.
    windows = [(4, 7, 9), (7, 9, 10), (9, 10, 6), (10, 6, 11), (6, 11, 3)]
    weights = []
    for window in windows:
        mean = sum(window) / 3
        weights.append(sum((value - mean) ** 2 for value in window) / 3)
    total = sum(weights)
    expected = torch.zeros(6, dtype=torch.float64)
    for code, weight in zip([0, 0, 4, 2, 4], weights):
        expected[code] += weight / total
    probability = ordinal_pattern_probabilities(TRAJECTORY, 3, 1, weighting="variance")
    assert torch.allclose(probability, expected, rtol=1e-12, atol=0)
    # literal values guard against a wrong formula
    assert math.isclose(weights[1], 14 / 9, rel_tol=1e-12)
    assert math.isclose(weights[2], 26 / 9, rel_tol=1e-12)
    entropy = permutation_entropy(TRAJECTORY, weighting="variance")
    assert torch.isclose(entropy, _shannon(expected))


def test_amplitude_weighting_hand_derived():
    # A = 0.5, m = 3: w = 0.5/3 * sum|x| + 0.5/2 * sum|diff|
    # e.g. (4,7,9): 0.5/3*20 + 0.25*5 = 10/3 + 1.25
    windows = [(4, 7, 9), (7, 9, 10), (9, 10, 6), (10, 6, 11), (6, 11, 3)]
    weights = [0.5 / 3 * sum(abs(v) for v in w) + 0.5 / 2 * (abs(w[1] - w[0]) + abs(w[2] - w[1])) for w in windows]
    assert math.isclose(weights[0], 20 / 6 + 5 / 4, rel_tol=1e-12)
    total = sum(weights)
    expected = torch.zeros(6, dtype=torch.float64)
    for code, weight in zip([0, 0, 4, 2, 4], weights):
        expected[code] += weight / total
    probability = ordinal_pattern_probabilities(TRAJECTORY, 3, 1, weighting="amplitude")
    assert torch.allclose(probability, expected, rtol=1e-12, atol=0)


def test_amplitude_weight_extremes_and_default():
    # A = 1: weight is the mean absolute level only; A default is 0.5.
    level = ordinal_pattern_probabilities(TRAJECTORY, 3, 1, weighting="amplitude", amplitude_weight=1.0)
    weights = torch.tensor([20.0, 26.0, 25.0, 27.0, 20.0], dtype=torch.float64) / 3
    expected = torch.zeros(6, dtype=torch.float64)
    expected.index_add_(0, torch.tensor([0, 0, 4, 2, 4]), weights / weights.sum())
    assert torch.allclose(level, expected, rtol=1e-12, atol=0)
    default = ordinal_pattern_probabilities(TRAJECTORY, 3, 1, weighting="amplitude")
    explicit = ordinal_pattern_probabilities(TRAJECTORY, 3, 1, weighting="amplitude", amplitude_weight=0.5)
    assert torch.equal(default, explicit)


def test_equal_weights_reduce_to_permutation_entropy():
    # Alternating +-1 sawtooth: every window has the same A=1 weight (mean |x| = 1),
    # so amplitude weighting with A=1 equals plain counting.
    trajectory = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    plain = ordinal_pattern_probabilities(trajectory, 3, 1)
    weighted = ordinal_pattern_probabilities(trajectory, 3, 1, weighting="amplitude", amplitude_weight=1.0)
    assert torch.allclose(plain, weighted, rtol=1e-12, atol=0)


def test_normalize_divides_by_log_factorial():
    raw = permutation_entropy(TRAJECTORY, weighting="variance")
    normalized = permutation_entropy(TRAJECTORY, weighting="variance", normalize=True)
    assert torch.isclose(normalized, raw / math.log(6))


def test_batched_shapes_and_unbatched_agreement():
    torch.manual_seed(0)
    batch = torch.randn(3, 30, 1, dtype=torch.float64)
    for weighting in ("variance", "amplitude"):
        probability = ordinal_pattern_probabilities(batch, 3, 1, weighting=weighting)
        assert probability.shape == (3, 6)
        assert torch.allclose(probability.sum(-1), torch.ones(3, dtype=torch.float64))
        entropy = permutation_entropy(batch, weighting=weighting)
        assert entropy.shape == (3,)
        assert torch.isclose(entropy[1], permutation_entropy(batch[1, :, 0], weighting=weighting))
        assert permutation_entropy(batch[1, :, 0], weighting=weighting).shape == ()


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        permutation_entropy(TRAJECTORY, weighting="bogus")
    with pytest.raises(ValueError):
        permutation_entropy(TRAJECTORY, weighting="amplitude", amplitude_weight=1.5)
    with pytest.raises(ValueError):
        permutation_entropy(TRAJECTORY, embedding_dim=1, weighting="amplitude")


def test_nan_and_zero_weight_rows():
    nan_row = torch.tensor([1.0, float("nan"), 2.0, 3.0, 4.0], dtype=torch.float64)
    constant = torch.ones(5, dtype=torch.float64)
    inf_row = torch.tensor([1.0, float("inf"), 2.0, 3.0, 4.0], dtype=torch.float64)
    for weighting in ("variance", "amplitude"):
        assert torch.isnan(permutation_entropy(nan_row, weighting=weighting))
        assert torch.isnan(permutation_entropy(inf_row, weighting=weighting))
    # Zero total weight: constant under variance weighting -> NaN; amplitude of a
    # constant nonzero trajectory with A=1 has positive weight, so is finite.
    assert torch.isnan(permutation_entropy(constant, weighting="variance"))
    assert torch.isfinite(permutation_entropy(constant, weighting="amplitude", amplitude_weight=1.0))
    assert torch.isnan(permutation_entropy(torch.zeros(5, dtype=torch.float64), weighting="amplitude"))


@pytest.mark.parametrize("weighting", ["variance", "amplitude"])
def test_gradcheck(weighting):
    torch.manual_seed(1)
    trajectory = (torch.randn(2, 12, 1, dtype=torch.float64) + 0.1).requires_grad_()
    assert torch.autograd.gradcheck(lambda value: permutation_entropy(value, weighting=weighting), (trajectory,))
    assert torch.autograd.gradcheck(
        lambda value: ordinal_pattern_probabilities(value, 3, 2, weighting=weighting), (trajectory,)
    )


@pytest.mark.parametrize("weighting", ["variance", "amplitude"])
def test_degenerate_row_does_not_poison_batch_gradients(weighting):
    # Regression: a NaN/constant/inf row next to a normal row must leave the
    # normal row's gradient finite and equal to its standalone gradient.
    torch.manual_seed(2)
    normal = torch.randn(12, dtype=torch.float64)
    standalone = normal.clone().requires_grad_()
    permutation_entropy(standalone, weighting=weighting).backward()
    degenerate_rows = [
        torch.full((12,), float("nan"), dtype=torch.float64),
        torch.zeros(12, dtype=torch.float64),  # zero total weight under both weightings
        torch.cat([torch.tensor([float("inf")], dtype=torch.float64), torch.randn(11, dtype=torch.float64)]),
    ]
    for degenerate in degenerate_rows:
        batch = torch.stack([degenerate, normal]).unsqueeze(-1).requires_grad_()
        entropy = permutation_entropy(batch, weighting=weighting)
        assert torch.isnan(entropy[0]) and torch.isfinite(entropy[1])
        entropy.nansum().backward()
        assert torch.isfinite(batch.grad).all()
        assert torch.allclose(batch.grad[1, :, 0], standalone.grad)
        assert (batch.grad[0] == 0).all()


@pytest.mark.parametrize("weighting", ["variance", "amplitude"])
def test_half_precision_matches_float32_then_cast(weighting):
    # Regression: entropy is computed in float32 and cast once at the end.
    trajectory = torch.randn(300).to(torch.float16)
    result = permutation_entropy(trajectory, weighting=weighting, normalize=True)
    expected = permutation_entropy(trajectory.float(), weighting=weighting, normalize=True).to(torch.float16)
    assert result.dtype == torch.float16 and torch.equal(result, expected)


@pytest.mark.parametrize("weighting", ["variance", "amplitude"])
@pytest.mark.parametrize("factor", [1e20, 1e-24])
def test_scale_invariance_float32(weighting, factor):
    # Regression: squaring unscaled coordinates overflowed/underflowed to NaN.
    torch.manual_seed(8)
    trajectory = torch.randn(200)
    reference = permutation_entropy(trajectory, weighting=weighting)
    scaled = permutation_entropy(trajectory * factor, weighting=weighting)
    assert torch.isfinite(scaled) and torch.isclose(scaled, reference, rtol=1e-5)


def test_variance_exact_shift_invariance_and_nonbinary_constant():
    # Regression: mean rounding made a shifted series differ and a constant
    # 0.1 series give -0.0 instead of NaN.
    torch.manual_seed(9)
    trajectory = torch.randn(200, dtype=torch.float64)
    shifted = trajectory + 2.0**24
    assert torch.equal(torch.argsort(trajectory), torch.argsort(shifted))
    assert torch.isclose(
        permutation_entropy(trajectory, weighting="variance"),
        permutation_entropy(shifted, weighting="variance"),
        rtol=1e-6,
    )
    assert torch.isnan(permutation_entropy(torch.full((20,), 0.1, dtype=torch.float64), weighting="variance"))


def test_amplitude_ties_go_to_one_pattern():
    # Regression: documents the tie convention (earliest index, one pattern).
    # Azami & Escudero split a tied window's weight among compatible
    # permutations ([1, 4, 4] -> entropy log 2); here it is a single pattern -> 0.
    trajectory = torch.tensor([1.0, 4.0, 4.0], dtype=torch.float64)
    assert permutation_entropy(trajectory, weighting="amplitude") == 0


@pytest.mark.parametrize("bad", [None, "0.5", 1j, torch.tensor([0.2, 0.3]), -0.1, 1.1, float("nan")])
def test_invalid_amplitude_weight_raises_value_error(bad):
    with pytest.raises(ValueError):
        permutation_entropy(TRAJECTORY, weighting="amplitude", amplitude_weight=bad)


def _run_julia(trajectory, constructor, embedding_dim, delay, extra=""):
    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = {constructor}{{{embedding_dim}}}({delay}{extra})
    println(entropy(Shannon(base=MathConstants.e), est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return torch.tensor(float(result.stdout.strip()), dtype=torch.float64)


@pytest.mark.parametrize("embedding_dim,delay", [(3, 1), (4, 2)])
def test_weighted_matches_complexitymeasures_jl(embedding_dim, delay):
    # Julia computes the population variance of each embedded vector, as in
    # this implementation (it documents a typo in the paper's index range);
    # no deviation.
    torch.manual_seed(5)
    trajectory = torch.randn(60, dtype=torch.float64)
    julia = _run_julia(trajectory, "WeightedOrdinalPatterns", embedding_dim, delay)
    ours = permutation_entropy(trajectory, embedding_dim, delay, weighting="variance")
    assert torch.isclose(ours, julia, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("amplitude_weight", [0.0, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("embedding_dim,delay", [(3, 1), (4, 2)])
def test_amplitude_matches_complexitymeasures_jl(amplitude_weight, embedding_dim, delay):
    torch.manual_seed(6)
    trajectory = torch.randn(60, dtype=torch.float64)
    julia = _run_julia(trajectory, "AmplitudeAwareOrdinalPatterns", embedding_dim, delay, f", {amplitude_weight}")
    ours = permutation_entropy(
        trajectory, embedding_dim, delay, weighting="amplitude", amplitude_weight=amplitude_weight
    )
    assert torch.isclose(ours, julia, rtol=1e-9, atol=1e-9)


def test_unweighted_large_integers_ranked_exactly():
    # Regression: values above 2**24 are distinct as integers but tie in float32.
    series = torch.tensor([2**24 + 1, 2**24, 2**24 + 2, 2**24 + 1, 2**24])
    expected = ordinal_pattern_probabilities(torch.tensor([1, 0, 2, 1, 0]))
    assert torch.equal(ordinal_pattern_probabilities(series), expected)
