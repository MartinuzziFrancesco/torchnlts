"""Tests for torchnlts.entropy.permutation.

Dual ground truth:
  1. Paper: Bandt & Pompe (2002), Sec. II-III -- hand-derived examples
     (untied and tied multi-window cases, each independently
     cross-checked against ComplexityMeasures.jl in its own test) plus a
     bijection check on the Lehmer-code pattern encoding.
  2. Julia: ComplexityMeasures.jl's ``OrdinalPatterns`` + ``Shannon``.
     Float64 comparisons use rtol=atol=1e-9 to allow rounding differences
     in logarithms and summation while detecting numerical disagreements.
"""

import itertools
import math
import subprocess

import torch

from torchnlts.entropy.permutation import _ordinal_pattern_codes, ordinal_pattern_probabilities, permutation_entropy

JULIA_PROJECT = "juliaenv"


def test_permutation_entropy_hand_derived():
    # Order 3, delay 1; ordinal patterns (0-indexed, ties by earliest index):
    #   (4,7,9)   -> (0,1,2)
    #   (7,9,10)  -> (0,1,2)
    #   (9,10,6)  -> (2,0,1)
    #   (10,6,11) -> (1,0,2)
    #   (6,11,3)  -> (2,0,1)
    # Counts: (0,1,2)=2, (2,0,1)=2, (1,0,2)=1 out of 5 -> Shannon entropy
    # -(2/5 ln(2/5) * 2 + 1/5 ln(1/5)) nats.
    trajectory = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
    result = permutation_entropy(trajectory, embedding_dim=3, delay=1)
    expected = -(2 * (2 / 5) * math.log(2 / 5) + (1 / 5) * math.log(1 / 5))
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_permutation_entropy_normalized():
    trajectory = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
    result = permutation_entropy(trajectory, embedding_dim=3, delay=1, normalize=True)
    raw = permutation_entropy(trajectory, embedding_dim=3, delay=1)
    assert torch.isclose(result, raw / math.log(6))


def test_ordinal_pattern_probabilities_matches_hand_derived_counts():
    # Same trajectory: Lehmer codes 0, 4, 1 with counts 2, 2, 1; assert the full vector.
    trajectory = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
    probability = ordinal_pattern_probabilities(trajectory, embedding_dim=3, delay=1)
    expected = torch.tensor([0.4, 0.0, 0.2, 0.0, 0.4, 0.0], dtype=torch.float64)
    assert torch.equal(probability, expected)


def test_permutation_entropy_matches_shannon_of_ordinal_pattern_probabilities():
    # Shannon entropy of ordinal_pattern_probabilities; torch.equal since both sides run the same computation.
    # Hand-derived trajectory chosen for its zero-probability outcomes.
    trajectory = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
    probability = ordinal_pattern_probabilities(trajectory, embedding_dim=3, delay=1)
    nonzero = probability > 0
    safe = torch.where(nonzero, probability, torch.ones_like(probability))
    expected = -(probability * torch.log(safe)).sum(dim=-1)
    assert torch.equal(permutation_entropy(trajectory, embedding_dim=3, delay=1), expected)


def test_ordinal_pattern_codes_ties_broken_by_index():
    # (5, 5, 3): tied 5s keep their order, giving (2, 0, 1) = Lehmer code 4 (digits (2, 0, 0), weights (2, 1, 1)).
    # Matches ComplexityMeasures.jl's tie convention (outcome [3, 1, 2] 1-indexed).
    codes = _ordinal_pattern_codes(torch.tensor([[5.0, 5.0, 3.0]], dtype=torch.float64))
    assert torch.equal(codes, torch.tensor([4]))


def test_permutation_entropy_ties_broken_by_index_multi_window():
    # Windows (order 3, delay 1) over [1,1,2,3,3,1]:
    #   (1,1,2) -> (0,1,2)   (1,2,3) -> (0,1,2)
    #   (2,3,3) -> (0,1,2)   (3,3,1) -> (2,0,1)
    # -> 3/4, 1/4 -> Shannon entropy -(3/4 ln(3/4) + 1/4 ln(1/4)) nats.
    trajectory = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 1.0], dtype=torch.float64)
    result = permutation_entropy(trajectory, embedding_dim=3, delay=1)
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_ordinal_pattern_codes_bijection():
    order = 4
    permutations = list(itertools.permutations(range(order)))
    codes = _ordinal_pattern_codes(torch.tensor(permutations, dtype=torch.float64))
    assert sorted(codes.tolist()) == list(range(math.factorial(order)))


def test_permutation_entropy_batched_matches_unbatched():
    torch.manual_seed(0)
    trajectory = torch.randn(3, 30, 1, dtype=torch.float64)
    batched = permutation_entropy(trajectory, embedding_dim=3)
    assert batched.shape == (3,)
    for i in range(3):
        expected = permutation_entropy(trajectory[i], embedding_dim=3)
        assert torch.isclose(batched[i], expected)


def test_permutation_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 20, 1, dtype=torch.float64)
    batched = permutation_entropy(trajectory, embedding_dim=3)
    unbatched = permutation_entropy(trajectory[0], embedding_dim=3)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_permutation_entropy_rejects_multivariate():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    try:
        permutation_entropy(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_permutation_entropy_nan_input_gives_nan_output():
    # NaN coordinate: argsort would rank it silently. Not raised: a data-dependent `if` breaks torch.vmap.
    trajectory = torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64)
    assert torch.isnan(permutation_entropy(trajectory, embedding_dim=3))


def test_permutation_entropy_vmap_compatible():
    torch.manual_seed(7)
    trajectory = torch.randn(2, 20, 1, dtype=torch.float64)
    direct = permutation_entropy(trajectory, embedding_dim=3)
    vmapped = torch.vmap(lambda t: permutation_entropy(t, embedding_dim=3))(trajectory)
    assert torch.allclose(direct, vmapped)


def test_permutation_entropy_too_short_raises():
    trajectory = torch.tensor([1.0, 2.0], dtype=torch.float64)
    try:
        permutation_entropy(trajectory, embedding_dim=3, delay=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_permutation_entropy_embedding_dim_one():
    # order=1: one possible pattern, raw entropy 0, normalized 0/0 = NaN (log(1!) = 0).
    trajectory = torch.tensor([1.0, 5.0, -3.0, 2.0], dtype=torch.float64)
    raw = permutation_entropy(trajectory, embedding_dim=1)
    normalized = permutation_entropy(trajectory, embedding_dim=1, normalize=True)
    assert torch.isclose(raw, torch.tensor(0.0, dtype=torch.float64))
    assert torch.isnan(normalized)


def test_permutation_entropy_accepts_trailing_singleton_feature_dim():
    torch.manual_seed(2)
    trajectory = torch.randn(20, dtype=torch.float64)
    unbatched = permutation_entropy(trajectory, embedding_dim=3)
    with_feature_dim = permutation_entropy(trajectory.unsqueeze(-1), embedding_dim=3)
    assert torch.isclose(unbatched, with_feature_dim)


def _run_julia_ordinal_entropy(trajectory: torch.Tensor, embedding_dim: int, delay: int) -> torch.Tensor:
    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = OrdinalPatterns(m={embedding_dim}, τ={delay})
    println(entropy(Shannon(base=MathConstants.e), est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return torch.tensor(float(result.stdout.strip()), dtype=torch.float64)


def test_permutation_entropy_matches_complexitymeasures_jl():
    torch.manual_seed(3)
    trajectory = torch.randn(40, dtype=torch.float64)
    embedding_dim = 3
    delay = 1
    julia_entropy = _run_julia_ordinal_entropy(trajectory, embedding_dim, delay)
    torch_entropy = permutation_entropy(trajectory, embedding_dim=embedding_dim, delay=delay)
    assert torch.isclose(torch_entropy, julia_entropy, rtol=1e-9, atol=1e-9)


def test_permutation_entropy_matches_complexitymeasures_jl_hand_derived():
    trajectory = torch.tensor([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0], dtype=torch.float64)
    julia_entropy = _run_julia_ordinal_entropy(trajectory, 3, 1)
    expected = -(2 * (2 / 5) * math.log(2 / 5) + (1 / 5) * math.log(1 / 5))
    assert torch.isclose(julia_entropy, torch.tensor(expected, dtype=torch.float64), rtol=1e-9, atol=1e-9)
    assert torch.isclose(permutation_entropy(trajectory), julia_entropy, rtol=1e-9, atol=1e-9)


def test_permutation_entropy_matches_complexitymeasures_jl_nonunit_delay_and_order():
    torch.manual_seed(4)
    trajectory = torch.randn(50, dtype=torch.float64)
    embedding_dim = 4
    delay = 3
    julia_entropy = _run_julia_ordinal_entropy(trajectory, embedding_dim, delay)
    torch_entropy = permutation_entropy(trajectory, embedding_dim=embedding_dim, delay=delay)
    assert torch.isclose(torch_entropy, julia_entropy, rtol=1e-9, atol=1e-9)


def test_permutation_entropy_matches_complexitymeasures_jl_with_ties():
    trajectory = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 1.0], dtype=torch.float64)
    embedding_dim, delay = 3, 1
    julia_entropy = _run_julia_ordinal_entropy(trajectory, embedding_dim, delay)
    torch_entropy = permutation_entropy(trajectory, embedding_dim=embedding_dim, delay=delay)
    assert torch.isclose(torch_entropy, julia_entropy, rtol=1e-9, atol=1e-9)


def test_permutation_entropy_nan_for_inf_rows():
    # Regression: an inf point used to be ranked like an ordinary value.
    batch = torch.randn(2, 30, 1, dtype=torch.float64)
    batch[1, 3] = float("inf")
    values = permutation_entropy(batch)
    assert torch.isnan(values[1]) and torch.isfinite(values[0])
