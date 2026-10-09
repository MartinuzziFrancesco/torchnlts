"""Tests for torchnlts.entropy.sample.

Dual ground truth:
  1. Paper: Richman & Moorman (2000), eq. 1-3 -- hand-derived examples
     for both delay=1 and delay=2 (the trickiest correctness point: the
     m/(m+1) embedding index alignment).
  2. Julia: ComplexityMeasures.jl's `SampleEntropy`. Its B(r) term has
     a documented boundary quirk (see the dual-validation test below)
     that this test neutralizes rather than replicates, since the
     quirk is a deviation from the cited paper, not a float-precision
     difference.
"""

import math
import subprocess

import torch

from torchnlts.entropy.sample import _matched_length_templates, sample_entropy, soft_sample_entropy

JULIA_PROJECT = "juliaenv"


def test_sample_entropy_hand_derived_delay_1():
    # Templates [0,1,0] (m) and [[0,1],[1,0],[0,2]] (m+1), radius 1.5: B=6/6=1; A=4/6 (only (w1,w2) is out of range).
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    result = sample_entropy(trajectory, radius=1.5, embedding_dim=1, delay=1)
    assert torch.isclose(result, torch.tensor(-math.log(2 / 3), dtype=torch.float64))


def test_sample_entropy_hand_derived_delay_2():
    # Anomaly at index 6 pollutes only the m+1 embedding: B=10/12 (5/6 unordered pairs match);
    # A=4/12 since w2=(2,4,100) matches nothing and only (0,1), (1,3) match.
    trajectory = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0], dtype=torch.float64)
    result = sample_entropy(trajectory, radius=2.5, embedding_dim=2, delay=2)
    expected = -math.log((4 / 12) / (10 / 12))
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_matched_length_templates_delay_3_coordinates():
    trajectory = torch.arange(10, dtype=torch.float64)
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim=2, delay=3)
    assert long_embed.shape == (4, 3)
    assert torch.equal(long_embed, torch.tensor([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]))
    # short_embed is truncated to the first 4 rows (7 - delay).
    assert short_embed.shape == (4, 2)
    assert torch.equal(short_embed, torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))


def test_sample_entropy_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 30, 2, dtype=torch.float64)
    radii = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    swept = sample_entropy(trajectory, radius=radii, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, radius in enumerate(radii):
        expected = sample_entropy(trajectory, radius=radius, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected, equal_nan=True)


def test_sample_entropy_zero_b_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    result = sample_entropy(trajectory, radius=0.1, embedding_dim=1, delay=1)
    assert torch.isnan(result)


def test_sample_entropy_zero_a_positive_b_is_inf_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    result = sample_entropy(trajectory, radius=1.5, embedding_dim=1, delay=1)
    assert torch.isinf(result) and result > 0


def test_sample_entropy_negative_theiler_window_raises():
    # Self-matches must always be excluded (Richman & Moorman), even with the no-exclusion convention.
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    try:
        sample_entropy(trajectory, radius=1.5, embedding_dim=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_soft_sample_entropy_negative_theiler_window_raises():
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    try:
        soft_sample_entropy(trajectory, temperature=0.5, embedding_dim=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sample_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(5)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = sample_entropy(trajectory, radius=1.0, embedding_dim=2)
    unbatched = sample_entropy(trajectory[0], radius=1.0, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_soft_sample_entropy_underflow_is_nan_not_crash():
    # Tiny temperature: A and B kernel weights underflow to 0, giving 0/0.
    trajectory = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    result = soft_sample_entropy(trajectory, temperature=0.01, embedding_dim=1)
    assert torch.isnan(result)


def test_sample_entropy_unpadded_disagrees_with_complexitymeasures_jl():
    # Locks in the boundary deviation from ComplexityMeasures.jl, whose B(r) leaks one out-of-range neighbor.
    # If a future ComplexityMeasures.jl fixes this, the test fails and the docs need revisiting.
    trajectory = [0.0, 1.0, 0.0, 2.0]
    julia_code = """
    using ComplexityMeasures
    x = [0.0, 1.0, 0.0, 2.0]
    est = SampleEntropy(x; r=1.5, m=1, τ=1)
    println(complexity(est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_entropy = sample_entropy(torch.tensor(trajectory, dtype=torch.float64), radius=1.5, embedding_dim=1, delay=1)
    # Julia's leaked neighbor gives A=4, B=7 -> ln(7/4); strict truncation here gives A=4, B=6 -> ln(3/2).
    assert torch.isclose(julia_entropy, torch.tensor(math.log(7 / 4), dtype=torch.float64))
    assert torch.isclose(torch_entropy, torch.tensor(math.log(3 / 2), dtype=torch.float64))


def test_soft_sample_entropy_hand_derived():
    # Truncated short_embed [0, 1]: B = exp(-0.5); long_embed [(0,1),(1,3)]: A = exp(-2); SampEn = -ln(A/B) = 1.5.
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    result = soft_sample_entropy(trajectory, temperature=1.0, embedding_dim=1, delay=1)
    assert torch.isclose(result, torch.tensor(1.5, dtype=torch.float64))


def test_soft_sample_entropy_gradcheck():
    torch.manual_seed(1)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_sample_entropy(x, temperature=0.5, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_sample_entropy_gradcheck_exponential():
    torch.manual_seed(2)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return soft_sample_entropy(x, temperature=0.5, embedding_dim=2, kind="exponential")

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_soft_sample_entropy_gradcheck_temperature():
    torch.manual_seed(3)
    trajectory = torch.randn(15, dtype=torch.float64)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_sample_entropy(trajectory, temperature=t, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (temperature,))


def test_sample_entropy_matches_complexitymeasures_jl():
    # ComplexityMeasures.jl builds B(r)'s KD-tree from the untruncated embedding but queries only the truncated points,
    # leaking one extra neighbor for delay=1. Far-outlier padding keeps it out of range so B matches the paper.
    torch.manual_seed(4)
    trajectory = torch.cat([torch.randn(28, dtype=torch.float64), torch.tensor([1e6, 2e6], dtype=torch.float64)])
    radius = 1.0
    embedding_dim = 2

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = SampleEntropy(m={embedding_dim}, τ=1, r={radius})
    println(complexity(est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_entropy = sample_entropy(trajectory, radius=radius, embedding_dim=embedding_dim, delay=1)
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-9)


def test_soft_sample_entropy_shared_input_backward_is_finite():
    # Regression: a row whose kernel sums underflow to 0 (value NaN) must not
    # put NaN into gradients of an input shared with other batch rows.
    torch.manual_seed(1)
    base = torch.randn(30, dtype=torch.float64, requires_grad=True)
    batch = torch.stack([base, base + 100 * torch.arange(30)]).unsqueeze(-1)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    entropy = soft_sample_entropy(batch, temperature)
    assert torch.isnan(entropy[1]) and torch.isfinite(entropy[0])
    grad_base, grad_temperature = torch.autograd.grad(entropy[0], (base, temperature))
    assert torch.isfinite(grad_base).all() and torch.isfinite(grad_temperature)


def test_sample_entropies_nan_for_nonfinite_rows():
    # Regression: a NaN/inf point used to be skipped, giving a finite entropy for that row.
    torch.manual_seed(0)
    for bad in (float("nan"), float("inf")):
        batch = torch.randn(3, 60, 1, dtype=torch.float64)
        batch[1, 5] = bad
        for function in (sample_entropy, soft_sample_entropy):
            values = function(batch, 0.5)
            assert torch.isnan(values[1]) and torch.isfinite(values[[0, 2]]).all()
