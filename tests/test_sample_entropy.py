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

import pytest
import torch

from tests.conftest import assert_first_row_gradients_unaffected, run_julia
from torchnlts.entropy.sample import _matched_length_templates, sample_entropy, soft_sample_entropy

NONFINITE = [float("nan"), float("inf"), -float("inf")]
# With embedding_dim=1 and delay=30 on 50 samples, samples 20-29 enter no template.
GAP = dict(embedding_dim=1, delay=30)


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
    with pytest.raises(ValueError):
        sample_entropy(trajectory, radius=1.5, embedding_dim=1, theiler_window=-1)


def test_soft_sample_entropy_negative_theiler_window_raises():
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        soft_sample_entropy(trajectory, temperature=0.5, embedding_dim=1, theiler_window=-1)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("function, parameter", [(sample_entropy, 1.0), (soft_sample_entropy, 0.05)])
def test_sample_entropies_half_input_is_computed_in_float32(function, parameter, dtype):
    torch.manual_seed(1)
    series = torch.randn(2, 300, 1).to(dtype)
    result = function(series, parameter)
    assert result.dtype == dtype and torch.isfinite(result).all()
    assert torch.equal(result, function(series.float(), parameter).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_soft_sample_entropy_half_kernel_sums_do_not_underflow(dtype):
    result = soft_sample_entropy(torch.arange(12, dtype=dtype), 0.15)
    assert result.dtype == dtype and result == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_soft_sample_entropy_half_temperature_is_computed_in_float32(dtype):
    torch.manual_seed(2)
    series = torch.randn(2, 300, 1).to(dtype)
    temperature = torch.tensor([0.05, 0.1], dtype=dtype)
    result = soft_sample_entropy(series, temperature)
    assert result.dtype == dtype and torch.isfinite(result).all()
    assert torch.equal(result, soft_sample_entropy(series.float(), temperature.float()).to(dtype))


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
    stdout = run_julia(julia_code)
    julia_entropy = torch.tensor(float(stdout.strip()), dtype=torch.float64)
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
    stdout = run_julia(julia_code)
    julia_entropy = torch.tensor(float(stdout.strip()), dtype=torch.float64)
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


@pytest.mark.parametrize("function", [sample_entropy, soft_sample_entropy])
@pytest.mark.parametrize("bad", NONFINITE)
def test_sample_entropies_nan_for_nonfinite_sample_outside_every_template(function, bad):
    # Regression: a non-finite sample that enters no template used to leave its row finite.
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


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_soft_sample_entropy_nonfinite_row_leaves_other_gradients(bad):
    torch.manual_seed(1)
    clean, bad_row = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    bad_row[25] = bad

    def gradients(rows):
        batch = torch.stack(rows).unsqueeze(-1).requires_grad_()
        temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        soft_sample_entropy(batch, temperature, **GAP).nansum().backward()
        return batch.grad, temperature.grad

    batch_grad, temperature_grad = gradients([bad_row, clean])
    single_grad, single_temperature_grad = gradients([clean])
    assert torch.isfinite(batch_grad).all() and (batch_grad[0] == 0).all()
    assert torch.allclose(batch_grad[1], single_grad[0])
    assert torch.isfinite(temperature_grad) and torch.allclose(temperature_grad, single_temperature_grad)


@pytest.mark.parametrize("function", [sample_entropy, soft_sample_entropy])
def test_sample_entropies_vmap_with_nonfinite_sample_outside_every_template(function):
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = float("nan")
    direct = function(batch, 0.5, **GAP)
    mapped = torch.vmap(lambda series: function(series, 0.5, **GAP))(batch)
    assert torch.allclose(direct, mapped, equal_nan=True) and torch.isnan(mapped[1]) and torch.isfinite(mapped[0])


def test_soft_sample_entropy_denormal_kernel_sums_keep_other_gradients_finite():
    # A kernel sum of about 2e-42 (denormal in float32) overflows a quotient's gradient, which would reach the shared
    # temperature as 0 * inf. The second row is an arithmetic progression whose nearest templates are 13.74 bandwidths
    # apart, so its A and B are tiny but not zero.
    small = [0.00, 0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.06, -0.01, 0.00, 0.04, -0.03]
    progression = [1.374 * index for index in range(12)]
    result = assert_first_row_gradients_unaffected(
        lambda series, temperature: soft_sample_entropy(series, temperature), [small, progression], 0.1, torch.float32
    )
    assert torch.isfinite(result).all() and result[1].abs() < 1e-3
