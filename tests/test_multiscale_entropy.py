"""Tests for torchnlts.entropy.multiscale.

Dual ground truth:
  1. Paper: Costa, Goldberger & Peng (2002), the (unnumbered)
     coarse-graining formula on p. 2 (their own eq. 1 is Shannon
     entropy) -- hand-derived coarse-graining checks, plus the identity
     that scale=1 must recover plain sample_entropy exactly (the
     composition's own correctness, not a re-test of sample_entropy
     itself).
  2. Julia: ComplexityMeasures.jl's `multiscale(RegularDownsampling,
     SampleEntropy, x)`. Its SampleEntropy has the same documented
     boundary quirk as tests/test_sample_entropy.py's own dual
     validation (a leaked out-of-range B(r) neighbor candidate for
     delay=1); neutralized here the same way, at every scale tested,
     since coarse-graining still leaves the padding values as
     far outliers relative to any radius used.
"""

import subprocess

import torch

from torchnlts.entropy.multiscale import (
    coarse_grain,
    composite_multiscale_entropy,
    multiscale_entropy,
    soft_composite_multiscale_entropy,
    soft_multiscale_entropy,
)
from torchnlts.entropy.sample import sample_entropy, soft_sample_entropy

JULIA_PROJECT = "juliaenv"


def test_coarse_grain_hand_derived():
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    expected_scale_2 = torch.tensor([1.5, 3.5, 5.5], dtype=torch.float64)
    expected_scale_3 = torch.tensor([2.0, 5.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 2), expected_scale_2)
    assert torch.allclose(coarse_grain(trajectory, 3), expected_scale_3)


def test_coarse_grain_scale_1_is_identity():
    torch.manual_seed(0)
    trajectory = torch.randn(20, dtype=torch.float64)
    assert torch.equal(coarse_grain(trajectory, 1), trajectory)


def test_coarse_grain_drops_trailing_remainder():
    # 7 points at scale=3: floor(7/3)=2 windows, last point dropped.
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 3), torch.tensor([2.0, 5.0], dtype=torch.float64))


def test_coarse_grain_preserves_shape_convention():
    torch.manual_seed(1)
    scalar = torch.randn(30, dtype=torch.float64)
    multivariate = torch.randn(30, 4, dtype=torch.float64)
    batched = torch.randn(2, 30, 4, dtype=torch.float64)
    assert coarse_grain(scalar, 5).shape == (6,)
    assert coarse_grain(multivariate, 5).shape == (6, 4)
    assert coarse_grain(batched, 5).shape == (2, 6, 4)


def test_coarse_grain_rejects_non_positive_scale():
    trajectory = torch.randn(10, dtype=torch.float64)
    for bad_scale in (0, -1):
        try:
            coarse_grain(trajectory, bad_scale)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_multiscale_entropy_scale_1_matches_sample_entropy():
    torch.manual_seed(2)
    trajectory = torch.randn(100, dtype=torch.float64)
    radius = 0.15 * trajectory.std()
    curve = multiscale_entropy(trajectory, scales=[1, 5], radius=radius, embedding_dim=2)
    assert torch.isclose(curve[0], sample_entropy(trajectory, radius, embedding_dim=2))


def test_multiscale_entropy_shape_batched_and_swept():
    torch.manual_seed(3)
    trajectory = torch.randn(2, 100, 1, dtype=torch.float64)
    radius = torch.tensor([0.3, 0.6], dtype=torch.float64)
    curve = multiscale_entropy(trajectory, scales=[1, 2, 3], radius=radius, embedding_dim=2)
    assert curve.shape == (3, 2, 2)
    for scale_idx, scale in enumerate((1, 2, 3)):
        expected = sample_entropy(coarse_grain(trajectory, scale), radius, embedding_dim=2)
        assert torch.allclose(curve[scale_idx], expected, equal_nan=True)


def test_soft_multiscale_entropy_gradcheck():
    torch.manual_seed(4)
    trajectory = torch.randn(20, dtype=torch.float64)
    temperature = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_multiscale_entropy(trajectory, scales=[1, 2], temperature=t, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (temperature,))


def test_soft_multiscale_entropy_gradcheck_trajectory():
    # The temperature-only gradcheck above
    # never differentiates through coarse_grain's reshape/mean (the one
    # genuinely new piece of logic this module adds), since it holds
    # the trajectory fixed. This exercises that path directly.
    torch.manual_seed(6)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_multiscale_entropy(t, scales=[1, 2, 3], temperature=0.4, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_multiscale_entropy_too_coarse_scale_raises():
    # A scale coarse enough to leave the
    # coarse-grained series shorter than embedding_dim+1 samples raises
    # (propagated from delay_embedding via sample_entropy) rather than
    # degrading that entry to NaN like a merely-degenerate-but-still-
    # embeddable scale does -- confirmed boundary.
    torch.manual_seed(7)
    trajectory = torch.randn(20, dtype=torch.float64)
    embedding_dim = 2

    # scale=6 -> coarse length 3 == embedding_dim+1: still embeddable (a
    # purely length-based guarantee, regardless of the random values).
    result = multiscale_entropy(trajectory, scales=[6], radius=1.0, embedding_dim=embedding_dim)
    assert result.shape == (1,)

    # scale=7 -> coarse length 2 < embedding_dim+1: too short, must raise.
    try:
        multiscale_entropy(trajectory, scales=[7], radius=1.0, embedding_dim=embedding_dim)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_multiscale_entropy_matches_complexitymeasures_jl():
    # Same ComplexityMeasures.jl SampleEntropy boundary-handling
    # deviation as test_sample_entropy.py's own dual-validation test
    # (its B(r) leaks one out-of-range neighbor candidate for delay=1,
    # confirmed there by reading the package's source directly).
    # multiscale_entropy calls sample_entropy at each coarse-grained
    # scale, so it needs the same neutralization -- far-outlier padding
    # that stays far even after being averaged into a coarse window at
    # every scale swept here.
    torch.manual_seed(5)
    trajectory = torch.cat([torch.randn(40, dtype=torch.float64), torch.tensor([1e6, 2e6], dtype=torch.float64)])
    radius = 1.0
    embedding_dim = 2
    scales = [1, 2, 3]

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = SampleEntropy(m={embedding_dim}, τ=1, r={radius})
    ds = RegularDownsampling(scales={scales[0]}:{scales[-1]})
    println(join(multiscale(ds, est, x), ","))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_curve = torch.tensor([float(v) for v in result.stdout.strip().split(",")], dtype=torch.float64)
    torch_curve = multiscale_entropy(trajectory, scales=scales, radius=radius, embedding_dim=embedding_dim, delay=1)
    assert torch.allclose(torch_curve, julia_curve, atol=1e-9)


# Composite multiscale entropy: Wu, Wu, Lin, Wang & Lee (2013), Entropy 15(3),
# 1069-1084. Julia: ComplexityMeasures.jl `multiscale(CompositeDownsampling, SampleEntropy)`.


def test_coarse_grain_offset_hand_derived():
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 3, offset=0), torch.tensor([2.0, 5.0], dtype=torch.float64))
    assert torch.allclose(coarse_grain(trajectory, 3, offset=1), torch.tensor([3.0, 6.0], dtype=torch.float64))
    assert torch.allclose(coarse_grain(trajectory, 3, offset=2), torch.tensor([4.0], dtype=torch.float64))
    assert torch.equal(coarse_grain(trajectory, 3), coarse_grain(trajectory, 3, offset=0))


def test_coarse_grain_rejects_bad_offset():
    trajectory = torch.randn(10, dtype=torch.float64)
    for bad_offset in (-1, 3):
        try:
            coarse_grain(trajectory, 3, offset=bad_offset)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_composite_multiscale_entropy_hand_derived():
    # Period-3 signal at scale 2: both offset series are period-3 too
    # ([1,2,3,...] and [3,1,2,...]), every matching 2-template extends to a
    # matching 3-template, so A = B and each SampEn = -log(1) = 0.
    trajectory = torch.tensor([0.0, 2.0, 4.0] * 8, dtype=torch.float64)[:23]
    assert torch.equal(coarse_grain(trajectory, 2, 0)[:3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.equal(coarse_grain(trajectory, 2, 1)[:3], torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(composite_multiscale_entropy(trajectory, scales=[2], radius=0.5), torch.zeros(1, dtype=torch.float64))
    # Perturbing one sample moves the two offsets by different amounts; the
    # composite value is exactly their mean.
    perturbed = trajectory.clone()
    perturbed[10] += 1.7
    perturbed[3] -= 1.2
    per_offset = [sample_entropy(coarse_grain(perturbed, 2, offset), 0.5) for offset in range(2)]
    assert per_offset[0] != per_offset[1] and all(torch.isfinite(value) for value in per_offset)
    assert torch.isclose(composite_multiscale_entropy(perturbed, scales=[2], radius=0.5)[0], sum(per_offset) / 2)


def test_composite_multiscale_entropy_scale_1_matches_sample_entropy():
    torch.manual_seed(2)
    trajectory = torch.randn(100, dtype=torch.float64)
    radius = 0.15 * trajectory.std()
    curve = composite_multiscale_entropy(trajectory, scales=[1, 4], radius=radius)
    assert torch.isclose(curve[0], sample_entropy(trajectory, radius))
    expected = torch.stack([sample_entropy(coarse_grain(trajectory, 4, offset), radius) for offset in range(4)]).mean()
    assert torch.isclose(curve[1], expected)


def test_composite_multiscale_entropy_shape_batched_and_swept():
    torch.manual_seed(3)
    trajectory = torch.randn(2, 100, 1, dtype=torch.float64)
    radius = torch.tensor([0.3, 0.6], dtype=torch.float64)
    curve = composite_multiscale_entropy(trajectory, scales=[1, 2, 3], radius=radius)
    assert curve.shape == (3, 2, 2)
    for scale_idx, scale in enumerate((1, 2, 3)):
        expected = torch.stack([sample_entropy(coarse_grain(trajectory, scale, k), radius) for k in range(scale)]).mean(0)
        assert torch.allclose(curve[scale_idx], expected, equal_nan=True)
    assert composite_multiscale_entropy(trajectory[0, :, 0], scales=[2], radius=0.3).shape == (1,)


def test_composite_multiscale_entropy_nonfinite_rows_and_propagation():
    torch.manual_seed(8)
    batch = torch.randn(3, 60, 1, dtype=torch.float64)
    batch[1, 5] = float("nan")
    curve = composite_multiscale_entropy(batch, scales=[1, 3], radius=0.5)
    # Regression: a non-finite row behaves exactly as under sample_entropy at scale 1.
    assert torch.allclose(curve[0], sample_entropy(batch, 0.5), equal_nan=True)
    for row in (0, 2):
        assert torch.equal(curve[:, row], composite_multiscale_entropy(batch[row], scales=[1, 3], radius=0.5))
    # An inf offset entropy (matches at length 2, none at length 3) is not excluded: plain mean.
    inf_series = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 9.0, 9.0, 9.0], dtype=torch.float64)
    assert torch.isinf(sample_entropy(inf_series, 0.5))
    doubled = inf_series.repeat_interleave(2)
    assert torch.equal(coarse_grain(doubled, 2, 0), inf_series)
    assert not torch.isfinite(composite_multiscale_entropy(doubled, scales=[2], radius=0.5)).any()


def test_composite_multiscale_entropy_too_coarse_scale_raises():
    trajectory = torch.randn(20, dtype=torch.float64)
    try:
        composite_multiscale_entropy(trajectory, scales=[7], radius=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_soft_composite_multiscale_entropy_matches_composition():
    torch.manual_seed(9)
    trajectory = torch.randn(25, dtype=torch.float64)
    curve = soft_composite_multiscale_entropy(trajectory, scales=[1, 3], temperature=0.5)
    expected = torch.stack([soft_sample_entropy(coarse_grain(trajectory, 3, k), 0.5) for k in range(3)]).mean()
    assert torch.isclose(curve[1], expected)


def test_soft_composite_multiscale_entropy_gradcheck():
    torch.manual_seed(10)
    trajectory = torch.randn(24, dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def entropy(series, bandwidth):
        return soft_composite_multiscale_entropy(series, scales=[1, 2, 3], temperature=bandwidth, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory, temperature))


def test_composite_multiscale_entropy_matches_complexitymeasures_jl():
    # Same SampleEntropy boundary deviation as above. Three far outliers at
    # the end sit in the last window of every offset series at scales 1-3,
    # and N = 41 = 5 mod 6 makes Julia's common truncation (N-s+1)//s equal
    # to every offset's natural length (N-k+1)//s, so the series are identical.
    torch.manual_seed(11)
    trajectory = torch.cat([torch.randn(38, dtype=torch.float64), torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64)])
    radius = 1.0
    scales = [1, 2, 3]
    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = SampleEntropy(m=2, τ=1, r={radius})
    ds = CompositeDownsampling(scales={scales[0]}:{scales[-1]})
    println(join(multiscale(ds, est, x), ","))
    """
    result = subprocess.run(["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    julia_curve = torch.tensor([float(v) for v in result.stdout.strip().split(",")], dtype=torch.float64)
    torch_curve = composite_multiscale_entropy(trajectory, scales=scales, radius=radius)
    assert torch.allclose(torch_curve, julia_curve, atol=1e-9)


def test_composite_multiscale_entropy_nan_offset_propagates():
    # Regression: one offset NaN (no template matches) and another finite gives NaN, not the finite value.
    torch.manual_seed(1)
    trajectory = torch.randn(24, dtype=torch.float64)
    per_offset = [sample_entropy(coarse_grain(trajectory, 2, offset), 0.3) for offset in range(2)]
    assert torch.isfinite(per_offset[0]) and torch.isnan(per_offset[1])
    assert torch.isnan(composite_multiscale_entropy(trajectory, scales=[2], radius=0.3)).all()


def test_composite_multiscale_entropy_forwards_nondefault_arguments():
    torch.manual_seed(12)
    trajectory = torch.randn(80, dtype=torch.float64)
    options = dict(embedding_dim=3, delay=2, theiler_window=2, norm_order=1.0)
    curve = composite_multiscale_entropy(trajectory, scales=[3], radius=1.2, **options)
    expected = torch.stack([sample_entropy(coarse_grain(trajectory, 3, k), 1.2, **options) for k in range(3)]).mean()
    assert torch.isclose(curve[0], expected)
    default = composite_multiscale_entropy(trajectory, scales=[3], radius=1.2)
    assert not torch.isclose(curve[0], default[0])


def test_soft_composite_multiscale_entropy_forwards_nondefault_arguments():
    torch.manual_seed(13)
    trajectory = torch.randn(80, dtype=torch.float64)
    options = dict(embedding_dim=3, delay=2, kind="exponential", theiler_window=2, norm_order=1.0)
    curve = soft_composite_multiscale_entropy(trajectory, scales=[3], temperature=0.8, **options)
    expected = torch.stack([soft_sample_entropy(coarse_grain(trajectory, 3, k), 0.8, **options) for k in range(3)]).mean()
    assert torch.isclose(curve[0], expected)
    default = soft_composite_multiscale_entropy(trajectory, scales=[3], temperature=0.8)
    assert not torch.isclose(curve[0], default[0])


def test_composite_multiscale_entropy_validates_scales_and_offset():
    trajectory = torch.randn(40, dtype=torch.float64)
    for bad_scales in ([], [0], [-1], [2.0], [1.5]):
        for function, argument in ((composite_multiscale_entropy, 0.5), (soft_composite_multiscale_entropy, 0.5)):
            try:
                function(trajectory, scales=bad_scales, **{("radius" if function is composite_multiscale_entropy else "temperature"): argument})
                assert False, "expected ValueError"
            except ValueError:
                pass
    for bad_offset in (0.5, 1.0):
        try:
            coarse_grain(trajectory, 3, offset=bad_offset)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_coarse_grain_and_composite_dtype_handling():
    integers = torch.arange(12)
    assert coarse_grain(integers, 2).dtype == torch.get_default_dtype()
    assert torch.allclose(coarse_grain(integers, 2, 1), torch.tensor([1.5, 3.5, 5.5, 7.5, 9.5]))
    torch.manual_seed(14)
    half = torch.randn(60).to(torch.float16)
    curve = composite_multiscale_entropy(half, scales=[1, 2], radius=0.5)
    assert curve.dtype == torch.float16
    reference = composite_multiscale_entropy(half.float(), scales=[1, 2], radius=0.5)
    assert torch.allclose(curve.float(), reference.to(torch.float16).float(), equal_nan=True)
    assert composite_multiscale_entropy(torch.arange(30) % 5, scales=[2], radius=0.5).dtype == torch.get_default_dtype()


def test_soft_composite_multiscale_entropy_shared_input_backward_is_finite():
    # Regression: a batch row whose kernel sums underflow to 0 must not put NaN into gradients of an input shared with other rows.
    torch.manual_seed(1)
    base = torch.randn(30, dtype=torch.float64, requires_grad=True)
    batch = torch.stack([base, base + 100 * torch.arange(30)]).unsqueeze(-1)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    curve = soft_composite_multiscale_entropy(batch, [2], temperature)
    assert torch.isnan(curve[0, 1])
    grad_base, grad_temperature = torch.autograd.grad(curve[0, 0], (base, temperature))
    assert torch.isfinite(grad_base).all() and torch.isfinite(grad_temperature)


def test_composite_multiscale_entropy_half_computed_in_float32():
    torch.manual_seed(0)
    series = torch.randn(60).half()
    expected = composite_multiscale_entropy(series.float(), [1, 2], 0.5).half()
    assert torch.equal(composite_multiscale_entropy(series, [1, 2], 0.5), expected)
    for scales in (None, 2):
        for function in (composite_multiscale_entropy, soft_composite_multiscale_entropy):
            try:
                function(series, scales, 0.5)
                assert False, "expected ValueError"
            except ValueError:
                pass
