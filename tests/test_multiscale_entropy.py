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

from torchnlts.entropy.multiscale import coarse_grain, multiscale_entropy, soft_multiscale_entropy
from torchnlts.entropy.sample import sample_entropy

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
    # Coverage gap: the temperature-only gradcheck above
    # never differentiates through coarse_grain's reshape/mean (the one
    # genuinely new piece of logic this module adds), since it holds
    # the trajectory fixed. This exercises that path directly.
    torch.manual_seed(6)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_multiscale_entropy(t, scales=[1, 2, 3], temperature=0.4, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_multiscale_entropy_too_coarse_scale_raises():
    # Documentation gap: a scale coarse enough to leave the
    # coarse-grained series shorter than embedding_dim+1 samples raises
    # (propagated from delay_embedding via sample_entropy) rather than
    # degrading that entry to NaN like a merely-degenerate-but-still-
    # embeddable scale does -- confirmed boundary, locked in here.
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
