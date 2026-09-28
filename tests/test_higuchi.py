"""Tests for torchnlts.dimension.higuchi.

Dual ground truth:
  1. Paper: Higuchi (1988) -- checked via a hand-derived example
     (independently re-derived by hand, then confirmed to match
     FractalDimensions.jl's own output exactly).
  2. Julia: FractalDimensions.jl's ``higuchi_length``/``higuchi_dim``.
"""

import subprocess

import torch

from torchnlts.dimension.higuchi import higuchi_dimension, higuchi_length

JULIA_PROJECT = "juliaenv"


def test_higuchi_length_hand_derived():
    # x = [1,2,1,3,2,4,1,5], N=8.
    # k=1: single offset m=0, 7 consecutive diffs: 1,1,2,1,2,3,4 -> sum=14,
    #   normalization=(8-1)/(7*1*1)=1 -> L(1)=14.
    # k=2: offset m=0: steps at indices 0,2,4,6 -> diffs |1-1|,|2-1|,|1-2|=0,1,1
    #   sum=2, num_steps=3, norm=7/(3*4)=7/12 -> L_0=7/6.
    #   offset m=1: indices 1,3,5,7 -> diffs |3-2|,|4-3|,|5-4|=1,1,1, sum=3,
    #   num_steps=3, norm=7/12 -> L_1=7/4. L(2)=(7/6+7/4)/2=1.4583333333333335.
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    length = higuchi_length(trajectory, [1, 2])
    assert torch.allclose(length, torch.tensor([14.0, 1.4583333333333335], dtype=torch.float64))


def test_higuchi_dimension_hand_derived():
    trajectory = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    k = torch.tensor([1, 2], dtype=torch.float64)
    length = higuchi_length(trajectory, [1, 2])
    dimension = higuchi_dimension(k, length)
    assert torch.isclose(dimension, torch.tensor(3.263034405833794, dtype=torch.float64))


def test_higuchi_length_degenerate_k_is_nan_not_crash():
    # k=9 on a length-10 trajectory: the offset m=1 has (10-1-1)//9=0 valid
    # steps, matching Julia's own natural division-by-zero-to-NaN failure
    # mode for that offset (confirmed directly against FractalDimensions.jl).
    trajectory = torch.arange(1.0, 11.0, dtype=torch.float64)
    result = higuchi_length(trajectory, 9)
    assert torch.isnan(result)


def test_higuchi_length_zero_k_raises():
    trajectory = torch.arange(1.0, 11.0, dtype=torch.float64)
    try:
        higuchi_length(trajectory, [0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_higuchi_length_fractional_k_raises():
    # k indexes a stride, not a continuous quantity -- unlike
    # radius/box_size elsewhere, a non-integer value is rejected
    # rather than silently truncated (confirmed directly: an earlier
    # version silently truncated [1.9, 2.9] to [1, 2] with no error).
    trajectory = torch.arange(1.0, 21.0, dtype=torch.float64)
    try:
        higuchi_length(trajectory, [1.9, 2.9])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_higuchi_length_empty_k_raises():
    trajectory = torch.arange(1.0, 11.0, dtype=torch.float64)
    try:
        higuchi_length(trajectory, [])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_higuchi_length_empty_trajectory_is_nan_not_crash():
    # Falls into the same degenerate-k NaN case as any other too-short
    # trajectory, not a separate crash -- a confirmed, documented
    # deviation from FractalDimensions.jl's own higuchi_length, which
    # returns 0.0 (not NaN) for a literally empty input.
    empty = torch.tensor([], dtype=torch.float64)
    result = higuchi_length(empty, [1, 2])
    assert torch.isnan(result).all()


def test_higuchi_length_multivariate_raises():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    try:
        higuchi_length(trajectory, 2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_higuchi_length_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 30, 1, dtype=torch.float64)
    ks = [1, 2, 3, 4]
    swept = higuchi_length(trajectory, ks)
    assert swept.shape == (4, 4)
    for i, k in enumerate(ks):
        expected = higuchi_length(trajectory, k)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected)


def test_higuchi_length_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 30, 1, dtype=torch.float64)
    batched = higuchi_length(trajectory, [1, 2, 3])
    unbatched = higuchi_length(trajectory[0], [1, 2, 3])
    assert batched.shape == (3, 1)
    assert torch.allclose(batched[:, 0], unbatched)


def test_higuchi_length_unbatched_scalar_k_is_scalar():
    trajectory = torch.randn(20, dtype=torch.float64)
    result = higuchi_length(trajectory, 3)
    assert result.shape == ()


def test_higuchi_length_gradcheck():
    torch.manual_seed(2)
    trajectory = torch.randn(25, dtype=torch.float64, requires_grad=True)

    def length(x):
        return higuchi_length(x, [1, 2, 3, 4])

    assert torch.autograd.gradcheck(length, (trajectory,))


def test_higuchi_length_gradcheck_with_exact_tie():
    # x[0]=0, x[2]=0: the stride-2 diff between indices 0 and 2 is an
    # exact abs() kink (a genuine tie, not just a random-data coincidence).
    # Unlike torchnlts/entropy/fuzzy.py's torch.where-based poisoning
    # history, this module has no masked/discarded branch for backward
    # to poison through -- confirmed passing here.
    trajectory = torch.tensor([0.0, 2.0, 0.0, 5.0, 1.0, 4.0, -1.0, 3.0], dtype=torch.float64, requires_grad=True)

    def length(x):
        return higuchi_length(x, [1, 2, 3])

    assert torch.autograd.gradcheck(length, (trajectory,))


def test_higuchi_dimension_gradcheck_through_trajectory():
    torch.manual_seed(3)
    trajectory = torch.randn(25, dtype=torch.float64, requires_grad=True)
    k = torch.tensor([1, 2, 3, 4], dtype=torch.float64)

    def dimension(x):
        return higuchi_dimension(k, higuchi_length(x, [1, 2, 3, 4]))

    assert torch.autograd.gradcheck(dimension, (trajectory,))


def test_higuchi_dimension_single_k_is_nan_not_crash():
    result = higuchi_dimension(torch.tensor([1.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64))
    assert torch.isnan(result)


def test_higuchi_matches_fractaldimensions_jl():
    torch.manual_seed(4)
    trajectory = torch.randn(200, dtype=torch.float64)
    ks = [1, 2, 3, 4, 6, 8, 12, 16]

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using FractalDimensions
    x = [{values}]
    L = FractalDimensions.higuchi_length(x, {ks})
    println(join(L, ","))
    d = higuchi_dim(x, {ks})
    println(d)
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().split("\n")
    julia_length = torch.tensor([float(v) for v in lines[0].split(",")], dtype=torch.float64)
    julia_dimension = torch.tensor(float(lines[1]), dtype=torch.float64)

    torch_length = higuchi_length(trajectory, ks)
    assert torch.allclose(torch_length, julia_length, atol=1e-9)

    torch_dimension = higuchi_dimension(torch.tensor(ks, dtype=torch.float64), torch_length)
    assert torch.isclose(torch_dimension, julia_dimension, atol=1e-9)


def test_higuchi_dimension_in_expected_range_for_known_signals():
    # A smooth sine wave should be near D_H=1 (smooth curve); white noise
    # should be near D_H=2 (maximally rough) -- Higuchi's own documented
    # [1, 2] range, checked here against two clearly-contrasting real
    # signals rather than trusting the range claim on faith.
    t = torch.linspace(0, 20 * torch.pi, 2000, dtype=torch.float64)
    sine = torch.sin(t)
    ks = torch.arange(1, 21, dtype=torch.float64)
    sine_dim = higuchi_dimension(ks, higuchi_length(sine, [int(k) for k in ks.tolist()]))
    assert sine_dim.item() < 1.3

    torch.manual_seed(5)
    noise = torch.randn(2000, dtype=torch.float64)
    noise_dim = higuchi_dimension(ks, higuchi_length(noise, [int(k) for k in ks.tolist()]))
    assert noise_dim.item() > 1.8


def test_higuchi_length_invalid_trajectory_raises():
    import pytest

    for bad in (torch.tensor([0j, 1 + 2j, 2j, 3j]), torch.tensor([0.0, float("inf"), 1.0, 2.0]), torch.tensor([0.0, float("nan"), 1.0, 2.0])):
        with pytest.raises(ValueError):
            higuchi_length(bad, 1)
    with pytest.raises(ValueError):
        higuchi_length(torch.arange(8.0), torch.tensor([1 + 1j]))


def test_higuchi_length_half_precision_does_not_overflow():
    # Absolute differences of 30000 summed in float16 overflow (max 65504).
    trajectory = torch.tensor([0, 0, 30000, 30000, 0, 0, 30000, 30000], dtype=torch.float16)
    result = higuchi_length(trajectory, 2)
    assert result.dtype == torch.float16 and torch.isfinite(result)
    assert torch.isclose(result.float(), higuchi_length(trajectory.float(), 2), rtol=1e-3)
