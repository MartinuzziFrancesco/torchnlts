"""Tests for torchnlts.entropy.dispersion.

Dual ground truth:
  1. Paper: Rostaghi & Azami (2016)'s formula, as restated in Azami,
     Rostaghi, Abasolo & Escudero (2017, arXiv:1606.01379, eqs. 2-4) --
     hand-derived independently in Python (`statistics`/`math`, no
     torch) for a small example.
  2. Julia: ComplexityMeasures.jl's `Dispersion` outcome space +
     `Shannon` entropy. Exact float64-precision match on ordinary
     input (a small hand-derived example, a longer random one, and --
     checked specifically since PyTorch's `round` breaks ties to even
     and the class-rounding step can land exactly on a half-integer --
     an input engineered to hit that exact tie). One confirmed
     deviation: an exactly-constant trajectory, where Julia returns
     `-0.0` rather than this module's `NaN`, locked in below.
"""

import math

import numpy as np
import pytest
import torch

from tests.conftest import pseudo_random_series, run_julia
from torchnlts.embedding.delay import delay_embedding
from torchnlts.entropy.dispersion import _gaussian_cdf_classes, dispersion_entropy


def test_dispersion_entropy_hand_derived():
    # mean=3.75, sample std 1.669, c=3, m=2: classes [1,1,1,2,3,3,3,2], 5 distinct pairs over n=7,
    # H = -sum(p*ln(p)) = 1.5498260458782016 (computed with statistics/math).
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0], dtype=torch.float64)
    result = dispersion_entropy(trajectory, num_classes=3, embedding_dim=2, delay=1)
    assert torch.isclose(result, torch.tensor(1.5498260458782016, dtype=torch.float64))


def test_dispersion_entropy_normalized_matches_formula():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    raw = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    normalized = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2, normalize=True)
    expected = raw / torch.log(torch.tensor(4.0**2, dtype=torch.float64))
    assert torch.isclose(normalized, expected)


def test_dispersion_entropy_batched_matches_independent_loop():
    torch.manual_seed(1)
    trajectory = torch.randn(3, 50, 1, dtype=torch.float64)
    batched = dispersion_entropy(trajectory, num_classes=5, embedding_dim=3)
    assert batched.shape == (3,)
    for i in range(3):
        expected = dispersion_entropy(trajectory[i], num_classes=5, embedding_dim=3)
        assert torch.isclose(batched[i], expected)


def test_dispersion_entropy_shape_conventions_agree():
    torch.manual_seed(2)
    trajectory = torch.randn(40, dtype=torch.float64)
    scalar_shape = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    column_shape = dispersion_entropy(trajectory.unsqueeze(-1), num_classes=4, embedding_dim=2)
    assert scalar_shape.shape == () and column_shape.shape == ()
    assert torch.isclose(scalar_shape, column_shape)


def test_dispersion_entropy_rejects_multivariate():
    trajectory = torch.randn(30, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        dispersion_entropy(trajectory)


def test_dispersion_entropy_rejects_too_few_classes():
    trajectory = torch.randn(30, dtype=torch.float64)
    with pytest.raises(ValueError):
        dispersion_entropy(trajectory, num_classes=1)


def test_dispersion_entropy_constant_trajectory_is_nan():
    trajectory = torch.full((20,), 3.0, dtype=torch.float64)
    result = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.isnan(result)


def test_dispersion_entropy_near_constant_float32_is_finite_not_nan():
    # A constant float32 trajectory's std need not be exactly 0 (reduction rounding), so the NaN path is not reliable.
    trajectory = torch.full((30,), 0.1, dtype=torch.float32)
    assert trajectory.std().item() != 0.0
    result = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.isfinite(result)


def test_dispersion_entropy_pattern_codes_do_not_collide_at_large_scale():
    # Regression: float pattern codes collided once num_classes**embedding_dim exceeded float32's 2**24.
    # The first and last of the 26 windows share their first 24 classes and differ only in the last one.
    torch.manual_seed(10)
    block = torch.sign(torch.randn(24))
    block[0] = 1.0
    trajectory = torch.cat([block, torch.tensor([-1.0]), block, torch.tensor([1.0])])
    classes = _gaussian_cdf_classes(trajectory.view(1, -1, 1), 2).view(-1)
    windows = delay_embedding(classes, 25)
    assert windows.shape[0] == 26 and torch.unique(windows, dim=0).shape[0] == 26
    result = dispersion_entropy(trajectory, num_classes=2, embedding_dim=25)
    assert torch.isclose(result, torch.tensor(math.log(26)))


def test_dispersion_entropy_mixed_batch_nan_does_not_poison_other_rows():
    torch.manual_seed(3)
    constant = torch.full((30,), 1.0, dtype=torch.float64)
    normal = torch.randn(30, dtype=torch.float64)
    trajectory = torch.stack([constant, normal]).unsqueeze(-1)
    result = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.isnan(result[0])
    assert not torch.isnan(result[1])
    assert torch.isclose(result[1], dispersion_entropy(normal, num_classes=4, embedding_dim=2))


def test_dispersion_entropy_vmap_compatible():
    torch.manual_seed(4)
    trajectory = torch.randn(5, 40, dtype=torch.float64)
    direct = torch.stack([dispersion_entropy(row, num_classes=5, embedding_dim=3) for row in trajectory])
    vmapped = torch.vmap(lambda t: dispersion_entropy(t, num_classes=5, embedding_dim=3))(trajectory)
    assert torch.allclose(direct, vmapped)


def test_dispersion_entropy_matches_complexitymeasures_jl_hand_example():
    trajectory = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0]
    values = ", ".join(f"{v:.17g}" for v in trajectory)
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = Dispersion(c=3, m=2, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    stdout = run_julia(julia_code)
    julia_entropy = torch.tensor(float(stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(
        torch.tensor(trajectory, dtype=torch.float64), num_classes=3, embedding_dim=2, delay=1
    )
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-12, rtol=0)


def test_dispersion_entropy_matches_complexitymeasures_jl_at_exact_rounding_tie():
    # One value sits at the mean so round(c*y+0.5) hits 2.5 (c=4); torch.round ties to even, as Julia does.
    trajectory = [-1.0, 0.5, 0.5, 0.5, 0.5, 2.0]
    values = ", ".join(f"{v:.17g}" for v in trajectory)
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = Dispersion(c=4, m=2, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    stdout = run_julia(julia_code)
    julia_entropy = torch.tensor(float(stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(
        torch.tensor(trajectory, dtype=torch.float64), num_classes=4, embedding_dim=2, delay=1
    )
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-12, rtol=0)


def test_dispersion_entropy_disagrees_with_complexitymeasures_jl_on_constant_input():
    # Constant trajectory: Julia returns -0.0, this module returns NaN (documented deviation).
    julia_code = """
    using ComplexityMeasures
    x = fill(3.0, 20)
    est = Dispersion(c=4, m=2, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    stdout = run_julia(julia_code)
    julia_entropy = float(stdout.strip())
    assert julia_entropy == 0.0

    torch_entropy = dispersion_entropy(torch.full((20,), 3.0, dtype=torch.float64), num_classes=4, embedding_dim=2)
    assert torch.isnan(torch_entropy)


def test_dispersion_entropy_matches_complexitymeasures_jl_random():
    torch.manual_seed(9)
    trajectory = torch.randn(80, dtype=torch.float64)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = Dispersion(c=6, m=3, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    stdout = run_julia(julia_code)
    julia_entropy = torch.tensor(float(stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(trajectory, num_classes=6, embedding_dim=3, delay=1)
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-9, rtol=0)


def test_dispersion_entropy_large_pattern_space_needs_no_length_by_patterns_array():
    # Regression: 6**7 = 279936 patterns over 20000 windows; a one-hot count needs tens of GB.
    torch.manual_seed(0)
    trajectory = torch.randn(20000, dtype=torch.float64)
    num_classes, embedding_dim = 6, 7
    classes = _gaussian_cdf_classes(trajectory.view(1, -1, 1), num_classes).view(-1)
    _, counts = torch.unique(delay_embedding(classes, embedding_dim), dim=0, return_counts=True)
    probability = counts.to(torch.float64) / counts.sum()
    expected = -(probability * probability.log()).sum()
    result = dispersion_entropy(trajectory, num_classes=num_classes, embedding_dim=embedding_dim)
    assert torch.isclose(result, expected, rtol=1e-12, atol=0)


def test_dispersion_entropy_integer_input_matches_default_float_dtype():
    # Regression: integer input raised in mean().
    torch.manual_seed(5)
    series = torch.randint(0, 12, (200,))
    result = dispersion_entropy(series, num_classes=4, embedding_dim=2)
    assert result.dtype == torch.get_default_dtype()
    assert torch.equal(result, dispersion_entropy(series.to(torch.get_default_dtype()), num_classes=4, embedding_dim=2))
    batch = torch.randint(0, 12, (3, 200, 1))
    assert dispersion_entropy(batch, num_classes=4, embedding_dim=2).shape == (3,)


@pytest.mark.parametrize(
    "dtype, num_classes, series",
    [(torch.float16, 4, (300, 1, 20.0, 1e4)), (torch.bfloat16, 3, (100, 2, 20.0, 3000.0))],
)
def test_dispersion_entropy_half_precision_matches_float32_then_cast(dtype, num_classes, series):
    # A large offset with a small spread: the mean and standard deviation are inexact in half arithmetic.
    values = pseudo_random_series(*series).to(dtype)
    result = dispersion_entropy(values, num_classes=num_classes, embedding_dim=3)
    assert result.dtype == dtype and torch.isfinite(result)
    expected = dispersion_entropy(values.float(), num_classes=num_classes, embedding_dim=3).to(dtype)
    assert torch.equal(result, expected)


def test_dispersion_entropy_nonfinite_rows_are_nan():
    torch.manual_seed(6)
    batch = torch.randn(4, 60, 1, dtype=torch.float64)
    batch[1, 3] = float("nan")
    batch[2, 5] = float("inf")
    batch[3, 7] = -float("inf")
    result = dispersion_entropy(batch, num_classes=4, embedding_dim=2)
    assert torch.isnan(result[1:]).all()
    assert torch.equal(result[0], dispersion_entropy(batch[0], num_classes=4, embedding_dim=2))


@pytest.mark.parametrize("bad", [2.0, 2.5, "4", None, float("nan"), 1, 0, -2])
def test_dispersion_entropy_rejects_invalid_num_classes(bad):
    with pytest.raises(ValueError):
        dispersion_entropy(torch.randn(30, dtype=torch.float64), num_classes=bad)


def test_dispersion_entropy_accepts_numpy_integer_num_classes():
    trajectory = torch.randn(40, dtype=torch.float64)
    expected = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.equal(dispersion_entropy(trajectory, num_classes=np.int64(4), embedding_dim=2), expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_dispersion_entropy_nan_for_nonfinite_sample_outside_every_window(bad):
    # With embedding_dim=2 and delay=30 on 50 samples, samples 20-29 enter no window; the class thresholds still see them.
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = bad
    values = dispersion_entropy(batch, num_classes=4, embedding_dim=2, delay=30)
    assert torch.isnan(values[1]) and torch.isfinite(values[[0, 2]]).all()
    mapped = torch.vmap(lambda series: dispersion_entropy(series, num_classes=4, embedding_dim=2, delay=30))(batch)
    assert torch.allclose(values, mapped, equal_nan=True)
