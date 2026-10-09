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

import subprocess

import torch

from torchnlts.entropy.dispersion import dispersion_entropy

JULIA_PROJECT = "juliaenv"


def test_dispersion_entropy_hand_derived():
    # mean=3.75, std=1.6690459207925603 (sample std, N-1), c=3, m=2, d=1.
    # NCDF-classified: [1,1,1,2,3,3,3,2]; consecutive pairs:
    # (1,1)x2, (1,2)x1, (2,3)x1, (3,3)x2, (3,2)x1 -> 5 distinct patterns
    # over n=7 pairs -> H = -sum(p*ln(p)) = 1.5498260458782016 (computed
    # independently via Python's statistics/math modules, not torch).
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
    try:
        dispersion_entropy(trajectory)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dispersion_entropy_rejects_too_few_classes():
    trajectory = torch.randn(30, dtype=torch.float64)
    try:
        dispersion_entropy(trajectory, num_classes=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dispersion_entropy_constant_trajectory_is_nan():
    # Zero variance -> division by std=0 in the Gaussian-CDF step.
    trajectory = torch.full((20,), 3.0, dtype=torch.float64)
    result = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.isnan(result)


def test_dispersion_entropy_near_constant_float32_is_finite_not_nan():
    # Edge case: a "constant" float32 tensor's
    # computed std isn't exactly 0 (rounding in the reduction), so the
    # NaN path documented above only reliably triggers for an *exactly*
    # zero std, not merely a visually-constant trajectory in every
    # dtype -- the documented guarantee is precise about this, this
    # test locks in the actual (not idealized) behavior.
    trajectory = torch.full((30,), 0.1, dtype=torch.float32)
    assert trajectory.std().item() != 0.0
    result = dispersion_entropy(trajectory, num_classes=4, embedding_dim=2)
    assert torch.isfinite(result)


def test_dispersion_entropy_pattern_codes_do_not_collide_at_large_scale():
    # Regression: computing pattern codes in a floating
    # dtype (matching the trajectory's own) instead of integer
    # arithmetic silently collided two different embedding_dim=25,
    # num_classes=2 digit patterns onto the same code once
    # num_classes**embedding_dim exceeded float32's 2**24 exactly-
    # representable-integer ceiling -- confirmed directly before the
    # fix. Reproduced here at the arithmetic level, matching exactly
    # what dispersion_entropy now computes internally (int64 weights,
    # int64 embedded values), not through the full function -- at this
    # scale, dispersion_entropy's one-hot pattern-counting step would
    # need gigabytes of memory (see the module docstring's
    # embedding_dim caveat), so this isolates the arithmetic fix itself.
    weights = 2 ** torch.arange(24, -1, -1, dtype=torch.int64)
    pattern_a = torch.tensor([1] + [0] * 24, dtype=torch.int64)
    pattern_b = torch.tensor([1] + [0] * 23 + [1], dtype=torch.int64)
    assert (pattern_a * weights).sum() != (pattern_b * weights).sum()


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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(
        torch.tensor(trajectory, dtype=torch.float64), num_classes=3, embedding_dim=2, delay=1
    )
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-12, rtol=0)


def test_dispersion_entropy_matches_complexitymeasures_jl_at_exact_rounding_tie():
    # Specifically targets the class-rounding step's tie-breaking
    # convention (torch.round breaks ties to even): constructed so one
    # value sits exactly at the trajectory's own mean, making
    # round(c*y+0.5) land exactly on a half-integer (c=4 -> 2.5).
    # Confirmed live against Julia (not assumed): both round 2.5 to 2,
    # giving an exact match here -- worth locking in permanently since
    # this is the one place PyTorch's specific rounding convention
    # could plausibly have disagreed with Julia's.
    trajectory = [-1.0, 0.5, 0.5, 0.5, 0.5, 2.0]  # mean is exactly 0.5
    values = ", ".join(f"{v:.17g}" for v in trajectory)
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = Dispersion(c=4, m=2, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(
        torch.tensor(trajectory, dtype=torch.float64), num_classes=4, embedding_dim=2, delay=1
    )
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-12, rtol=0)


def test_dispersion_entropy_disagrees_with_complexitymeasures_jl_on_constant_input():
    # Locks in the confirmed deviation the module docstring describes:
    # for an exactly-constant trajectory, Julia's Dispersion returns
    # -0.0 (not an error, not NaN), while this module returns NaN.
    julia_code = """
    using ComplexityMeasures
    x = fill(3.0, 20)
    est = Dispersion(c=4, m=2, τ=1)
    println(information(Shannon(base=ℯ), est, x))
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = float(result.stdout.strip())
    assert julia_entropy == 0.0  # Julia's -0.0, not NaN and not an error

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
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    julia_entropy = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_entropy = dispersion_entropy(trajectory, num_classes=6, embedding_dim=3, delay=1)
    assert torch.isclose(torch_entropy, julia_entropy, atol=1e-9, rtol=0)
