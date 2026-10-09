"""Tests for torchnlts.entropy.fuzzy.

Dual ground truth:
  1. Paper: Chen, Wang, Xie & Yu (2007), eq. 2-10 -- hand-derived example.
  2. EntropyHub's ``FuzzEn`` (``Fx='default'`` matches the paper exactly),
     used in place of a Julia reference since ComplexityMeasures.jl has
     no fuzzy entropy estimator.
"""

import math

import torch

from torchnlts.entropy.fuzzy import fuzzy_entropy


def test_fuzzy_entropy_hand_derived_embedding_dim_1():
    # Hand-derived: w0 -> (-0.5, 0.5), w1 -> (-1, 1), Chebyshev d = 0.5, D = exp(-0.25); phi^1 = 1 for embedding_dim=1.
    # FuzzyEn = ln(phi^1) - ln(phi^2) = 0 - (-0.25) = 0.25.
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=2.0, embedding_dim=1, delay=1)
    assert torch.isclose(result, torch.tensor(0.25, dtype=torch.float64))


def test_fuzzy_entropy_hand_derived_embedding_dim_2():
    # Re-derived from the paper's eq. 2-10 and matched to EntropyHub.FuzzEn(..., r=(1.0, 2.0), Logx=math.e).
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.5, 2.5, 3.5], dtype=torch.float64)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=2.0, embedding_dim=2, delay=1)
    assert torch.isclose(result, torch.tensor(0.6567823559794231, dtype=torch.float64))


def test_fuzzy_entropy_matches_entropyhub():
    import numpy as np
    from EntropyHub import FuzzEn

    torch.manual_seed(0)
    trajectory = torch.randn(40, dtype=torch.float64)
    radius, fuzzy_power, embedding_dim = 0.7, 2.0, 2

    torch_entropy = fuzzy_entropy(trajectory, radius=radius, fuzzy_power=fuzzy_power, embedding_dim=embedding_dim)
    fuzz, _, _ = FuzzEn(
        trajectory.numpy().astype(np.float64), m=embedding_dim, tau=1, r=(radius, fuzzy_power), Logx=math.e
    )
    entropyhub_entropy = torch.tensor(fuzz[embedding_dim - 1], dtype=torch.float64)
    assert torch.isclose(torch_entropy, entropyhub_entropy, rtol=0, atol=1e-9)


def test_fuzzy_entropy_batched_and_sweep_shapes():
    torch.manual_seed(1)
    trajectory = torch.randn(4, 30, 2, dtype=torch.float64)
    radii = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    swept = fuzzy_entropy(trajectory, radius=radii, embedding_dim=2)
    assert swept.shape == (3, 4)
    for i, radius in enumerate(radii):
        expected = fuzzy_entropy(trajectory, radius=radius, embedding_dim=2)
        assert expected.shape == (4,)
        assert torch.allclose(swept[i], expected, equal_nan=True)


def test_fuzzy_entropy_batch_size_one_matches_unbatched():
    torch.manual_seed(2)
    trajectory = torch.randn(1, 20, 2, dtype=torch.float64)
    batched = fuzzy_entropy(trajectory, radius=1.0, embedding_dim=2)
    unbatched = fuzzy_entropy(trajectory[0], radius=1.0, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_fuzzy_entropy_negative_theiler_window_raises():
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    try:
        fuzzy_entropy(trajectory, radius=1.5, embedding_dim=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fuzzy_entropy_gradcheck_embedding_dim_3():
    # embedding_dim=3 avoids the systematic Chebyshev tie of embedding_dim=2.
    torch.manual_seed(3)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return fuzzy_entropy(x, radius=1.0, embedding_dim=3)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_fuzzy_entropy_gradcheck_embedding_dim_2_euclidean():
    # norm_order=2 avoids the Chebyshev tie at embedding_dim=2.
    torch.manual_seed(4)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return fuzzy_entropy(x, radius=1.0, embedding_dim=2, norm_order=2.0)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_fuzzy_entropy_gradcheck_radius():
    torch.manual_seed(5)
    trajectory = torch.randn(15, dtype=torch.float64)
    radius = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    def entropy(r):
        return fuzzy_entropy(trajectory, radius=r, embedding_dim=3)

    assert torch.autograd.gradcheck(entropy, (radius,))


def test_fuzzy_entropy_embedding_dim_2_chebyshev_gradient_mismatches_central_difference():
    # Locks in the documented wrong autograd gradient at the Chebyshev tie (cdist backward sums tied components).
    # If torch changes cdist's tie-breaking, this fails and the module docstring caveat needs revisiting.
    trajectory = torch.tensor([0.0, 1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, embedding_dim=2)
    result.backward()
    assert torch.isfinite(trajectory.grad).all()

    eps = 1e-6
    central_difference = torch.empty(4, dtype=torch.float64)
    for i in range(4):
        bumped_up = trajectory.detach().clone()
        bumped_up[i] += eps
        bumped_down = trajectory.detach().clone()
        bumped_down[i] -= eps
        central_difference[i] = (
            fuzzy_entropy(bumped_up, radius=1.0, embedding_dim=2)
            - fuzzy_entropy(bumped_down, radius=1.0, embedding_dim=2)
        ) / (2 * eps)
    assert not torch.allclose(trajectory.grad, central_difference, atol=1e-3)


def test_fuzzy_entropy_fractional_power_gradient_is_not_poisoned():
    # Regression: fuzzy_power < 1 gave NaN gradients through the masked diagonal (0 * inf).
    torch.manual_seed(42)
    trajectory = torch.randn(12, dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=0.5, embedding_dim=3)
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(trajectory.grad).all()
    assert (trajectory.grad != 0).any()

    assert torch.autograd.gradcheck(
        lambda x: fuzzy_entropy(x, radius=1.0, fuzzy_power=0.5, embedding_dim=3),
        (trajectory.detach().requires_grad_(),),
    )


def test_fuzzy_entropy_embedding_dim_1_long_branch_also_ties():
    # embedding_dim=1's long branch has the same Chebyshev tie; embedding_dim >= 3 avoids it.
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, embedding_dim=1)
    result.backward()
    assert torch.isfinite(trajectory.grad).all()

    eps = 1e-6
    central_difference = torch.empty(3, dtype=torch.float64)
    for i in range(3):
        bumped_up = trajectory.detach().clone()
        bumped_up[i] += eps
        bumped_down = trajectory.detach().clone()
        bumped_down[i] -= eps
        central_difference[i] = (
            fuzzy_entropy(bumped_up, radius=1.0, embedding_dim=1)
            - fuzzy_entropy(bumped_down, radius=1.0, embedding_dim=1)
        ) / (2 * eps)
    assert not torch.allclose(trajectory.grad, central_difference, atol=1e-3)


def test_fuzzy_entropy_univariate_embedding_dim_1_fractional_power_gradient_is_not_poisoned():
    # Regression: embedding_dim=1 short templates are zero, so every distance sits at the fuzzy_power < 1 singularity.
    # Needs the Chebyshev default to exercise the fix; asserts finiteness only (long-branch tie, see above).
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=0.5, embedding_dim=1)
    assert torch.isclose(result, torch.tensor(2**0.5 / 2, dtype=torch.float64))
    result.backward()
    assert torch.isfinite(trajectory.grad).all()


def test_fuzzy_entropy_gradcheck_embedding_dim_1_euclidean():
    torch.manual_seed(9)
    trajectory = torch.randn(12, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return fuzzy_entropy(x, radius=1.0, fuzzy_power=0.5, embedding_dim=1, norm_order=2.0)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_fuzzy_entropy_multivariate_embedding_dim_1_not_short_circuited():
    # A multivariate embedding_dim=1 template is not zero after mean-removal; compare against the general pipeline.
    from torchnlts.entropy.fuzzy import _fuzzy_correlation
    from torchnlts.entropy.sample import _matched_length_templates

    torch.manual_seed(1)
    trajectory = torch.randn(10, 2, dtype=torch.float64, requires_grad=True)
    radius, fuzzy_power, embedding_dim = 1.0, 2.0, 1

    short_embed, long_embed = _matched_length_templates(trajectory.detach(), embedding_dim, 1)
    short_embed = short_embed - short_embed.mean(dim=-1, keepdim=True)
    long_embed = long_embed - long_embed.mean(dim=-1, keepdim=True)
    assert short_embed.shape[-1] == 2, "test assumption: D=2 must NOT trigger the univariate D=1 special case"
    expected_phi_short = _fuzzy_correlation(short_embed, radius, fuzzy_power, 0, float("inf"))
    expected_phi_long = _fuzzy_correlation(long_embed, radius, fuzzy_power, 0, float("inf"))
    expected = -torch.log(expected_phi_long / expected_phi_short)

    result = fuzzy_entropy(trajectory, radius=radius, embedding_dim=embedding_dim, fuzzy_power=fuzzy_power)
    assert torch.isclose(result, expected)
    result.backward()
    assert torch.isfinite(trajectory.grad).all()
    assert (trajectory.grad != 0).any()
