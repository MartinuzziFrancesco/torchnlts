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
    # embedding_dim=1: every mean-removed 1-length template is exactly
    # [0.0] (a window's own mean equals itself), so phi^1 = 1
    # regardless of radius/fuzzy_power -- isolates the m+1=2 term.
    # long templates (delay=1): w0=(0,1) mean 0.5 -> (-0.5, 0.5);
    # w1=(1,3) mean 2.0 -> (-1.0, 1.0). Chebyshev distance
    # max(|-0.5+1.0|, |0.5-1.0|) = 0.5. D_01 = exp(-0.5^2/1.0) = exp(-0.25).
    # FuzzyEn = ln(phi^1) - ln(phi^2) = 0 - (-0.25) = 0.25.
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=2.0, embedding_dim=1, delay=1)
    assert torch.isclose(result, torch.tensor(0.25, dtype=torch.float64))


def test_fuzzy_entropy_hand_derived_embedding_dim_2():
    # Independently re-derived from the paper's own eq. 2-10 (not
    # copied from EntropyHub's source) in plain Python, then confirmed
    # to match EntropyHub.FuzzEn(..., r=(1.0, 2.0), Logx=math.e) exactly.
    trajectory = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.5, 2.5, 3.5], dtype=torch.float64
    )
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=2.0, embedding_dim=2, delay=1)
    assert torch.isclose(result, torch.tensor(0.6567823559794231, dtype=torch.float64))


def test_fuzzy_entropy_matches_entropyhub():
    import numpy as np
    from EntropyHub import FuzzEn

    torch.manual_seed(0)
    trajectory = torch.randn(40, dtype=torch.float64)
    radius, fuzzy_power, embedding_dim = 0.7, 2.0, 2

    torch_entropy = fuzzy_entropy(trajectory, radius=radius, fuzzy_power=fuzzy_power, embedding_dim=embedding_dim)
    fuzz, _, _ = FuzzEn(trajectory.numpy().astype(np.float64), m=embedding_dim, tau=1, r=(radius, fuzzy_power), Logx=math.e)
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
    # embedding_dim=2's default Chebyshev distance has a systematic tie
    # after mean-removal (see the module docstring); embedding_dim=3
    # has no such tie, so it's the cleanest gradcheck of the default
    # norm_order.
    torch.manual_seed(3)
    trajectory = torch.randn(15, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return fuzzy_entropy(x, radius=1.0, embedding_dim=3)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_fuzzy_entropy_gradcheck_embedding_dim_2_euclidean():
    # embedding_dim=2 with norm_order=2 avoids the documented systematic
    # Chebyshev tie, so this exercises the default embedding_dim cleanly.
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
    # Locks in the module docstring's claim: at the default
    # embedding_dim/norm_order, autograd's gradient is finite but
    # confirmed *wrong* (torch.cdist's backward sums the gradient
    # contribution across every tied argmax component of the
    # systematic (v, -v) tie), not merely a non-canonical subgradient
    # choice -- so it disagrees with a central-difference estimate by
    # more than any reasonable numerical tolerance. If a future
    # PyTorch version changes cdist's tie-breaking convention, this
    # test starts failing, which means this caveat (and
    # test_fuzzy_entropy_gradcheck_embedding_dim_2_euclidean's
    # implicit contrast) should be revisited.
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
            fuzzy_entropy(bumped_up, radius=1.0, embedding_dim=2) - fuzzy_entropy(bumped_down, radius=1.0, embedding_dim=2)
        ) / (2 * eps)
    assert not torch.allclose(trajectory.grad, central_difference, atol=1e-3)


def test_fuzzy_entropy_fractional_power_gradient_is_not_poisoned():
    # Regression test for a torch.where gradient-poisoning bug (same
    # class as torchnlts/entropy/generalized.py's history): the
    # diagonal's distance is always exactly 0, and
    # distance**fuzzy_power has an infinite local gradient there for
    # fuzzy_power < 1, which used to poison every trajectory gradient
    # via 0 * inf = NaN even though the diagonal is masked out of the
    # forward sum. Fixed via a safe placeholder distance before the
    # power (mirroring trajectory_kernel's finite-distance substitution).
    #
    # Uses a longer, non-degenerate trajectory rather than the earlier
    # [0, 1, 3, 7] (too short at embedding_dim=3 for any long-template
    # pair to exist, giving a trivially all-zero gradient through a
    # NaN forward value -- passing `isfinite` without exercising real
    # gradient flow through a defined entropy).
    torch.manual_seed(42)
    trajectory = torch.randn(12, dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=0.5, embedding_dim=3)
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(trajectory.grad).all()
    assert (trajectory.grad != 0).any()

    assert torch.autograd.gradcheck(
        lambda x: fuzzy_entropy(x, radius=1.0, fuzzy_power=0.5, embedding_dim=3), (trajectory.detach().requires_grad_(),)
    )


def test_fuzzy_entropy_embedding_dim_1_long_branch_also_ties():
    # embedding_dim=1's *long* (m+1=2) branch has the same systematic
    # (v, -v) tie as embedding_dim=2's short branch, so it's affected
    # by the same confirmed-wrong-not-just-noncanonical gradient (see
    # module docstring); embedding_dim >= 3 is required to avoid a
    # length-2 template on either branch, not just embedding_dim != 2.
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
            fuzzy_entropy(bumped_up, radius=1.0, embedding_dim=1) - fuzzy_entropy(bumped_down, radius=1.0, embedding_dim=1)
        ) / (2 * eps)
    assert not torch.allclose(trajectory.grad, central_difference, atol=1e-3)


def test_fuzzy_entropy_univariate_embedding_dim_1_fractional_power_gradient_is_not_poisoned():
    # Regression test for a second, distinct torch.where-style
    # poisoning case (not the masked-diagonal one the first fractional-
    # power test covers): a univariate embedding_dim=1 short template
    # is the zero vector *unconditionally* (its own mean always equals
    # itself), so every pairwise distance in that branch -- not just
    # the masked diagonal -- sits at the fuzzy_power<1 singularity.
    #
    # Must use the default Chebyshev distance to actually exercise the
    # fix: confirmed directly that torch.cdist's Euclidean
    # (norm_order=2.0) backward already returns a finite gradient at a
    # zero distance on its own, so a norm_order=2.0 version of this
    # test would pass whether or not the fix (or even the original
    # bug) were present -- caught only because a mutation
    # check removed the fix in memory and reran this test's
    # earlier norm_order=2.0 form, which didn't catch it.
    #
    # Chebyshev also carries the module docstring's *separate*,
    # already-covered long-branch tie caveat, so this only asserts
    # finiteness (what the fix actually guarantees), not full
    # gradcheck agreement -- test_fuzzy_entropy_gradcheck_embedding_dim_1_euclidean
    # below covers general correctness under a norm where the
    # singularity doesn't arise.
    trajectory = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = fuzzy_entropy(trajectory, radius=1.0, fuzzy_power=0.5, embedding_dim=1)
    assert torch.isclose(result, torch.tensor(2**0.5 / 2, dtype=torch.float64))
    result.backward()
    assert torch.isfinite(trajectory.grad).all()


def test_fuzzy_entropy_gradcheck_embedding_dim_1_euclidean():
    # General correctness check for the embedding_dim=1 special case
    # under a norm where neither documented caveat (the Chebyshev tie,
    # the Chebyshev-backward fractional-power poisoning) arises, so
    # this can assert full gradcheck agreement -- unlike the Chebyshev
    # regression test above, which can only assert finiteness.
    torch.manual_seed(9)
    trajectory = torch.randn(12, dtype=torch.float64, requires_grad=True)

    def entropy(x):
        return fuzzy_entropy(x, radius=1.0, fuzzy_power=0.5, embedding_dim=1, norm_order=2.0)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_fuzzy_entropy_multivariate_embedding_dim_1_not_short_circuited():
    # The embedding_dim=1 zero-template special case only holds for a
    # *univariate* trajectory (flattened template length 1); a
    # multivariate one's embedding_dim=1 template flattens to D
    # values whose mean-removal generally isn't zero, so it must still
    # go through the general pipeline. Asserts the forward value
    # itself matches an independent reconstruction of the general
    # pipeline (not just finiteness/nonzero-gradient, which a wrong
    # embedding_dim==1-only condition would also satisfy -- caught by
    # a mutation check, which restored that wrong condition
    # in memory and found this test's earlier finiteness-only form
    # still passed with a materially different forward value).
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
