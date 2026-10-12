"""Tests for torchnlts.entropy.fuzzy.

Dual ground truth:
  1. Paper: Chen, Wang, Xie & Yu (2007), eq. 2-10 -- hand-derived example.
  2. EntropyHub's ``FuzzEn`` (``Fx='default'`` matches the paper exactly),
     used in place of a Julia reference since ComplexityMeasures.jl has
     no fuzzy entropy estimator.
"""

import math

import pytest
import torch

from tests.conftest import assert_first_row_gradients_unaffected
from torchnlts.entropy.fuzzy import fuzzy_entropy

NONFINITE = [float("nan"), float("inf"), -float("inf")]
# With embedding_dim=1 and delay=30 on 50 samples, samples 20-29 enter no template.
GAP = dict(embedding_dim=1, delay=30)


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


def test_fuzzy_entropy_follows_the_input_dtype():
    torch.manual_seed(9)
    reference = torch.randn(60)
    integers = (4 * reference).round().long()
    result = fuzzy_entropy(integers, 0.5)
    assert result.dtype == torch.get_default_dtype() and torch.isfinite(result)
    assert torch.equal(result, fuzzy_entropy(integers.to(torch.get_default_dtype()), 0.5))
    for dtype in (torch.float16, torch.bfloat16):
        half = reference.to(dtype)
        result = fuzzy_entropy(half, 0.5)
        assert result.dtype == dtype and torch.isfinite(result)
        assert torch.equal(result, fuzzy_entropy(half.float(), 0.5).to(dtype))
    with pytest.raises(ValueError, match="complex"):
        fuzzy_entropy(torch.ones(20, dtype=torch.complex64), 0.5)


def test_fuzzy_entropy_negative_theiler_window_raises():
    trajectory = torch.tensor([0.0, 1.0, 0.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        fuzzy_entropy(trajectory, radius=1.5, embedding_dim=1, theiler_window=-1)


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


@pytest.mark.parametrize("bad", NONFINITE)
def test_fuzzy_entropy_nan_for_nonfinite_sample_outside_every_template(bad):
    # Regression: a non-finite sample that enters no template used to leave its row finite.
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = bad
    values = fuzzy_entropy(batch, 0.5, **GAP)
    assert torch.isnan(values[1]) and torch.isfinite(values[[0, 2]]).all()
    for row in (0, 2):
        assert torch.equal(values[row], fuzzy_entropy(batch[row], 0.5, **GAP))
    assert torch.isnan(fuzzy_entropy(batch[1], 0.5, **GAP)) and torch.isnan(fuzzy_entropy(batch[1, :, 0], 0.5, **GAP))
    swept = fuzzy_entropy(batch, torch.tensor([0.5, 0.8], dtype=torch.float64), **GAP)
    assert swept.shape == (2, 3) and torch.isnan(swept[:, 1]).all() and torch.isfinite(swept[:, [0, 2]]).all()


@pytest.mark.parametrize(
    "options, position",
    [
        (dict(embedding_dim=3), 5),
        (dict(embedding_dim=2, norm_order=2.0), 5),
        (dict(embedding_dim=3, fuzzy_power=0.5), 5),
        (GAP, 25),
    ],
    ids=["inside_window", "euclidean", "fractional_power", "outside_every_template"],
)
def test_fuzzy_entropy_nonfinite_row_leaves_other_gradients(options, position):
    # Regression: the NaN row put NaN into the gradients of the other rows and of the shared radius.
    torch.manual_seed(1)
    clean, bad_row = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    bad_row[position] = float("nan")

    def gradients(rows):
        batch = torch.stack(rows).unsqueeze(-1).requires_grad_()
        radius = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        fuzzy_entropy(batch, radius, **options).nansum().backward()
        return batch.grad, radius.grad

    batch_grad, radius_grad = gradients([bad_row, clean])
    single_grad, single_radius_grad = gradients([clean])
    assert torch.isfinite(batch_grad).all() and (batch_grad[0] == 0).all()
    assert torch.allclose(batch_grad[1], single_grad[0])
    assert torch.isfinite(radius_grad) and torch.allclose(radius_grad, single_radius_grad)


def test_fuzzy_entropy_vmap_with_nonfinite_sample_outside_every_template():
    torch.manual_seed(0)
    batch = torch.randn(3, 50, 1, dtype=torch.float64)
    batch[1, 25] = float("nan")
    direct = fuzzy_entropy(batch, 0.5, **GAP)
    mapped = torch.vmap(lambda series: fuzzy_entropy(series, 0.5, **GAP))(batch)
    assert torch.allclose(direct, mapped, equal_nan=True) and torch.isnan(mapped[1]) and torch.isfinite(mapped[0])


SMALL_SERIES = [0.00, 0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.06, -0.01, 0.00, 0.04, -0.03]
# Mean-removed templates of this series are at least 10 apart, so every membership at a radius of 0.1 underflows.
QUADRATIC_SERIES = [10.0 * index**2 for index in range(12)]
# At embedding_dim=3 some length-3 templates are close, but no length-4 pair is, in float32.
SPREAD_SERIES = [2.225, -6.662, 6.846, 5.74, -5.829, 7.679, 0.571, -11.894, -5.659, -8.327, 9.014, 2.116]
SPREAD_ROW_ZERO = [-0.049, -0.06, -0.056, 0.067, -0.02, 0.194, -0.14, -0.076, 0.063, -0.09, -0.056, 0.069]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "options, expected",
    [
        (dict(embedding_dim=1), math.inf),
        (dict(embedding_dim=2), math.nan),
        (dict(embedding_dim=3, norm_order=2.0), math.nan),
    ],
    ids=["long_underflows", "both_underflow", "both_underflow_euclidean"],
)
def test_fuzzy_entropy_underflowed_memberships_keep_other_gradients_finite(options, expected, dtype):
    # log(0) of an underflowed membership sum must not put NaN into the gradient of the shared radius.
    result = assert_first_row_gradients_unaffected(
        lambda series, radius: fuzzy_entropy(series, radius, **options), [SMALL_SERIES, QUADRATIC_SERIES], 0.1, dtype
    )
    assert torch.isfinite(result[0])
    torch.testing.assert_close(result[1], torch.tensor(expected, dtype=dtype), equal_nan=True)


def test_fuzzy_entropy_only_the_long_membership_sum_underflows_keeps_other_gradients_finite():
    result = assert_first_row_gradients_unaffected(
        lambda series, radius: fuzzy_entropy(series, radius, embedding_dim=3, norm_order=2.0),
        [SPREAD_ROW_ZERO, SPREAD_SERIES],
        0.1,
        torch.float32,
    )
    assert torch.isfinite(result[0]) and result[1] == math.inf
