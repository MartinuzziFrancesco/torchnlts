"""Tests for torchnlts.sciml.losses.

Note: each loss here composes
primitives already implemented, tested, and dual-validated (paper +
Julia) elsewhere in the package -- none introduces a new statistic, so
none needs its own fresh paper/Julia ground truth. What these tests
validate instead is that each loss *correctly wires up* the primitive
it reuses: hand-derived by computing the same primitive directly
(outside the loss module) and comparing, exactly mirroring how
``kantz_divergence`` reusing ``rosenstein_lyapunov`` didn't need its own
independent numerical ground truth either. ``AttractorLoss`` is the
exception -- its MMD combination is genuinely new logic, not just a
single reused call, and a review found (by direct
counterexample) that it could go negative for a non-Euclidean
``norm_order``, and that constructing a loss with a float64 target/
temperature silently downcast it to the module's default dtype. Both
are now locked in as regressions below, not just fixed silently.
"""

import torch

from torchnlts.dimension.correlation import correlation_dimension, soft_correlation_sum
from torchnlts.lyapunov.spectrum import lyapunov_spectrum
from torchnlts.recurrence.matrix import soft_recurrence_matrix
from torchnlts.recurrence.rqa import recurrence_rate
from torchnlts.sciml.losses import AttractorLoss, CorrelationDimensionLoss, LyapunovLoss, RecurrenceLoss


def _henon_step(x, a=1.4, b=0.3):
    return torch.stack([1.0 - a * x[0] ** 2 + x[1], b * x[0]])


# ---------------------------------------------------------------- Lyapunov


def test_lyapunov_loss_matches_direct_computation():
    state = torch.zeros(2, dtype=torch.float64)
    target = torch.tensor([0.3, -1.5], dtype=torch.float64)
    loss_module = LyapunovLoss(target=target, num_steps=50)
    predicted = lyapunov_spectrum(_henon_step, state, num_steps=50)
    expected = ((predicted - target) ** 2).mean()
    assert torch.allclose(loss_module(_henon_step, state), expected)


def test_lyapunov_loss_zero_at_exact_target():
    state = torch.zeros(2, dtype=torch.float64)
    target = lyapunov_spectrum(_henon_step, state, num_steps=50)
    loss_module = LyapunovLoss(target=target, num_steps=50)
    assert torch.isclose(loss_module(_henon_step, state), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_lyapunov_loss_reduction_modes():
    state = torch.zeros(2, dtype=torch.float64)
    target = torch.tensor([0.0, 0.0], dtype=torch.float64)
    predicted = lyapunov_spectrum(_henon_step, state, num_steps=20)
    expected_none = (predicted - target) ** 2

    none_loss = LyapunovLoss(target=target, num_steps=20, reduction="none")(_henon_step, state)
    sum_loss = LyapunovLoss(target=target, num_steps=20, reduction="sum")(_henon_step, state)
    mean_loss = LyapunovLoss(target=target, num_steps=20, reduction="mean")(_henon_step, state)

    assert torch.allclose(none_loss, expected_none)
    assert torch.isclose(sum_loss, expected_none.sum())
    assert torch.isclose(mean_loss, expected_none.mean())


def test_lyapunov_loss_rejects_bad_reduction():
    state = torch.zeros(2, dtype=torch.float64)
    loss_module = LyapunovLoss(target=0.0, num_steps=10, reduction="bogus")
    try:
        loss_module(_henon_step, state)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lyapunov_loss_gradcheck_w_r_t_model_parameter():
    values = torch.tensor([1.5, 0.5], dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    target = torch.tensor([0.3, -0.5], dtype=torch.float64)
    loss_module = LyapunovLoss(target=target, num_steps=4)

    def f(values):
        return loss_module(lambda x: values * x, state)

    assert torch.autograd.gradcheck(f, (values,))


# ------------------------------------------------------- CorrelationDimension


def test_correlation_dimension_loss_matches_direct_computation():
    torch.manual_seed(0)
    trajectory = torch.randn(100, 2, dtype=torch.float64).cumsum(0) * 0.1
    temperature = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    loss_module = CorrelationDimensionLoss(target=1.5, temperature=temperature)

    sums = soft_correlation_sum(trajectory, temperature)
    dimension = correlation_dimension(temperature, sums)
    expected = (dimension - 1.5) ** 2
    assert torch.allclose(loss_module(trajectory), expected)


def test_correlation_dimension_loss_zero_at_exact_target():
    torch.manual_seed(1)
    trajectory = torch.randn(100, 2, dtype=torch.float64).cumsum(0) * 0.1
    temperature = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    dimension = correlation_dimension(temperature, soft_correlation_sum(trajectory, temperature))
    loss_module = CorrelationDimensionLoss(target=dimension, temperature=temperature)
    assert torch.isclose(loss_module(trajectory), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_correlation_dimension_loss_gradcheck():
    torch.manual_seed(2)
    trajectory = torch.randn(30, 2, dtype=torch.float64, requires_grad=True)
    temperature = torch.linspace(0.5, 2.0, 5, dtype=torch.float64)
    loss_module = CorrelationDimensionLoss(target=1.0, temperature=temperature)
    assert torch.autograd.gradcheck(loss_module, (trajectory,))


# ------------------------------------------------------------- Recurrence


def test_recurrence_loss_matches_direct_computation():
    torch.manual_seed(3)
    trajectory = torch.randn(50, 2, dtype=torch.float64)
    loss_module = RecurrenceLoss(target=0.2, temperature=0.5)

    weights = soft_recurrence_matrix(trajectory, 0.5)
    rate = recurrence_rate(weights)
    expected = (rate - 0.2) ** 2
    assert torch.allclose(loss_module(trajectory), expected)


def test_recurrence_loss_zero_at_exact_target():
    torch.manual_seed(4)
    trajectory = torch.randn(50, 2, dtype=torch.float64)
    rate = recurrence_rate(soft_recurrence_matrix(trajectory, 0.5))
    loss_module = RecurrenceLoss(target=rate, temperature=0.5)
    assert torch.isclose(loss_module(trajectory), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_recurrence_loss_gradcheck():
    torch.manual_seed(5)
    trajectory = torch.randn(20, 2, dtype=torch.float64, requires_grad=True)
    loss_module = RecurrenceLoss(target=0.3, temperature=1.0)
    assert torch.autograd.gradcheck(loss_module, (trajectory,))


# --------------------------------------------------------------- Attractor


def test_attractor_loss_zero_for_identical_trajectory():
    torch.manual_seed(6)
    trajectory = torch.randn(40, 2, dtype=torch.float64)
    loss_module = AttractorLoss(temperature=0.5)
    assert torch.isclose(loss_module(trajectory, trajectory), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_attractor_loss_symmetric():
    torch.manual_seed(7)
    a = torch.randn(40, 2, dtype=torch.float64)
    b = torch.randn(35, 2, dtype=torch.float64) + 1.0
    loss_module = AttractorLoss(temperature=0.5)
    assert torch.isclose(loss_module(a, b), loss_module(b, a), atol=1e-10)


def test_attractor_loss_positive_for_different_trajectories():
    torch.manual_seed(8)
    a = torch.randn(40, 2, dtype=torch.float64)
    b = torch.randn(40, 2, dtype=torch.float64) + 5.0  # well-separated clusters
    loss_module = AttractorLoss(temperature=0.5)
    assert loss_module(a, b).item() > 0.1


def test_attractor_loss_matches_direct_computation():
    torch.manual_seed(9)
    a = torch.randn(30, 2, dtype=torch.float64)
    b = torch.randn(25, 2, dtype=torch.float64)
    from torchnlts.neighbors.distance import pairwise_distance
    from torchnlts.neighbors.kernels import trajectory_kernel

    pp = trajectory_kernel(pairwise_distance(a), 0.5).mean()
    qq = trajectory_kernel(pairwise_distance(b), 0.5).mean()
    pq = trajectory_kernel(pairwise_distance(a, other=b), 0.5).mean()
    expected = pp + qq - 2 * pq

    loss_module = AttractorLoss(temperature=0.5)
    assert torch.allclose(loss_module(a, b), expected)


def test_attractor_loss_gradcheck():
    torch.manual_seed(10)
    a = torch.randn(15, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(15, 2, dtype=torch.float64)
    loss_module = AttractorLoss(temperature=1.0)
    assert torch.autograd.gradcheck(loss_module, (a, b))


def test_attractor_loss_batched_matches_independent_loop():
    torch.manual_seed(11)
    a = torch.randn(3, 20, 2, dtype=torch.float64)
    b = torch.randn(3, 20, 2, dtype=torch.float64)
    loss_module = AttractorLoss(temperature=0.5, reduction="none")
    batched = loss_module(a, b)
    assert batched.shape == (3,)
    for i in range(3):
        expected = loss_module(a[i], b[i])
        assert torch.isclose(batched[i], expected)


def test_attractor_loss_rejects_norm_order():
    # Regression: a non-Euclidean norm_order can make the
    # Gaussian/Laplacian kernel non-positive-semidefinite, letting the
    # "squared" MMD go negative (reproduced with these exact points and
    # norm_order=1 before the parameter was removed). No longer exposed
    # as a parameter at all, rather than validated-and-rejected, since
    # Euclidean is the only value with the MMD non-negativity guarantee.
    try:
        AttractorLoss(temperature=2.0, norm_order=1)
        assert False, "expected TypeError"
    except TypeError:
        pass


# --------------------------------------------------------- dtype regressions


def test_loss_modules_preserve_float64_target_and_temperature_precision():
    # Regression: register_buffer used to force
    # torch.get_default_dtype() (float32) regardless of the dtype
    # actually passed in, silently losing precision no later .to() call
    # could recover. [1.0, 1.0 + 1e-8] only survives round-tripping in
    # float64 -- float32 collapses it to [1.0, 1.0].
    close_values = torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.float64)

    cdl = CorrelationDimensionLoss(target=2.0, temperature=close_values)
    assert cdl.temperature.dtype == torch.float64
    assert not torch.equal(cdl.temperature, torch.full_like(cdl.temperature, 1.0))

    rl = RecurrenceLoss(target=0.5, temperature=close_values[0])
    assert rl.temperature.dtype == torch.float64

    al = AttractorLoss(temperature=close_values[0])
    assert al.temperature.dtype == torch.float64

    lyapunov_target = torch.tensor([0.1, -0.1 + 1e-9], dtype=torch.float64)
    ll = LyapunovLoss(target=lyapunov_target, num_steps=5)
    assert ll.target.dtype == torch.float64
    assert not torch.equal(ll.target, torch.tensor([0.1, -0.1], dtype=torch.float64))
