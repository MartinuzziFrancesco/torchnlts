"""Tests for torchnlts.sciml.losses.

LyapunovLoss, CorrelationDimensionLoss and RecurrenceLoss are compared with direct calls of the primitives they wrap;
AttractorLoss with an independent MMD computation; SinkhornLoss with a brute-force regularized objective and with the
exact optimal transport cost (scipy's ``linear_sum_assignment``).
"""

import numpy as np
import pytest
import torch

from torchnlts.dimension.correlation import correlation_dimension, soft_correlation_sum
from torchnlts.lyapunov.spectrum import lyapunov_spectrum
from torchnlts.recurrence.matrix import soft_recurrence_matrix
from torchnlts.recurrence.rqa import recurrence_rate
from torchnlts.sciml.losses import (
    AttractorLoss,
    CorrelationDimensionLoss,
    LyapunovLoss,
    RecurrenceLoss,
    SinkhornLoss,
    _sinkhorn_cost,
    _squared_error,
)


def _henon_step(x, a=1.4, b=0.3):
    return torch.stack([1.0 - a * x[0] ** 2 + x[1], b * x[0]])


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


def test_lyapunov_loss_gradcheck_w_r_t_model_parameter():
    values = torch.tensor([1.5, 0.5], dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    target = torch.tensor([0.3, -0.5], dtype=torch.float64)
    loss_module = LyapunovLoss(target=target, num_steps=4)

    def f(values):
        return loss_module(lambda x: values * x, state)

    assert torch.autograd.gradcheck(f, (values,))


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
    b = torch.randn(40, 2, dtype=torch.float64) + 5.0
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
    # Regression: a non-Euclidean norm_order made the kernel non-PSD, so squared MMD went negative; the parameter was removed.
    with pytest.raises(TypeError):
        AttractorLoss(temperature=2.0, norm_order=1)


def test_sinkhorn_loss_zero_for_identical_trajectory():
    torch.manual_seed(12)
    trajectory = torch.randn(40, 2, dtype=torch.float64)
    loss_module = SinkhornLoss(reg=0.5)
    assert torch.isclose(loss_module(trajectory, trajectory), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_sinkhorn_loss_symmetric():
    torch.manual_seed(13)
    a = torch.randn(40, 2, dtype=torch.float64)
    b = torch.randn(35, 2, dtype=torch.float64) + 1.0
    loss_module = SinkhornLoss(reg=0.5)
    assert torch.isclose(loss_module(a, b), loss_module(b, a), atol=1e-10)


def test_sinkhorn_loss_positive_for_different_trajectories():
    torch.manual_seed(14)
    a = torch.randn(40, 2, dtype=torch.float64)
    b = torch.randn(40, 2, dtype=torch.float64) + 5.0
    loss_module = SinkhornLoss(reg=0.5)
    assert loss_module(a, b).item() > 0.1


def test_sinkhorn_loss_gradcheck():
    torch.manual_seed(15)
    a = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(8, 2, dtype=torch.float64)
    loss_module = SinkhornLoss(reg=1.0, num_iters=20)
    assert torch.autograd.gradcheck(loss_module, (a, b))


def test_sinkhorn_loss_batched_matches_independent_loop():
    torch.manual_seed(16)
    a = torch.randn(3, 20, 2, dtype=torch.float64)
    b = torch.randn(3, 20, 2, dtype=torch.float64)
    loss_module = SinkhornLoss(reg=0.5, reduction="none")
    batched = loss_module(a, b)
    assert batched.shape == (3,)
    for i in range(3):
        expected = loss_module(a[i], b[i])
        assert torch.isclose(batched[i], expected)


def test_sinkhorn_cost_converges_to_exact_optimal_transport():
    # Ground truth: for equal-size uniform clouds the unregularized OT plan is the minimum-cost matching (scipy Hungarian);
    # _sinkhorn_cost should converge to it as reg shrinks and iterations grow.
    from scipy.optimize import linear_sum_assignment

    from torchnlts.neighbors.distance import pairwise_distance

    torch.manual_seed(17)
    a = torch.randn(12, 2, dtype=torch.float64)
    b = torch.randn(12, 2, dtype=torch.float64) + 0.5

    cost = 0.5 * pairwise_distance(a, other=b, norm_order=2.0) ** 2
    row_idx, col_idx = linear_sum_assignment(cost.numpy())
    exact_cost = cost.numpy()[row_idx, col_idx].sum() / a.shape[0]

    entropic_cost = _sinkhorn_cost(cost, torch.tensor(0.01, dtype=torch.float64), 2000).item()
    assert abs(entropic_cost - exact_cost) < 0.05


def test_sinkhorn_cost_matches_brute_force_regularized_objective():
    # Regression: _sinkhorn_cost dropped the reg*KL(P | a(x)b) term; checked against the objective built from the explicit plan.
    torch.manual_seed(18)
    a = torch.randn(6, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64) + 0.4
    reg = 0.6
    num_iters = 3000
    cost = 0.5 * (a.unsqueeze(-1) - b.unsqueeze(0)) ** 2

    num_source, num_target = cost.shape
    log_source = -torch.log(torch.tensor(float(num_source), dtype=cost.dtype))
    log_target = -torch.log(torch.tensor(float(num_target), dtype=cost.dtype))
    f = torch.zeros(num_source, dtype=cost.dtype)
    g = torch.zeros(num_target, dtype=cost.dtype)
    for _ in range(num_iters):
        f = reg * (log_source - torch.logsumexp((g.unsqueeze(0) - cost) / reg, dim=1))
        g = reg * (log_target - torch.logsumexp((f.unsqueeze(1) - cost) / reg, dim=0))
    plan = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - cost) / reg)
    reference = torch.full((num_source,), 1 / num_source, dtype=cost.dtype).unsqueeze(1) * torch.full(
        (num_target,), 1 / num_target, dtype=cost.dtype
    ).unsqueeze(0)
    brute_force = (plan * cost).sum() + reg * (plan * torch.log(plan / reference)).sum()

    assert torch.isclose(
        _sinkhorn_cost(cost, torch.tensor(reg, dtype=torch.float64), num_iters), brute_force, atol=1e-6
    )


def test_sinkhorn_loss_rejects_non_positive_reg():
    with pytest.raises(ValueError):
        SinkhornLoss(reg=0.0)
    with pytest.raises(ValueError):
        SinkhornLoss(reg=-1.0)


def test_sinkhorn_loss_rejects_non_positive_num_iters():
    with pytest.raises(ValueError):
        SinkhornLoss(reg=0.5, num_iters=0)


_LOSS_FACTORIES = [
    lambda reduction: LyapunovLoss(target=0.0, num_steps=10, reduction=reduction),
    lambda reduction: CorrelationDimensionLoss(
        target=1.0, temperature=torch.linspace(0.1, 1.0, 5), reduction=reduction
    ),
    lambda reduction: RecurrenceLoss(target=0.2, temperature=0.5, reduction=reduction),
    lambda reduction: AttractorLoss(temperature=0.5, reduction=reduction),
    lambda reduction: SinkhornLoss(reg=0.5, reduction=reduction),
]


@pytest.mark.parametrize("make_loss", _LOSS_FACTORIES)
def test_losses_reject_bad_reduction_at_construction(make_loss):
    for bad in ("bogus", "", None):
        with pytest.raises(ValueError, match="reduction"):
            make_loss(bad)
    for reduction in ("mean", "sum", "none"):
        assert make_loss(reduction).reduction == reduction


def test_loss_forward_rejects_reduction_changed_after_construction():
    trajectory = torch.randn(10, 2, dtype=torch.float64)
    loss_module = SinkhornLoss(reg=0.5)
    loss_module.reduction = "invalid"
    with pytest.raises(ValueError, match="reduction"):
        loss_module(trajectory, trajectory)


@pytest.mark.parametrize("num_iters", [2.5, 3.0, "3", None])
def test_sinkhorn_loss_rejects_non_integer_num_iters(num_iters):
    with pytest.raises(ValueError, match="num_iters"):
        SinkhornLoss(reg=0.5, num_iters=num_iters)


def test_sinkhorn_loss_accepts_numpy_integer_num_iters():
    torch.manual_seed(19)
    a = torch.randn(10, 2, dtype=torch.float64)
    b = torch.randn(12, 2, dtype=torch.float64)
    loss_module = SinkhornLoss(reg=1.0, num_iters=np.int64(7))
    assert type(loss_module.num_iters) is int and loss_module.num_iters == 7
    assert torch.equal(loss_module(a, b), SinkhornLoss(reg=1.0, num_iters=7)(a, b))


@pytest.mark.parametrize("loss_module", [AttractorLoss(temperature=0.5), SinkhornLoss(reg=0.5)])
def test_trajectory_losses_promote_mixed_float_dtypes(loss_module):
    torch.manual_seed(20)
    wide_predicted = torch.randn(20, 2, dtype=torch.float64)
    wide_target = torch.randn(15, 2, dtype=torch.float64) + 0.5
    expected = loss_module(wide_predicted, wide_target)
    mixed = loss_module(wide_predicted.float(), wide_target)
    mixed_reversed = loss_module(wide_predicted, wide_target.float())
    assert mixed.dtype == mixed_reversed.dtype == torch.float64
    assert torch.allclose(mixed, expected, atol=1e-5) and torch.allclose(mixed_reversed, expected, atol=1e-5)
    assert loss_module(wide_predicted.half(), wide_target.half()).dtype == torch.float32


@pytest.mark.parametrize(
    "make_loss, num_args",
    [
        (lambda: CorrelationDimensionLoss(target=1.0, temperature=torch.tensor([1.5, 2.5, 3.5, 5.5])), 1),
        (lambda: RecurrenceLoss(target=0.2, temperature=0.7), 1),
        (lambda: SinkhornLoss(reg=0.5), 2),
    ],
)
def test_losses_compute_integer_trajectories_in_default_float_dtype(make_loss, num_args):
    # The bandwidth buffer must follow the computation dtype, not the integer input dtype (0.5 would become 0).
    generator = torch.Generator().manual_seed(21)
    integer_args = [torch.randint(-6, 7, (40, 2), generator=generator) for _ in range(num_args)]
    float_args = [value.to(torch.get_default_dtype()) for value in integer_args]
    loss_module = make_loss()
    result = loss_module(*integer_args)
    assert result.dtype == torch.get_default_dtype()
    assert torch.isfinite(result) and torch.equal(result, loss_module(*float_args))


def test_loss_modules_preserve_float64_target_and_temperature_precision():
    # Regression: register_buffer forced float32; [1.0, 1.0 + 1e-8] only survives in float64.
    close_values = torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.float64)

    cdl = CorrelationDimensionLoss(target=2.0, temperature=close_values)
    assert cdl.temperature.dtype == torch.float64
    assert not torch.equal(cdl.temperature, torch.full_like(cdl.temperature, 1.0))

    rl = RecurrenceLoss(target=0.5, temperature=close_values[0])
    assert rl.temperature.dtype == torch.float64

    al = AttractorLoss(temperature=close_values[0])
    assert al.temperature.dtype == torch.float64

    sl = SinkhornLoss(reg=close_values[0])
    assert sl.reg.dtype == torch.float64

    lyapunov_target = torch.tensor([0.1, -0.1 + 1e-9], dtype=torch.float64)
    ll = LyapunovLoss(target=lyapunov_target, num_steps=5)
    assert ll.target.dtype == torch.float64
    assert not torch.equal(ll.target, torch.tensor([0.1, -0.1], dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1j, torch.tensor([0.5, float("nan")])])
def test_sinkhorn_loss_rejects_non_finite_or_complex_reg(bad):
    with pytest.raises(ValueError, match="reg must be a finite real number > 0"):
        SinkhornLoss(reg=bad)


def _overwrite_with(values: torch.Tensor, row: int, point: int, bad: float) -> torch.Tensor:
    """``values`` with one entry overwritten by ``bad``, so the entry adds nothing to a shared parameter's graph."""
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[row, point, 0] = True
    return values.masked_scatter(mask, torch.full((1,), bad, dtype=values.dtype))


_ROW_PREDICTED = torch.tensor(
    [
        [[0.0], [1.0], [2.5], [3.0], [4.5], [5.0]],
        [[0.0], [1.5], [3.0], [4.5], [4.0], [6.0]],
        [[0.5], [2.0], [3.0], [5.0], [7.0], [6.5]],
    ],
    dtype=torch.float64,
)
_ROW_TARGET = _ROW_PREDICTED.flip(1) + 0.3
_ROW_LOSSES = {
    "attractor": (lambda reduction: AttractorLoss(0.7, reduction=reduction), 2),
    "sinkhorn": (lambda reduction: SinkhornLoss(0.5, reduction=reduction), 2),
    "recurrence": (lambda reduction: RecurrenceLoss(0.2, 0.7, reduction=reduction), 1),
    "correlation_dimension": (
        lambda reduction: CorrelationDimensionLoss(
            1.0, torch.linspace(0.5, 4.0, 6, dtype=torch.float64), reduction=reduction
        ),
        1,
    ),
}


def _run_row_loss(name, reduction, predicted, target):
    make_loss, num_args = _ROW_LOSSES[name]
    loss_module = make_loss(reduction)
    return loss_module(predicted, target) if num_args == 2 else loss_module(predicted)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "name, which",
    [
        ("attractor", "predicted"),
        ("attractor", "target"),
        ("sinkhorn", "predicted"),
        ("sinkhorn", "target"),
        ("recurrence", "predicted"),
        ("correlation_dimension", "predicted"),
    ],
)
def test_losses_give_nan_for_nonfinite_rows_and_keep_gradients_finite(name, which, bad):
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    predicted, target = parameter * _ROW_PREDICTED, _ROW_TARGET.clone()
    if which == "predicted":
        predicted = _overwrite_with(predicted, row=1, point=2, bad=bad)
    else:
        target[1, 2, 0] = bad
    loss = _run_row_loss(name, "none", predicted, target)
    assert torch.isnan(loss).tolist() == [False, True, False]
    assert torch.isnan(_run_row_loss(name, "mean", predicted, target))
    assert torch.isnan(_run_row_loss(name, "sum", predicted, target))

    valid = [0, 2]
    expected_loss = _run_row_loss(name, "none", parameter * _ROW_PREDICTED[valid], target[valid])
    torch.testing.assert_close(loss[valid], expected_loss)
    (gradient,) = torch.autograd.grad(loss[valid].sum(), parameter)
    (expected,) = torch.autograd.grad(expected_loss.sum(), parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("name", list(_ROW_LOSSES))
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_losses_give_nan_for_a_nonfinite_unbatched_trajectory(name, bad):
    predicted, target = _ROW_PREDICTED[0, :, 0], _ROW_TARGET[0, :, 0]
    spoiled = predicted.clone()
    spoiled[2] = bad
    assert torch.isfinite(_run_row_loss(name, "mean", predicted, target))
    assert _run_row_loss(name, "mean", spoiled, target).isnan()
    if _ROW_LOSSES[name][1] == 2:
        assert _run_row_loss(name, "mean", predicted, spoiled).isnan()


def test_lyapunov_loss_nonfinite_initial_state_rows_are_nan_and_keep_gradients_finite():
    nan, inf = float("nan"), float("inf")
    states = torch.tensor([[0.1, 0.2], [nan, 0.2], [0.3, 0.1], [inf, 0.0]], dtype=torch.float64)
    parameter = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)

    def step(state):
        return torch.stack([1.0 - parameter * state[0] ** 2 + state[1], 0.3 * state[0]])

    loss = LyapunovLoss(target=0.4, num_steps=10, reduction="none")(step, states)
    assert torch.isnan(loss).all(dim=-1).tolist() == [False, True, False, True]
    assert torch.isfinite(loss[[0, 2]]).all()
    assert torch.isnan(LyapunovLoss(target=0.4, num_steps=10)(step, states))

    expected_loss = LyapunovLoss(target=0.4, num_steps=10, reduction="none")(step, states[[0, 2]])
    torch.testing.assert_close(loss[[0, 2]], expected_loss)
    (gradient,) = torch.autograd.grad(loss[[0, 2]].sum(), parameter)
    (expected,) = torch.autograd.grad(expected_loss.sum(), parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize(
    "bad",
    [None, "0.5", b"0.5", [], [0.1, 0.2], [0.5], torch.tensor([]), torch.tensor([0.1, 0.2]), object(), 10**30],
)
def test_sinkhorn_loss_rejects_reg_that_is_not_a_single_real_number(bad):
    with pytest.raises(ValueError, match="reg must be a finite real number > 0"):
        SinkhornLoss(reg=bad)


@pytest.mark.parametrize(
    "good", [0.5, 2, np.float32(0.5), np.int64(3), torch.tensor(0.5), torch.tensor([0.5]), torch.tensor([[2.0]])]
)
def test_sinkhorn_loss_accepts_a_single_positive_real_number_as_reg(good):
    loss_module = SinkhornLoss(reg=good)
    assert loss_module.reg.shape == () and float(loss_module.reg) == float(torch.as_tensor(good))
    trajectory = torch.randn(8, 2, dtype=torch.float64)
    assert torch.isfinite(loss_module(trajectory, trajectory + 0.5))


def test_squared_error_keeps_nonfinite_statistics_out_of_backward():
    target = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    statistic = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    error = _squared_error(statistic, target)
    assert error[0] == 0.25 and torch.isnan(error[1]) and (error[2:] == float("inf")).all()
    (gradient,) = torch.autograd.grad(error[0], target)
    assert gradient == -1.0


def _expected_loss_for(bad_target: float) -> torch.Tensor:
    """The loss of an entry whose target is ``bad_target``: NaN for NaN, inf for either infinity."""
    return torch.tensor(float("nan") if np.isnan(bad_target) else float("inf"), dtype=torch.float64)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_squared_error_keeps_nonfinite_targets_out_of_backward(bad):
    statistic = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.5, bad, 0.25], dtype=torch.float64, requires_grad=True)
    error = _squared_error(statistic, target)
    assert error[0] == 0.25 and error[2] == 7.5625
    torch.testing.assert_close(error[1], _expected_loss_for(bad), equal_nan=True)
    statistic_gradient, target_gradient = torch.autograd.grad(error[[0, 2]].sum(), (statistic, target))
    assert statistic_gradient.tolist() == [1.0, 0.0, 5.5]
    assert target_gradient.tolist() == [-1.0, 0.0, -5.5]


_TARGET_LOSSES = {
    "recurrence": lambda target: RecurrenceLoss(target, 0.7, reduction="none"),
    "correlation_dimension": lambda target: CorrelationDimensionLoss(
        target, torch.linspace(0.5, 4.0, 6, dtype=torch.float64), reduction="none"
    ),
}


def _target_gradient(name, trajectory, target_value, rows):
    """Gradient w.r.t. the target of the summed loss of ``rows`` of ``trajectory``, and the loss of every row."""
    target = torch.tensor(target_value, dtype=torch.float64, requires_grad=True)
    loss = _TARGET_LOSSES[name](target)(trajectory)
    return torch.autograd.grad(loss[rows].sum(), target)[0], loss


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("name", list(_TARGET_LOSSES))
def test_trajectory_statistic_losses_nonfinite_row_keeps_shared_target_gradient_finite(name, bad):
    spoiled = _overwrite_with(_ROW_PREDICTED, row=1, point=2, bad=bad)
    gradient, loss = _target_gradient(name, spoiled, 0.5, [0, 2])
    assert torch.isnan(loss).tolist() == [False, True, False]
    expected, _ = _target_gradient(name, _ROW_PREDICTED[[0, 2]], 0.5, [0, 1])
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


def test_trajectory_statistic_losses_per_row_target_gets_no_gradient_from_a_nonfinite_row():
    spoiled = _overwrite_with(_ROW_PREDICTED, row=1, point=2, bad=float("nan"))
    target = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float64, requires_grad=True)
    loss = RecurrenceLoss(target, 0.7, reduction="none")(spoiled)
    (gradient,) = torch.autograd.grad(loss[[0, 2]].sum(), target)
    assert torch.isfinite(gradient).all() and gradient[1] == 0 and (gradient[[0, 2]] != 0).all()


_PER_ROW_TARGET_LOSSES = {
    "recurrence": (
        lambda target, temperature: RecurrenceLoss(target, temperature, reduction="none"),
        lambda: torch.tensor(0.7, dtype=torch.float64, requires_grad=True),
        [0.5, 0.2, 0.3],
    ),
    "correlation_dimension": (
        lambda target, temperature: CorrelationDimensionLoss(target, temperature, reduction="none"),
        lambda: torch.linspace(0.5, 4.0, 6, dtype=torch.float64, requires_grad=True),
        [1.0, 1.1, 1.2],
    ),
}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("name", list(_PER_ROW_TARGET_LOSSES))
def test_trajectory_statistic_losses_nonfinite_per_row_target_keeps_shared_gradients_finite(name, bad):
    make_loss, make_temperature, target_values = _PER_ROW_TARGET_LOSSES[name]

    def loss_and_gradients(rows, values):
        parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        temperature = make_temperature()
        target = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = make_loss(target, temperature)(parameter * _ROW_PREDICTED[rows])
        return loss, torch.autograd.grad(loss[[0, -1]].sum(), (parameter, temperature, target))

    loss, gradients = loss_and_gradients([0, 1, 2], [target_values[0], bad, target_values[2]])
    expected_loss, expected = loss_and_gradients([0, 2], [target_values[0], target_values[2]])
    torch.testing.assert_close(loss[1], _expected_loss_for(bad), equal_nan=True)
    torch.testing.assert_close(loss[[0, 2]], expected_loss)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[2][1] == 0 and (gradients[2][[0, 2]] != 0).all()
    torch.testing.assert_close(gradients[0], expected[0])
    torch.testing.assert_close(gradients[1], expected[1])
    torch.testing.assert_close(gradients[2][[0, 2]], expected[2])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_lyapunov_loss_nonfinite_target_entry_keeps_shared_gradients_finite(bad):
    states = torch.tensor([[1.0, 1.0], [0.5, 2.0]], dtype=torch.float64)

    def loss_and_gradients(values):
        weights = torch.tensor([1.5, 0.5], dtype=torch.float64, requires_grad=True)
        target = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = LyapunovLoss(target, 4, reduction="none")(lambda state: weights * state, states)
        return loss, torch.autograd.grad(loss[:, 0].sum(), (weights, target))

    loss, gradients = loss_and_gradients([0.3, bad])
    expected_loss, expected = loss_and_gradients([0.3, -0.5])
    torch.testing.assert_close(loss[:, 1], _expected_loss_for(bad).expand(2), equal_nan=True)
    torch.testing.assert_close(loss[:, 0], expected_loss[:, 0])
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0][0] != 0 and gradients[1].tolist() == [expected[1][0].item(), 0.0]
    torch.testing.assert_close(gradients[0], expected[0])


def test_correlation_dimension_loss_degenerate_row_keeps_shared_target_gradient_finite():
    # Row 1 is finite but so spread out that every kernel weight underflows: C(r) = 0 and its dimension is NaN.
    spread = _ROW_PREDICTED.clone()
    spread[1] = spread[1] * 1e4
    gradient, loss = _target_gradient("correlation_dimension", spread, 1.0, [0, 2])
    assert torch.isnan(loss).tolist() == [False, True, False]
    expected, _ = _target_gradient("correlation_dimension", _ROW_PREDICTED[[0, 2]], 1.0, [0, 1])
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)


def test_lyapunov_loss_nonfinite_initial_state_keeps_shared_target_gradient_finite():
    nan = float("nan")
    states = torch.tensor([[0.1, 0.2], [nan, 0.2], [0.3, 0.1]], dtype=torch.float64)

    def target_gradient(states, rows):
        target = torch.tensor([0.4, -1.0], dtype=torch.float64, requires_grad=True)
        loss = LyapunovLoss(target, 10, reduction="none")(_henon_step, states)
        return torch.autograd.grad(loss[rows].sum(), target)[0], loss

    gradient, loss = target_gradient(states, [0, 2])
    assert torch.isnan(loss).all(dim=-1).tolist() == [False, True, False]
    expected, _ = target_gradient(states[[0, 2]], [0, 1])
    assert torch.isfinite(gradient).all() and (gradient != 0).all()
    torch.testing.assert_close(gradient, expected)


def test_lyapunov_loss_all_nonfinite_states_keep_shared_gradients_finite():
    parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    states = torch.tensor([[float("nan")], [float("inf")]], dtype=torch.float64)
    loss = LyapunovLoss(target, 3, reduction="none")(lambda state: parameter * state.square(), states)
    assert torch.isnan(loss).all()
    gradients = torch.autograd.grad(parameter + target + torch.nan_to_num(loss).sum(), [parameter, target])
    assert [gradient.item() for gradient in gradients] == [1.0, 1.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_lyapunov_loss_unbatched_nonfinite_initial_state_is_nan(bad):
    # A linear map has a constant Jacobian, so only the masking of the state can make the loss NaN.
    loss_module = LyapunovLoss(0.4, 10, reduction="none")

    def step(state):
        return 2 * state

    loss = loss_module(step, torch.tensor([0.1, bad], dtype=torch.float64))
    assert loss.shape == (2,) and torch.isnan(loss).all()
    assert torch.isfinite(loss_module(step, torch.tensor([0.1, 0.2], dtype=torch.float64))).all()


@pytest.mark.parametrize("reduction", ["none", "sum"])
def test_lyapunov_loss_empty_batch(reduction):
    loss = LyapunovLoss(0.5, 3, reduction=reduction)(lambda state: 2 * state, torch.empty(0, 2))
    assert loss.shape == ((0, 2) if reduction == "none" else ()) and (loss == 0).all()


_HALF_INPUT = [0.0, 0.25, 0.75, 1.5]  # exact in float16 and bfloat16
_HALF_LOSSES = {
    "correlation_dimension": (lambda: CorrelationDimensionLoss(1.0, torch.tensor([0.7001, 0.7002])), 1, True),
    "recurrence": (lambda: RecurrenceLoss(0.2, 0.7001), 1, True),
    "attractor": (lambda: AttractorLoss(0.7001), 2, False),
    "sinkhorn": (lambda: SinkhornLoss(0.7001), 2, False),
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", list(_HALF_LOSSES))
def test_losses_cast_bandwidths_to_the_work_dtype_not_the_half_input_dtype(name, dtype):
    # 0.7001 and 0.7002 collapse in half precision; the bandwidths have to follow the float32 computation.
    make_loss, num_args, casts_back = _HALF_LOSSES[name]
    loss_module = make_loss()
    half = torch.tensor(_HALF_INPUT, dtype=dtype)
    args = (half,) if num_args == 1 else (half, half + 0.25)
    result = loss_module(*args)
    expected = loss_module(*(value.float() for value in args))
    assert torch.isfinite(result)
    if casts_back:
        assert result.dtype == dtype and torch.equal(result, expected.to(dtype))
    else:
        assert result.dtype == torch.float32 and torch.equal(result, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("which", ["predicted", "target"])
@pytest.mark.parametrize("name", ["attractor", "sinkhorn"])
def test_trajectory_losses_nonfinite_row_keeps_gradients_of_the_target_parameter_finite(name, which, bad):
    # The parameter produces the target trajectory; row 1 of either trajectory is non-finite.
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    predicted, target = _ROW_PREDICTED.clone(), parameter * _ROW_TARGET
    if which == "predicted":
        predicted[1, 2, 0] = bad
    else:
        target = _overwrite_with(target, row=1, point=2, bad=bad)
    loss = _run_row_loss(name, "none", predicted, target)
    assert torch.isnan(loss).tolist() == [False, True, False]
    valid = [0, 2]
    (gradient,) = torch.autograd.grad(loss[valid].sum(), parameter)
    expected_parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    expected_loss = _run_row_loss(name, "none", _ROW_PREDICTED[valid], (expected_parameter * _ROW_TARGET)[valid])
    (expected,) = torch.autograd.grad(expected_loss.sum(), expected_parameter)
    assert torch.isfinite(gradient) and gradient != 0
    torch.testing.assert_close(gradient, expected)
