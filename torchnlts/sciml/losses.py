"""Invariant-matching training losses for differentiable dynamics models.

Training on pointwise MSE alone can fit short trajectories well while the long-term statistics (Lyapunov exponents,
fractal dimension, recurrence structure, attractor shape) come out wrong. These losses penalize that directly and are
added to an MSE term. ``LyapunovLoss``, ``CorrelationDimensionLoss`` and ``RecurrenceLoss`` match a statistic to a
target value (the last two use the soft, differentiable variants, since the hard ones have no gradient);
``AttractorLoss`` (maximum mean discrepancy) and ``SinkhornLoss`` (optimal transport) compare the state-space point
clouds of two trajectories. Each loss is a ``torch.nn.Module`` whose configuration is set at construction;
``reduction`` is ``"mean"``, ``"sum"`` or ``"none"``, as in ``torch.nn.MSELoss``.

A batch row whose predicted or target trajectory contains NaN or inf has a ``NaN`` loss, so the ``"mean"`` and ``"sum"``
reductions are ``NaN`` too. In the three statistic-matching losses a NaN entry in ``target`` gives that entry a NaN loss and an
inf entry an inf loss. An entry with a non-finite loss, whatever the cause, adds nothing to the gradients of the other
entries, of ``target`` or of shared parameters.
"""

from __future__ import annotations

import numbers

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _is_complex,
    _mask_nonfinite_rows,
    _output_dtype,
    _restore_batch,
    _validate_matching_batch,
    _work_dtype,
)
from torchnlts.convert import as_tensor
from torchnlts.dimension.correlation import correlation_dimension, soft_correlation_sum
from torchnlts.lyapunov.spectrum import lyapunov_spectrum
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.recurrence.matrix import soft_recurrence_matrix
from torchnlts.recurrence.rqa import recurrence_rate

_REDUCTIONS = ("mean", "sum", "none")


def _check_reduction(reduction: str) -> str:
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    return reduction


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    reduction = _check_reduction(reduction)
    if reduction == "mean":
        return loss.mean()
    return loss.sum() if reduction == "sum" else loss


def _clean_rows(trajectory) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``[B, T, D]`` trajectory in the work dtype with every row containing NaN/inf zeroed (so none reaches backward),
    the ``[B]`` mask of those rows and the input's ndim."""
    batched, ndim = _ensure_batched(trajectory)
    cleaned, nonfinite = _mask_nonfinite_rows(batched.to(_work_dtype(batched.dtype)))
    return cleaned, nonfinite, ndim


def _nan_rows(loss: torch.Tensor, nonfinite: torch.Tensor, ndim: int) -> torch.Tensor:
    """``loss`` ``[B]`` with ``NaN`` at the ``nonfinite`` rows, in the shape of an input with ``ndim`` dimensions."""
    return _restore_batch(torch.where(nonfinite, torch.full_like(loss, float("nan")), loss), ndim)


def _squared_error(statistic: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(statistic - target) ** 2``, where an entry with a NaN/inf statistic or target keeps that loss value but is
    kept out of backward: its ``0 * NaN`` would otherwise reach the other operand, ``target`` or shared parameters."""
    finite = torch.isfinite(statistic) & torch.isfinite(target)
    safe_statistic = torch.where(finite, statistic, torch.zeros_like(statistic))
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    return torch.where(finite, (safe_statistic - safe_target) ** 2, (statistic.detach() - target.detach()) ** 2)


def _positive_scalar(value, name: str) -> torch.Tensor:
    """``value`` (a Python or numpy real number, or a one-element tensor) as a 0-d tensor of its own dtype;
    ``ValueError`` unless it is finite and ``> 0``."""
    message = f"{name} must be a finite real number > 0, got {value!r}"
    if not isinstance(value, (numbers.Real, torch.Tensor)) or _is_complex(value):
        raise ValueError(message)
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError(message) from None
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all() & (tensor > 0).all()):
        raise ValueError(message)
    return tensor.reshape(())


def _sinkhorn_cost(cost: torch.Tensor, reg: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Entropy-regularized optimal transport objective ``<P, C> + reg * KL(P | a (x) b)`` at the optimal plan ``P``
    between two uniform empirical measures (Feydy et al. 2019, eq. 1).

    The dual potentials come from ``num_iters`` log-domain Sinkhorn iterations (Peyre & Cuturi 2019, sec. 4.4),
    unrolled and differentiated through directly; the objective is read off them by strong duality instead of
    forming ``P``.

    Args:
        cost: Shape ``[..., num_source, num_target]``.
        reg: Entropic regularization strength, a scalar.
        num_iters: Number of Sinkhorn iterations.

    Returns:
        Shape ``[...]``.
    """
    num_source, num_target = cost.shape[-2], cost.shape[-1]
    log_num_source = torch.log(torch.tensor(float(num_source), dtype=cost.dtype, device=cost.device))
    log_num_target = torch.log(torch.tensor(float(num_target), dtype=cost.dtype, device=cost.device))
    source_potential = torch.zeros(cost.shape[:-1], dtype=cost.dtype, device=cost.device)
    target_potential = torch.zeros(cost.shape[:-2] + cost.shape[-1:], dtype=cost.dtype, device=cost.device)
    for _ in range(num_iters):
        source_potential = reg * (
            -log_num_source - torch.logsumexp((target_potential.unsqueeze(-2) - cost) / reg, dim=-1)
        )
        target_potential = reg * (
            -log_num_target - torch.logsumexp((source_potential.unsqueeze(-1) - cost) / reg, dim=-2)
        )
    return source_potential.mean(dim=-1) + target_potential.mean(dim=-1) + reg * (log_num_source + log_num_target)


class LyapunovLoss(torch.nn.Module):
    """Squared error between a differentiable model's rollout Lyapunov spectrum and a target spectrum.

    Gradient class: exact. The gradient runs through ``num_steps`` chained step, Jacobian and QR evaluations and can
    have exploding variance for chaotic ``step_fn`` (Metz et al. 2021, "Gradients are Not All You Need"); keep
    ``num_steps`` modest, clip gradients, or detach the state in an outer rollout.

    Args:
        target: Target Lyapunov exponent(s): a scalar, or shape ``[num_exponents]`` for the full or a partial
            spectrum. Broadcasts against the output of :func:`~torchnlts.lyapunov.lyapunov_spectrum`.
        num_steps: Forwarded to ``lyapunov_spectrum``.
        num_exponents: Forwarded to ``lyapunov_spectrum``.
        dt: Forwarded to ``lyapunov_spectrum``.
        transient_steps: Forwarded to ``lyapunov_spectrum``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``reduction`` is not one of the three.
    """

    def __init__(
        self,
        target,
        num_steps: int,
        *,
        num_exponents: int | None = None,
        dt: float = 1.0,
        transient_steps: int = 0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.num_steps = num_steps
        self.num_exponents = num_exponents
        self.dt = dt
        self.transient_steps = transient_steps
        self.reduction = _check_reduction(reduction)

    def forward(self, step_fn, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_fn: The differentiable dynamics model, one discrete step; forwarded to ``lyapunov_spectrum``.
            initial_state: Forwarded to ``lyapunov_spectrum``.

        Returns:
            Scalar loss, or with ``reduction="none"`` the squared error of shape ``[num_exponents]`` for an unbatched
            ``initial_state`` or ``[B, num_exponents]`` for a batched one; ``NaN`` for an ``initial_state`` (or batch
            row) containing NaN or inf.

        Raises:
            ValueError: As :func:`~torchnlts.lyapunov.lyapunov_spectrum`, e.g. if ``num_steps``, ``num_exponents`` or
                ``transient_steps`` is not an integer in its range.
        """
        predicted = lyapunov_spectrum(
            step_fn,
            initial_state,
            self.num_steps,
            num_exponents=self.num_exponents,
            dt=self.dt,
            transient_steps=self.transient_steps,
        )
        return _reduce(_squared_error(predicted, self.target.to(predicted.dtype)), self.reduction)


class CorrelationDimensionLoss(torch.nn.Module):
    """Squared error between a predicted trajectory's correlation dimension and a target dimension.

    Gradient class: smooth (inherits :func:`~torchnlts.dimension.soft_correlation_sum`'s; the least-squares slope is
    exact).

    Args:
        target: Target correlation dimension: a scalar, or ``[B]`` to match a batched trajectory.
        temperature: Kernel-bandwidth sweep of shape ``[R]``, ``R >= 2``; forwarded to ``soft_correlation_sum`` and,
            as ``radius``, to ``correlation_dimension``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_correlation_sum``.
        theiler_window: Forwarded to ``soft_correlation_sum``.
        norm_order: Forwarded to ``soft_correlation_sum``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``reduction`` is not one of the three.
    """

    def __init__(
        self,
        target,
        temperature,
        *,
        kind: str = "gaussian",
        theiler_window: int = 0,
        norm_order: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.theiler_window = theiler_window
        self.norm_order = norm_order
        self.reduction = _check_reduction(reduction)

    def forward(self, predicted_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``; forwarded to ``soft_correlation_sum``.

        Returns:
            Scalar loss, or with ``reduction="none"`` the squared error of shape ``[]`` for an unbatched trajectory
            or ``[B]`` for a batched one, in the trajectory's floating dtype (half precision is computed in float32
            and cast back); ``NaN`` for a row containing NaN or inf.

        Raises:
            ValueError: As :func:`~torchnlts.dimension.soft_correlation_sum`, e.g. if ``theiler_window`` is not an
                integer.
        """
        trajectory = as_tensor(predicted_trajectory)
        output_dtype = _output_dtype(trajectory)
        trajectory = trajectory.to(_work_dtype(trajectory.dtype))
        temperature = self.temperature.to(trajectory.dtype)
        correlation_sum = soft_correlation_sum(
            trajectory,
            temperature,
            kind=self.kind,
            theiler_window=self.theiler_window,
            norm_order=self.norm_order,
        )
        dimension = correlation_dimension(temperature, correlation_sum)
        loss = _reduce(_squared_error(dimension, self.target.to(dimension.dtype)), self.reduction)
        return loss.to(output_dtype)


class RecurrenceLoss(torch.nn.Module):
    """Squared error between a predicted trajectory's recurrence rate and a target rate.

    Gradient class: smooth (inherits :func:`~torchnlts.recurrence.soft_recurrence_matrix`'s; the rate is exact).

    Args:
        target: Target recurrence rate in ``[0, 1]``: a scalar, or ``[B]`` to match a batched trajectory.
        temperature: Kernel bandwidth, forwarded to ``soft_recurrence_matrix``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_recurrence_matrix``.
        theiler_window: Forwarded to ``recurrence_rate``.
        norm_order: Forwarded to ``soft_recurrence_matrix``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``reduction`` is not one of the three.
    """

    def __init__(
        self,
        target,
        temperature,
        *,
        kind: str = "gaussian",
        theiler_window: int = 0,
        norm_order: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.theiler_window = theiler_window
        self.norm_order = norm_order
        self.reduction = _check_reduction(reduction)

    def forward(self, predicted_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``; forwarded to ``soft_recurrence_matrix``.

        Returns:
            Scalar loss, or with ``reduction="none"`` the squared error of shape ``[]`` for an unbatched trajectory
            or ``[B]`` for a batched one, in the trajectory's floating dtype (half precision is computed in float32
            and cast back); ``NaN`` for a row containing NaN or inf.

        Raises:
            ValueError: As :func:`~torchnlts.recurrence.recurrence_rate`, e.g. if ``theiler_window`` is not an
                integer.
        """
        trajectory, nonfinite, ndim = _clean_rows(predicted_trajectory)
        temperature = self.temperature.to(trajectory.dtype)
        weights = soft_recurrence_matrix(trajectory, temperature, kind=self.kind, norm_order=self.norm_order)
        rate = _nan_rows(recurrence_rate(weights, theiler_window=self.theiler_window), nonfinite, ndim)
        loss = _reduce(_squared_error(rate, self.target.to(rate.dtype)), self.reduction)
        return loss.to(_output_dtype(as_tensor(predicted_trajectory)))


class AttractorLoss(torch.nn.Module):
    """Squared maximum mean discrepancy (MMD) between the state-space point clouds of a predicted and a target
    trajectory.

    Reference: Gretton et al. (2012), J. Mach. Learn. Res. 13, 723, the biased V-statistic estimator
    ``E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]``, with ``k`` from :func:`~torchnlts.neighbors.trajectory_kernel`.

    Gradient class: smooth (inherits :func:`~torchnlts.neighbors.trajectory_kernel`'s).

    Distances are Euclidean and there is no ``norm_order`` argument: another norm can make the Gaussian kernel
    non-positive-semidefinite and the squared MMD negative (reproduced with ``norm_order=1``).
    Limitation: no Theiler exclusion on the within-trajectory terms; add one if nearby-in-time points inflate the
    self-similarity terms in a workload.

    Args:
        temperature: Kernel bandwidth, forwarded to ``trajectory_kernel``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``reduction`` is not one of the three.
    """

    def __init__(
        self,
        temperature,
        *,
        kind: str = "gaussian",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.reduction = _check_reduction(reduction)

    def forward(self, predicted_trajectory, target_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``.
            target_trajectory: Same batched-ness and feature dimension as ``predicted_trajectory``; its length may
                differ. The floating dtypes may differ too (float32 and float64 are promoted).

        Returns:
            Scalar loss, or with ``reduction="none"`` shape ``[]`` for unbatched trajectories or ``[B]`` for batched
            ones; ``NaN`` for a row where either trajectory contains NaN or inf. The dtype is the promoted dtype of
            the two trajectories (half precision is computed in float32).
        """
        predicted, predicted_nonfinite, ndim = _clean_rows(predicted_trajectory)
        target, target_nonfinite, target_ndim = _clean_rows(target_trajectory)
        _validate_matching_batch(predicted, target, ndim, target_ndim)
        predicted_distance = pairwise_distance(predicted, norm_order=2.0)
        temperature = self.temperature.to(torch.promote_types(predicted.dtype, target.dtype))
        pp = trajectory_kernel(predicted_distance, temperature, kind=self.kind).mean(dim=(-2, -1))
        qq = trajectory_kernel(pairwise_distance(target, norm_order=2.0), temperature, kind=self.kind).mean(
            dim=(-2, -1)
        )
        pq = trajectory_kernel(
            pairwise_distance(predicted, other=target, norm_order=2.0), temperature, kind=self.kind
        ).mean(dim=(-2, -1))
        mmd_squared = _nan_rows(pp + qq - 2 * pq, predicted_nonfinite | target_nonfinite, ndim)
        return _reduce(mmd_squared, self.reduction)


class SinkhornLoss(torch.nn.Module):
    """Debiased Sinkhorn divergence between the state-space point clouds of a predicted and a target trajectory.

    Reference: Feydy et al. (2019), AISTATS, PMLR 89, 2681; applied to chaotic attractors by Jiang, Lu, Orlova &
    Willett (2023), NeurIPS. Like :class:`~torchnlts.sciml.AttractorLoss` with entropy-regularized optimal transport
    (cost ``0.5 * d^2`` for Euclidean ``d``) in place of MMD: ``OT(P, Q) - (OT(P, P) + OT(Q, Q)) / 2``.

    Gradient class: smooth (a fixed number of unrolled ``exp``/``logsumexp`` iterations, differentiated through).

    Args:
        reg: Entropic regularization strength, a single finite real number ``> 0`` (a Python or numpy number, or a
            one-element tensor). Smaller values approach the exact optimal transport cost but need more ``num_iters``
            and lose numerical precision.
        num_iters: Number of Sinkhorn iterations, ``>= 1``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Raises:
        ValueError: If ``reg`` is not a single finite real number ``> 0``, ``num_iters`` is not an integer
            ``>= 1``, or ``reduction`` is not one of the three.
    """

    def __init__(self, reg, *, num_iters: int = 50, reduction: str = "mean") -> None:
        super().__init__()
        reg_tensor = _positive_scalar(reg, "reg")
        num_iters = _as_int(num_iters, "num_iters")
        if num_iters < 1:
            raise ValueError(f"num_iters must be >= 1, got {num_iters}")
        self.register_buffer("reg", reg_tensor)
        self.num_iters = num_iters
        self.reduction = _check_reduction(reduction)

    def forward(self, predicted_trajectory, target_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``.
            target_trajectory: Same conventions as :class:`~torchnlts.sciml.AttractorLoss`'s: same batched-ness and
                feature dimension, any length, floating dtypes may differ.

        Returns:
            Scalar loss, or with ``reduction="none"`` shape ``[]`` for unbatched trajectories or ``[B]`` for batched
            ones; ``NaN`` for a row where either trajectory contains NaN or inf. The dtype is the promoted dtype of
            the two trajectories (half precision is computed in float32).
        """
        predicted, predicted_nonfinite, ndim = _clean_rows(predicted_trajectory)
        target, target_nonfinite, target_ndim = _clean_rows(target_trajectory)
        _validate_matching_batch(predicted, target, ndim, target_ndim)
        reg = self.reg.to(torch.promote_types(predicted.dtype, target.dtype))
        predicted_cost = 0.5 * pairwise_distance(predicted, norm_order=2.0) ** 2
        target_cost = 0.5 * pairwise_distance(target, norm_order=2.0) ** 2
        cross_cost = 0.5 * pairwise_distance(predicted, other=target, norm_order=2.0) ** 2
        pp = _sinkhorn_cost(predicted_cost, reg, self.num_iters)
        qq = _sinkhorn_cost(target_cost, reg, self.num_iters)
        pq = _sinkhorn_cost(cross_cost, reg, self.num_iters)
        divergence = _nan_rows(pq - 0.5 * (pp + qq), predicted_nonfinite | target_nonfinite, ndim)
        return _reduce(divergence, self.reduction)
