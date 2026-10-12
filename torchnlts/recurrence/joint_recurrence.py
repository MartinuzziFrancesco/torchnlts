"""Joint recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6), 237-329, Sec. 3.4:
``JR_ij = Theta(epsilon_x - ||x_i - x_j||) * Theta(epsilon_y - ||y_i - y_j||)``, the element-wise product of the
self-recurrence matrices of ``x`` and ``y``. Unlike the cross recurrence matrix, ``x`` and ``y`` may differ in
feature dimension (states of two different systems) but must share a time axis: a longer one is truncated to the
shorter length before the thresholds are computed, as in RecurrenceAnalysis.jl's ``JointRecurrenceMatrix``.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix


def hard_joint_recurrence_matrix(
    trajectory,
    other,
    *,
    epsilon: float | None = None,
    other_epsilon: float | None = None,
    norm_order: float = 2.0,
    recurrence_rate: float | None = None,
    local_rate: float | None = None,
    scale: str | None = None,
) -> torch.Tensor:
    """Boolean joint recurrence matrix: where ``trajectory`` and ``other`` recur simultaneously.

    Reference: Marwan et al. (2007), Phys. Rep. 438(5-6), 237, Sec. 3.4.

    Gradient class: none (Heaviside threshold).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        other: Second trajectory with the same batchedness as ``trajectory`` (both batched with equal batch size, or
            both unbatched); its time length and feature dim may differ.
        epsilon: Recurrence threshold for ``trajectory``, as in :func:`~torchnlts.recurrence.hard_recurrence_matrix`.
        other_epsilon: Recurrence threshold for ``other``. Default: ``epsilon``.
        norm_order: Minkowski norm order, shared by both trajectories.
        recurrence_rate: Instead of ``epsilon``: global recurrence rate of each trajectory's own matrix.
        local_rate: Instead of ``epsilon``: local recurrence rate of each trajectory's own matrix.
        scale: ``"max"``, ``"mean"`` or ``"std"``; makes ``epsilon`` and ``other_epsilon`` ratios.

    Returns:
        Bool tensor of shape ``[T, T]`` or ``[B, T, T]``, ``T`` the shorter of the two lengths.

    Raises:
        ValueError: As :func:`~torchnlts.recurrence.hard_recurrence_matrix`; also if ``other_epsilon`` is given
            with ``recurrence_rate`` or ``local_rate``, or the trajectories do not match in batchedness or batch
            size.
    """
    trajectory_batched, trajectory_ndim = _ensure_batched(trajectory)
    other_batched, other_ndim = _ensure_batched(other)
    _validate_matching_batch(trajectory_batched, other_batched, trajectory_ndim, other_ndim, check_feature_dim=False)
    epsilon_other = epsilon if other_epsilon is None else other_epsilon
    # Truncate first: rate-based thresholds depend on the length.
    length = min(trajectory_batched.shape[-2], other_batched.shape[-2])
    options = dict(norm_order=norm_order, recurrence_rate=recurrence_rate, local_rate=local_rate, scale=scale)
    recurrence = hard_recurrence_matrix(trajectory_batched[:, :length], epsilon=epsilon, **options)
    other_recurrence = hard_recurrence_matrix(other_batched[:, :length], epsilon=epsilon_other, **options)
    return _restore_batch(recurrence & other_recurrence, trajectory_ndim)


def soft_joint_recurrence_matrix(
    trajectory,
    other,
    temperature: torch.Tensor | float,
    *,
    other_temperature: torch.Tensor | float | None = None,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted joint recurrence matrix, a smooth stand-in for the hard threshold.

    Reference: Marwan et al. (2007), Phys. Rep. 438(5-6), 237, Sec. 3.4, with the Heaviside steps replaced by
    :func:`~torchnlts.neighbors.trajectory_kernel`.

    Gradient class: smooth (product of two :func:`~torchnlts.recurrence.soft_recurrence_matrix` relaxations).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        other: Second trajectory, as in :func:`~torchnlts.recurrence.hard_joint_recurrence_matrix`.
        temperature: Kernel bandwidth for ``trajectory``; must be > 0 (see
            :func:`~torchnlts.neighbors.trajectory_kernel`).
        other_temperature: Kernel bandwidth for ``other``. Default: ``temperature``.
        kind: ``"gaussian"`` or ``"exponential"``, shared by both trajectories.
        norm_order: Minkowski norm order, shared by both trajectories.

    Returns:
        Weights in ``[0, 1]``, shaped like the result of :func:`~torchnlts.recurrence.hard_joint_recurrence_matrix`.

    Raises:
        ValueError: If a temperature is not a positive real number or tensor, ``kind`` is unknown, or the
            trajectories do not match in batchedness or batch size.
    """
    trajectory_batched, trajectory_ndim = _ensure_batched(trajectory)
    other_batched, other_ndim = _ensure_batched(other)
    _validate_matching_batch(trajectory_batched, other_batched, trajectory_ndim, other_ndim, check_feature_dim=False)
    temperature_other = temperature if other_temperature is None else other_temperature
    length = min(trajectory_batched.shape[-2], other_batched.shape[-2])
    weight = soft_recurrence_matrix(trajectory_batched[:, :length], temperature, kind=kind, norm_order=norm_order)
    other_weight = soft_recurrence_matrix(
        other_batched[:, :length], temperature_other, kind=kind, norm_order=norm_order
    )
    return _restore_batch(weight * other_weight, trajectory_ndim)
