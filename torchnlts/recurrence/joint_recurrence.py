"""Joint recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329, Sec. 3.4: JR_ij = Theta(epsilon_x - ||x_i - x_j||) * Theta(epsilon_y
- ||y_i - y_j||), the element-wise product of ``x``'s and ``y``'s own
self-recurrence matrices. Unlike :mod:`torchnlts.recurrence.cross_recurrence`
(which compares ``x`` and ``y`` directly, so both must lie in the same
space), joint recurrence compares each trajectory's *own* recurrence
structure at matching time indices -- ``x`` and ``y`` may have different
feature dimensionality (states of two different systems), but must share a
time axis: if they differ in length, both are truncated to the shorter one,
matching RecurrenceAnalysis.jl's ``JointRecurrenceMatrix``.

Reuses :func:`torchnlts.recurrence.hard_recurrence_matrix` /
:func:`torchnlts.recurrence.soft_recurrence_matrix` directly, one
call per trajectory.

Julia ground truth note (see tests/test_joint_recurrence.py): RecurrenceAnalysis.jl's
two-argument ``JointRecurrenceMatrix(R1, R2)`` constructor errors in the
pinned package version, so the tests instead compute
``Matrix(RecurrenceMatrix(x, epsilon_x)) .& Matrix(RecurrenceMatrix(y,
epsilon_y))`` directly -- confirmed to match the single-threshold
``JointRecurrenceMatrix(x, y, rthres)`` overload's output when
``epsilon_x == epsilon_y``, and it's the same element-wise-AND definition
the package's own docstring gives for ``JointRecurrenceMatrix``.

Limitation: the soft variant computes the full ``T_full x T_full``
recurrence matrix on each trajectory before slicing down to ``T_min x
T_min`` when ``x`` and ``y`` differ in length (O(T_full^2) compute/memory
and a wasted autograd graph over the discarded region). Upgrade path: slice
``trajectory``/``other`` to ``min(len(trajectory), len(other))`` before
calling ``soft_recurrence_matrix``. The hard variant truncates first.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix


def hard_joint_recurrence_matrix(
    trajectory,
    other,
    epsilon: float | None = None,
    other_epsilon: float | None = None,
    norm_order: float = 2.0,
    *,
    recurrence_rate: float | None = None,
    local_rate: float | None = None,
    scale: str | None = None,
) -> torch.Tensor:
    """Boolean joint recurrence matrix: where ``trajectory`` and ``other`` recur simultaneously.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same batched-ness as ``trajectory``
            (both batched with equal batch size, or both unbatched); its
            time length and feature dimensionality may otherwise differ.
        epsilon: Recurrence threshold for ``trajectory`` (see
            :func:`torchnlts.recurrence.hard_recurrence_matrix`).
        other_epsilon: Recurrence threshold for ``other``. Defaults to
            ``epsilon`` (matching RecurrenceAnalysis.jl's single-threshold
            ``JointRecurrenceMatrix(x, y, rthres)`` overload).
        norm_order: Minkowski norm order, shared by both trajectories.
        recurrence_rate: Instead of ``epsilon``: global recurrence rate of each
            trajectory's own matrix (see :func:`~torchnlts.recurrence.hard_recurrence_matrix`).
        local_rate: Instead of ``epsilon``: local recurrence rate of each
            trajectory's own matrix.
        scale: ``"max"``, ``"mean"`` or ``"std"``; makes ``epsilon`` and
            ``other_epsilon`` ratios.

    Returns:
        Bool tensor of shape ``[T, T]`` or ``[B, T, T]``, ``T`` = the
        shorter of ``trajectory``'s and ``other``'s lengths.

    Raises:
        ValueError: As :func:`~torchnlts.recurrence.hard_recurrence_matrix`; also if
            ``other_epsilon`` is given with ``recurrence_rate`` or ``local_rate``.
    """
    trajectory_batched, trajectory_ndim = _ensure_batched(trajectory)
    other_batched, other_ndim = _ensure_batched(other)
    _validate_matching_batch(trajectory_batched, other_batched, trajectory_ndim, other_ndim, check_feature_dim=False)
    epsilon_other = epsilon if other_epsilon is None else other_epsilon
    # Truncate first: rate-based thresholds depend on the length.
    length = min(trajectory_batched.shape[-2], other_batched.shape[-2])
    options = dict(norm_order=norm_order, recurrence_rate=recurrence_rate, local_rate=local_rate, scale=scale)
    recurrence = hard_recurrence_matrix(trajectory_batched[:, :length], epsilon, **options)
    other_recurrence = hard_recurrence_matrix(other_batched[:, :length], epsilon_other, **options)
    return _restore_batch(recurrence & other_recurrence, trajectory_ndim)


def soft_joint_recurrence_matrix(
    trajectory,
    other,
    temperature: torch.Tensor | float,
    other_temperature: torch.Tensor | float | None = None,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted joint recurrence matrix, a smooth stand-in for the hard threshold.

    Gradient class: smooth (product of two
    :func:`torchnlts.recurrence.soft_recurrence_matrix` relaxations).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same batched-ness as ``trajectory`` (see
            :func:`~torchnlts.recurrence.hard_joint_recurrence_matrix`).
        temperature: Kernel bandwidth for ``trajectory``, forwarded to
            ``trajectory_kernel``.
        other_temperature: Kernel bandwidth for ``other``. Defaults to
            ``temperature``.
        kind: ``"gaussian"`` or ``"exponential"``, shared by both trajectories.
        norm_order: Minkowski norm order, shared by both trajectories.

    Returns:
        Weight tensor in ``[0, 1]``, same shape as :func:`~torchnlts.recurrence.hard_joint_recurrence_matrix`.
    """
    trajectory_batched, trajectory_ndim = _ensure_batched(trajectory)
    other_batched, other_ndim = _ensure_batched(other)
    _validate_matching_batch(trajectory_batched, other_batched, trajectory_ndim, other_ndim, check_feature_dim=False)
    temperature_other = temperature if other_temperature is None else other_temperature
    weight = soft_recurrence_matrix(trajectory, temperature, kind=kind, norm_order=norm_order)
    other_weight = soft_recurrence_matrix(other, temperature_other, kind=kind, norm_order=norm_order)
    length = min(weight.shape[-1], other_weight.shape[-1])
    return weight[..., :length, :length] * other_weight[..., :length, :length]
