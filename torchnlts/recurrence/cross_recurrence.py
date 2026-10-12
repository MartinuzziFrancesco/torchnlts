"""Cross recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6), 237-329, Sec. 3.3:
``CR_ij = Theta(epsilon - ||x_i - y_j||)``. Unlike the self-recurrence matrix, it is generally neither square nor
symmetric (``x`` and ``y`` are different trajectories, possibly of different lengths) and has no always-recurrent
diagonal.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.recurrence.matrix import _threshold_matrix


def hard_cross_recurrence_matrix(
    trajectory,
    other,
    *,
    epsilon: float | None = None,
    norm_order: float = 2.0,
    recurrence_rate: float | None = None,
    local_rate: float | None = None,
    scale: str | None = None,
) -> torch.Tensor:
    """Boolean cross recurrence matrix at a fixed, rate-derived or relative distance threshold.

    Reference: Marwan et al. (2007), Phys. Rep. 438(5-6), 237, Sec. 3.3. Matches RecurrenceAnalysis.jl's
    ``CrossRecurrenceMatrix``.

    Gradient class: none (Heaviside threshold).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same convention as ``trajectory``; its time length may differ, but any batch
            dimension must match.
        epsilon: Recurrence threshold, as in :func:`~torchnlts.recurrence.hard_recurrence_matrix`.
        norm_order: Minkowski norm order.
        recurrence_rate: As in :func:`~torchnlts.recurrence.hard_recurrence_matrix`, over all ``T * T'`` distances.
        local_rate: As in :func:`~torchnlts.recurrence.hard_recurrence_matrix`, with a threshold per column (point of
            ``other``).
        scale: As in :func:`~torchnlts.recurrence.hard_recurrence_matrix`.

    Returns:
        Bool tensor of shape ``[T, T']`` or ``[B, T, T']``. Each batch row gets its own threshold, computed over its
        finite distances.

    Raises:
        ValueError: If not exactly one of ``epsilon``, ``recurrence_rate``, ``local_rate`` is given, ``scale`` is
            invalid or given without ``epsilon``, a rate is outside ``(0, 1]``, or the trajectories do not match in
            batchedness, feature dim or batch size.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return _threshold_matrix(distance, epsilon, recurrence_rate, local_rate, scale, self_distance=False)


def soft_cross_recurrence_matrix(
    trajectory,
    other,
    temperature: torch.Tensor | float,
    *,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted cross recurrence matrix, a smooth stand-in for the hard threshold.

    Reference: Marwan et al. (2007), Phys. Rep. 438(5-6), 237, Sec. 3.3, with the Heaviside step replaced by
    :func:`~torchnlts.neighbors.trajectory_kernel`.

    Gradient class: smooth.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same convention as ``trajectory``; its time length may differ, but any batch
            dimension must match.
        temperature: Kernel bandwidth, the smooth analogue of ``epsilon``; must be > 0 (see
            :func:`~torchnlts.neighbors.trajectory_kernel`).
        kind: ``"gaussian"`` or ``"exponential"``.
        norm_order: Minkowski norm order.

    Returns:
        Weights in ``[0, 1]``, shaped like the result of :func:`~torchnlts.recurrence.hard_cross_recurrence_matrix`;
        ``0`` for pairs involving a point with a NaN/inf coordinate.

    Raises:
        ValueError: If ``temperature`` is not a positive real number or tensor, ``kind`` is unknown, or the
            trajectories do not match in batchedness, feature dim or batch size.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return trajectory_kernel(distance, temperature, kind=kind)
