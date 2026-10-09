"""Cross recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329, Sec. 3.3: CR_ij = Theta(epsilon - ||x_i - y_j||). Unlike
:mod:`torchnlts.recurrence.matrix`'s self-recurrence matrix, the cross
recurrence matrix is generally neither square nor symmetric (``x`` and
``y`` are different trajectories, possibly of different lengths), and has
no always-recurrent diagonal.

Reuses :func:`torchnlts.neighbors.pairwise_distance`'s ``other``
argument and :func:`torchnlts.neighbors.trajectory_kernel`, the
same primitives :mod:`torchnlts.recurrence.matrix` builds on.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.recurrence.matrix import _threshold_matrix


def hard_cross_recurrence_matrix(
    trajectory,
    other,
    epsilon: float | None = None,
    norm_order: float = 2.0,
    *,
    recurrence_rate: float | None = None,
    local_rate: float | None = None,
    scale: str | None = None,
) -> torch.Tensor:
    """Boolean cross recurrence matrix at a fixed, rate-derived or relative distance threshold.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same convention as ``trajectory``; its
            time length may differ, but any batch dimension must match.
        epsilon: Recurrence threshold. Points ``i``, ``j`` recur when
            their distance is ``<= epsilon``, matching RecurrenceAnalysis.jl's
            ``CrossRecurrenceMatrix``. With ``scale``, a ratio of that
            statistic of the distances.
        norm_order: Minkowski norm order.
        recurrence_rate: Instead of ``epsilon``: the threshold is this
            quantile of all ``T * T'`` distances. In ``(0, 1]``.
        local_rate: Instead of ``epsilon``: each column (point of ``other``)
            has its own threshold giving it about this fraction of recurrences
            (quantile-derived; ties can add recurrences).
            In ``(0, 1]``.
        scale: ``"max"``, ``"mean"`` or ``"std"``; makes ``epsilon`` a ratio.

    Returns:
        Bool tensor of shape ``[T, T']`` or ``[B, T, T']``; each batch row gets
        its own threshold, computed over its finite distances.

    Raises:
        ValueError: If not exactly one of ``epsilon``, ``recurrence_rate``,
            ``local_rate`` is given, ``scale`` is invalid or given without
            ``epsilon``, or a rate is outside ``(0, 1]``.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return _threshold_matrix(distance, epsilon, recurrence_rate, local_rate, scale, self_distance=False)


def soft_cross_recurrence_matrix(
    trajectory,
    other,
    temperature: torch.Tensor | float,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted cross recurrence matrix, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`~torchnlts.recurrence.hard_cross_recurrence_matrix`, built on
    :func:`torchnlts.neighbors.trajectory_kernel` rather than a
    hidden default swap).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same convention as ``trajectory``; its
            time length may differ, but any batch dimension must match.
        temperature: Kernel bandwidth, forwarded to ``trajectory_kernel``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        norm_order: Minkowski norm order.

    Returns:
        Weight tensor in ``(0, 1]``, same shape as ``hard_cross_recurrence_matrix``;
        ``0`` for pairs involving a point with a NaN/inf coordinate.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return trajectory_kernel(distance, temperature, kind=kind)
