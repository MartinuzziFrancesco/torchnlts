"""Cross recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329, Sec. 4.1: CR_ij = Theta(epsilon - ||x_i - y_j||). Unlike
:mod:`torchnlts.recurrence.matrix`'s self-recurrence matrix, the cross
recurrence matrix is generally neither square nor symmetric (``x`` and
``y`` are different trajectories, possibly of different lengths), and has
no always-recurrent diagonal.

Reuses :func:`torchnlts.neighbors.distance.pairwise_distance`'s ``other``
argument and :func:`torchnlts.neighbors.kernels.trajectory_kernel`, the
same primitives :mod:`torchnlts.recurrence.matrix` builds on.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel


def hard_cross_recurrence_matrix(trajectory, other, epsilon: float, norm_order: float = 2.0) -> torch.Tensor:
    """Boolean cross recurrence matrix at a fixed distance threshold.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Second trajectory, same convention as ``trajectory``; its
            time length may differ, but any batch dimension must match.
        epsilon: Recurrence threshold. Points ``i``, ``j`` recur when
            their distance is ``<= epsilon``, matching RecurrenceAnalysis.jl's
            ``CrossRecurrenceMatrix``.
        norm_order: Minkowski norm order.

    Returns:
        Bool tensor of shape ``[T, T']`` or ``[B, T, T']``.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return distance <= epsilon


def soft_cross_recurrence_matrix(
    trajectory,
    other,
    temperature: torch.Tensor | float,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted cross recurrence matrix, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`hard_cross_recurrence_matrix`, built on
    :func:`torchnlts.neighbors.kernels.trajectory_kernel` rather than a
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
        Weight tensor in ``(0, 1]``, same shape as ``hard_cross_recurrence_matrix``.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    return trajectory_kernel(distance, temperature, kind=kind)
