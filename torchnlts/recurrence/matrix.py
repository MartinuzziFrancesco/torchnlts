"""Recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329, eq. 1: R_ij = Theta(epsilon - ||x_i - x_j||).

The hard variant reuses :func:`torchnlts.neighbors.distance.pairwise_distance`
and is validated against RecurrenceAnalysis.jl's ``RecurrenceMatrix``. The
soft variant reuses :func:`torchnlts.neighbors.kernels.trajectory_kernel` as
its base primitive -- ``temperature`` plays the role of
``epsilon``'s smooth analogue, not a relaxed threshold at a fixed epsilon.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel


def hard_recurrence_matrix(trajectory, epsilon: float, norm_order: float = 2.0) -> torch.Tensor:
    """Boolean recurrence matrix at a fixed distance threshold.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        epsilon: Recurrence threshold. Points ``i``, ``j`` recur when
            their distance is ``<= epsilon``, matching RecurrenceAnalysis.jl's
            ``RecurrenceMatrix`` (including the always-recurrent diagonal).
        norm_order: Minkowski norm order.

    Returns:
        Bool tensor of shape ``[T, T]`` or ``[B, T, T]``.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    return distance <= epsilon


def soft_recurrence_matrix(
    trajectory,
    temperature: torch.Tensor | float,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted recurrence matrix, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`hard_recurrence_matrix`, built on
    :func:`torchnlts.neighbors.kernels.trajectory_kernel` rather than a
    hidden default swap).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        temperature: Kernel bandwidth, forwarded to ``trajectory_kernel``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        norm_order: Minkowski norm order.

    Returns:
        Weight tensor in ``(0, 1]``, same shape as ``hard_recurrence_matrix``.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    return trajectory_kernel(distance, temperature, kind=kind)
