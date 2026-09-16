"""Pairwise distance between points of a (batch of) trajectory.

Gradient class: exact (non-differentiable only at zero distance).
Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1-2.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch


def pairwise_distance(
    trajectory: torch.Tensor,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Pairwise Minkowski distance between points of a trajectory.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        other: Optional second trajectory, same convention as ``trajectory``.
            Defaults to ``trajectory`` (self-distance matrix).
        norm_order: Minkowski norm order (``2`` = Euclidean, ``1`` =
            Manhattan, ``float('inf')`` = Chebyshev).

    Returns:
        Distance matrix of shape ``[T, T']`` or ``[B, T, T']``.
    """
    batched, ndim = _ensure_batched(trajectory)
    other_batched = batched if other is None else _ensure_batched(other)[0]
    # compute_mode forced: torch.cdist's default matmul path for p=2
    # loses precision in float32 for points sharing a large offset.
    distance = torch.cdist(batched, other_batched, p=norm_order, compute_mode="donot_use_mm_for_euclid_dist")
    return _restore_batch(distance, ndim)
