"""Theiler-window masking of a pairwise distance matrix.

Gradient class: exact (mask depends only on time indices, not data).
Reference: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III.
"""

from __future__ import annotations

import torch


def theiler_window_mask(num_points: int, window: int, device=None) -> torch.Tensor:
    """Boolean mask of valid (temporally uncorrelated) index pairs.

    Args:
        num_points: Number of points in the trajectory.
        window: Theiler window. Pairs with ``|i - j| <= window`` are
            excluded; ``window = 0`` excludes only the diagonal.
        device: Device for the returned mask.

    Returns:
        Bool tensor of shape ``[T, T]``, ``True`` where ``|i - j| > window``.
    """
    index = torch.arange(num_points, device=device)
    return (index.unsqueeze(0) - index.unsqueeze(1)).abs() > window


def apply_theiler_window(distance: torch.Tensor, window: int, fill_value: float = float("inf")) -> torch.Tensor:
    """Mask out temporally correlated pairs of a distance matrix.

    Args:
        distance: Shape ``[T, T]`` or ``[B, T, T]``, as produced by
            :func:`torchnlts.neighbors.distance.pairwise_distance`.
        window: Theiler window (see :func:`theiler_window_mask`).
        fill_value: Value written into excluded entries. Use ``inf``
            (default) for a nearest-neighbor search; use ``0`` if
            ``distance`` instead holds kernel weights.

    Returns:
        Distance matrix of the same shape, with excluded entries set to
        ``fill_value``.
    """
    if distance.dim() not in (2, 3):
        raise ValueError(f"expected a [T, T] or [B, T, T] distance matrix, got {tuple(distance.shape)}")
    mask = theiler_window_mask(distance.shape[-1], window, device=distance.device)
    return torch.where(mask, distance, torch.full_like(distance, fill_value))
