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
            excluded; ``window = 0`` excludes only the diagonal. A
            negative window excludes nothing (every pair passes, since
            ``|i - j| >= 0 > window``) -- the literal "whole matrix, no
            exclusion" case some recurrence-quantification formulas use.
        device: Device for the returned mask.

    Returns:
        Bool tensor of shape ``[T, T]``, ``True`` where ``|i - j| > window``.
    """
    index = torch.arange(num_points, device=device)
    return (index.unsqueeze(0) - index.unsqueeze(1)).abs() > window


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    """Shared shape guard for every function that masks a ``[T, T]``-style
    matrix by time-index pairs (Theiler window, RQA measures, ...):
    without it, a malformed shape surfaces as a confusing broadcast
    ``RuntimeError`` deep inside whichever masking op runs first, instead
    of a clear message at the call site.
    """
    if matrix.dim() not in (2, 3) or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected a [T, T] or [B, T, T] square matrix, got {tuple(matrix.shape)}")


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
    _validate_square_matrix(distance)
    mask = theiler_window_mask(distance.shape[-1], window, device=distance.device)
    return torch.where(mask, distance, torch.full_like(distance, fill_value))
