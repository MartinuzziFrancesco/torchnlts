"""Theiler-window masking of a pairwise distance matrix.

Gradient class: exact (mask depends only on time indices, not data).
Reference: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _validate_square_matrix


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


def _require_self_exclusion(theiler_window: int, context: str) -> None:
    """Shared guard for statistics that require excluding self-matches
    unconditionally (sample entropy, KSG mutual information, ...): a
    negative ``theiler_window`` would otherwise admit them via
    :func:`theiler_window_mask`'s documented "no exclusion" case, where
    every point trivially matches itself at distance 0 and the
    statistic's defining ratio/neighbor-count breaks down.
    """
    if theiler_window < 0:
        raise ValueError(f"theiler_window must be >= 0 for {context} (self-matches must be excluded), got {theiler_window}")


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
