"""Theiler-window masking of a pairwise distance matrix.

Gradient class: **exact**. The mask depends only on time indices ``i, j``,
never on the distance values themselves, so masked-out entries get a
constant fill value (zero gradient) and kept entries pass their gradient
through unchanged -- no data-dependent boundary is crossed.

Theiler (1986), "Spurious dimension from correlation algorithms applied
to limited time-series data", Phys. Rev. A 34, 2427: temporally close
points on a trajectory are dynamically correlated, not independent
samples of the attractor, and including them in a correlation sum or
nearest-neighbor search biases dimension/Lyapunov estimates low. The
fix (Theiler's Sec. III) is to exclude pairs with ``|i - j| <= w`` before
computing the sum or picking neighbors, where ``w`` is the "Theiler
window".
"""

from __future__ import annotations

import torch


def theiler_window_mask(t: int, w: int, device=None) -> torch.Tensor:
    """Boolean mask of valid (temporally uncorrelated) index pairs.

    Args:
        t: Number of points in the trajectory.
        w: Theiler window. Pairs with ``|i - j| <= w`` are excluded;
            ``w = 0`` excludes only the diagonal (self-pairs).
        device: Device for the returned mask.

    Returns:
        Bool tensor of shape ``[T, T]``, ``True`` where ``|i - j| > w``.
    """
    idx = torch.arange(t, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > w


def apply_theiler_window(distance: torch.Tensor, w: int, fill_value: float = float("inf")) -> torch.Tensor:
    """Mask out temporally correlated pairs of a distance matrix.

    Args:
        distance: Distance matrix of shape ``[T, T]`` or ``[B, T, T]``,
            as produced by :func:`torchnlts.neighbors.distance.pairwise_distance`.
        w: Theiler window (see :func:`theiler_window_mask`).
        fill_value: Value written into excluded entries. Use ``inf``
            (default) so excluded pairs never win a nearest-neighbor
            search; use ``0`` if the matrix instead holds kernel weights.

    Returns:
        Distance matrix of the same shape, with ``|i - j| <= w`` entries
        set to ``fill_value``.
    """
    if distance.dim() not in (2, 3):
        raise ValueError(f"expected a [T, T] or [B, T, T] distance matrix, got {tuple(distance.shape)}")
    t = distance.shape[-1]
    mask = theiler_window_mask(t, w, device=distance.device)
    return torch.where(mask, distance, torch.full_like(distance, fill_value))
