"""Theiler-window masking of a pairwise distance matrix.

Reference: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III.

Gradient class: exact (the mask depends only on time indices).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _validate_square_matrix


def theiler_window_mask(num_points: int, window: int, *, device=None) -> torch.Tensor:
    """Boolean mask of valid (temporally uncorrelated) index pairs.

    Reference: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III.

    Gradient class: exact (the mask depends only on time indices).

    Args:
        num_points: Number of points in the trajectory.
        window: Theiler window. Pairs with ``|i - j| <= window`` are excluded, so ``0`` excludes only the diagonal.
            A negative window excludes nothing (the whole matrix, as some recurrence-quantification formulas use).
        device: Device of the returned mask.

    Returns:
        Bool tensor of shape ``[T, T]``, ``True`` where ``|i - j| > window``.

    Raises:
        ValueError: If ``num_points`` is not a non-negative integer or ``window`` is not an integer.
    """
    num_points = _as_int(num_points, "num_points")
    if num_points < 0:
        raise ValueError(f"num_points must be >= 0, got {num_points}")
    # Clamped to [-1, T] (same mask) so a huge Python int cannot overflow the int64 comparison.
    window = max(min(_as_int(window, "theiler_window"), num_points), -1)
    index = torch.arange(num_points, device=device)
    return (index.unsqueeze(0) - index.unsqueeze(1)).abs() > window


def _require_self_exclusion(theiler_window: int, context: str) -> int:
    """``theiler_window`` as a Python int; ``ValueError`` if it is not an integer or is negative, for a statistic that
    must exclude self-matches (sample entropy, KSG, ...)."""
    theiler_window = _as_int(theiler_window, "theiler_window")
    if theiler_window < 0:
        raise ValueError(
            f"theiler_window must be >= 0 for {context} (self-matches must be excluded), got {theiler_window}"
        )
    return theiler_window


def apply_theiler_window(distance: torch.Tensor, window: int, *, fill_value: float = float("inf")) -> torch.Tensor:
    """Mask out temporally correlated pairs of a distance matrix.

    Reference: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III.

    Gradient class: exact (excluded entries receive no gradient).

    Args:
        distance: Shape ``[T, T]`` or ``[B, T, T]``, as returned by :func:`~torchnlts.neighbors.pairwise_distance`.
        window: Theiler window, as in :func:`~torchnlts.neighbors.theiler_window_mask`.
        fill_value: Value written into excluded entries: ``inf`` (default) for a nearest-neighbor search, ``0`` for
            kernel weights.

    Returns:
        Matrix of the same shape with the excluded entries set to ``fill_value``.

    Raises:
        ValueError: If ``distance`` is not a square ``[T, T]`` or ``[B, T, T]`` matrix, or ``window`` is not an
            integer.
    """
    _validate_square_matrix(distance)
    mask = theiler_window_mask(distance.shape[-1], window, device=distance.device)
    return torch.where(mask, distance, torch.full_like(distance, fill_value))
