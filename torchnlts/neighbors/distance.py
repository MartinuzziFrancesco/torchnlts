"""Pairwise distance between points of a (batch of) trajectory.

Gradient class: exact (non-differentiable only at zero distance).
Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1-2.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _ensure_batched,
    _is_complex,
    _power_of_two_scale,
    _restore_batch,
    _validate_matching_batch,
    _work_dtype,
)


def pairwise_distance(
    trajectory: torch.Tensor,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Pairwise Minkowski distance between points of a trajectory.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        other: Optional second trajectory, same convention as ``trajectory``.
            Defaults to ``trajectory`` (self-distance matrix). Must agree
            with ``trajectory`` on batched-ness (both ``[B, T, D]`` or both
            unbatched) and feature dim; batch size must match if both are
            batched.
        norm_order: Minkowski norm order (``2`` = Euclidean, ``1`` =
            Manhattan, ``float('inf')`` = Chebyshev, ``0`` = number of
            unequal coordinates).

    Returns:
        Distance matrix of shape ``[T, T']`` or ``[B, T, T']``, in the
        input's floating dtype (the default dtype for integer input,
        float32 for half precision). ``NaN`` for every pair involving a
        point with a NaN/inf coordinate.

    Raises:
        ValueError: If either trajectory is complex or ``norm_order`` is
            not a real number >= 0.

    Note:
        Distances far below the trajectory's largest magnitude underflow to
        zero: in float32 below about ``1e-19`` times it for ``norm_order=2``,
        ``2**(-126 / norm_order)`` in general (``1`` and ``inf`` are not
        affected). See the numerical range limits in :doc:`/primitives`.
    """
    norm_order = _validate_norm_order(norm_order)
    batched, ndim = _ensure_batched(trajectory)
    if other is None:
        other_batched = batched
    else:
        other_batched, other_ndim = _ensure_batched(other)
        _validate_matching_batch(batched, other_batched, ndim, other_ndim)
    # cdist supports neither integer nor (on CPU) half precision.
    batched = batched.to(_work_dtype(batched.dtype))
    other_batched = other_batched.to(_work_dtype(other_batched.dtype))
    # Zero non-finite points so cdist's backward can't put NaN into other pairs; their distances are NaN below.
    nonfinite = ~torch.isfinite(batched).all(dim=-1)
    other_nonfinite = ~torch.isfinite(other_batched).all(dim=-1)
    batched = torch.where(nonfinite[..., None], torch.zeros_like(batched), batched)
    other_batched = torch.where(other_nonfinite[..., None], torch.zeros_like(other_batched), other_batched)
    # The default matmul path loses float32 precision for points sharing a large offset.
    # Exact power-of-two scaling keeps cdist from overflowing; skipped below norm_order=1, where it can't overflow.
    if norm_order < 1:
        distance = torch.cdist(batched, other_batched, p=norm_order)
    else:
        scale = _power_of_two_scale(batched, other_batched)
        scaled_distance = torch.cdist(
            batched / scale / 4, other_batched / scale / 4, p=norm_order, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distance = scaled_distance * 4 * scale
    distance = distance.masked_fill(nonfinite.unsqueeze(-1) | other_nonfinite.unsqueeze(-2), float("nan"))
    return _restore_batch(distance, ndim)


def _validate_norm_order(norm_order) -> float:
    """``norm_order`` as a Python float >= 0."""
    try:
        value = None if _is_complex(norm_order) else float(norm_order)
    except (TypeError, ValueError, RuntimeError):
        value = None
    if value is None or not value >= 0:
        raise ValueError(f"norm_order must be a real number >= 0, got {norm_order!r}")
    return value
