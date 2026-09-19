"""Tensor-convention normalization shared by every public function."""

from __future__ import annotations

import torch

from torchnlts.convert import as_tensor


def _ensure_batched(tensor) -> tuple[torch.Tensor, int]:
    """Normalize ``[T]`` / ``[T, D]`` / ``[B, T, D]`` to ``[B, T, D]``.

    ``tensor`` may be a torch.Tensor, numpy array, or pandas
    Series/DataFrame (converted via :func:`torchnlts.convert.as_tensor`).

    Returns:
        The normalized tensor and its original number of dimensions.
    """
    tensor = as_tensor(tensor)
    if tensor.dim() == 1:
        return tensor.view(1, -1, 1), 1
    if tensor.dim() == 2:
        return tensor.unsqueeze(0), 2
    if tensor.dim() == 3:
        return tensor, 3
    raise ValueError(f"expected a tensor of shape [T], [T, D], or [B, T, D], got {tuple(tensor.shape)}")


def _restore_batch(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`_ensure_batched` on a result that kept the batch dim."""
    if ndim == 3:
        return tensor
    if ndim in (1, 2):
        return tensor.squeeze(0)
    raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")


def _broadcast_over_matrix(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar or ``[R]`` radius/temperature sweep to broadcast
    against ``matrix``'s trailing ``[..., T, T]`` dims as a leading ``R``
    axis, e.g. ``[R]`` -> ``[R, 1, 1]`` for an unbatched ``[T, T]`` matrix.
    An already-scalar (0-dim) ``value`` reshapes to all-ones, which
    broadcasts the same as leaving it scalar.
    """
    return value.view(value.shape + (1,) * matrix.dim())


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    """Shared shape guard for every function that masks a ``[T, T]``-style
    matrix by time-index pairs (Theiler window, RQA measures, ...):
    without it, a malformed shape surfaces as a confusing broadcast
    ``RuntimeError`` deep inside whichever masking op runs first, instead
    of a clear message at the call site.
    """
    if matrix.dim() not in (2, 3) or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected a [T, T] or [B, T, T] square matrix, got {tuple(matrix.shape)}")


def _validate_matching_batch(
    trajectory: torch.Tensor,
    other: torch.Tensor,
    trajectory_ndim: int,
    other_ndim: int,
    check_feature_dim: bool = True,
) -> None:
    """Shared shape guard for every two-trajectory function built on
    :func:`_ensure_batched` (``pairwise_distance``'s ``other`` argument, and
    everything that forwards it: ``knn_search``, ``radius_neighbors``, the
    cross/joint recurrence matrices, ...).

    Without it, a batch-dim mismatch (one trajectory batched, the other
    not) silently broadcasts to the wrong output shape instead of raising
    -- ``_restore_batch`` decides whether to squeeze using only
    ``trajectory``'s ndim, so a batched ``other`` paired with an unbatched
    ``trajectory`` returns an unintended ``[B, T, T']`` where a squeeze
    would otherwise have been a no-op. A feature-dim or non-broadcastable
    batch-size mismatch instead surfaces as a raw, context-free
    ``RuntimeError`` deep inside ``torch.cdist``.

    Args:
        trajectory: First trajectory, already normalized via ``_ensure_batched``.
        other: Second trajectory, already normalized via ``_ensure_batched``.
        trajectory_ndim: ``trajectory``'s original ndim, as returned by ``_ensure_batched``.
        other_ndim: ``other``'s original ndim, as returned by ``_ensure_batched``.
        check_feature_dim: Whether the two trajectories' feature dims must
            match. ``True`` for a shared-space comparison like cross
            distance (``x`` and ``y`` are points in the same space).
            ``False`` for a per-trajectory comparison like joint recurrence,
            where ``x`` and ``y`` may be states of two different systems of
            different dimensionality, combined only by matching time index.
    """
    trajectory_batched = trajectory_ndim == 3
    other_batched = other_ndim == 3
    if trajectory_batched != other_batched:
        raise ValueError(
            "trajectory and other must both be batched ([B, T, D]) or both "
            f"unbatched ([T] or [T, D]), got ndim {trajectory_ndim} and {other_ndim}"
        )
    if trajectory_batched and trajectory.shape[0] != other.shape[0]:
        raise ValueError(f"batch size mismatch between trajectory ({trajectory.shape[0]}) and other ({other.shape[0]})")
    if check_feature_dim and trajectory.shape[-1] != other.shape[-1]:
        raise ValueError(f"feature dim mismatch between trajectory ({trajectory.shape[-1]}) and other ({other.shape[-1]})")
