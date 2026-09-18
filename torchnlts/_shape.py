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


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    """Shared shape guard for every function that masks a ``[T, T]``-style
    matrix by time-index pairs (Theiler window, RQA measures, ...):
    without it, a malformed shape surfaces as a confusing broadcast
    ``RuntimeError`` deep inside whichever masking op runs first, instead
    of a clear message at the call site.
    """
    if matrix.dim() not in (2, 3) or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected a [T, T] or [B, T, T] square matrix, got {tuple(matrix.shape)}")
