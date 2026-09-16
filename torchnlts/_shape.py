"""Tensor-convention normalization shared by every public function.

Every public torchnlts function accepts a trajectory as ``[T]``, ``[T, D]``,
or ``[B, T, D]`` and normalizes internally to ``[B, T, D]``. ``restore_batch`` undoes that
normalization on the way out so unbatched input gets unbatched output.
"""

from __future__ import annotations

import torch


def ensure_batched(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Normalize ``[T]`` / ``[T, D]`` / ``[B, T, D]`` to ``[B, T, D]``.

    Returns the normalized tensor and the original number of dimensions
    (1, 2, or 3), so the caller can later strip the dimensions it added.
    """
    if x.dim() == 1:
        return x.view(1, -1, 1), 1
    if x.dim() == 2:
        return x.unsqueeze(0), 2
    if x.dim() == 3:
        return x, 3
    raise ValueError(f"expected a tensor of shape [T], [T, D], or [B, T, D], got {tuple(x.shape)}")


def restore_batch(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`ensure_batched` on an output that kept the batch dim.

    ``x`` is expected to still carry its leading batch dimension (shape
    ``[B, ...]``); this strips it for inputs that came in unbatched.
    """
    if ndim == 3:
        return x
    if ndim in (1, 2):
        return x.squeeze(0)
    raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
