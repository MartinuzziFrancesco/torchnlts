"""Shared argument checks for the matrix profile index and profile arrays."""

from __future__ import annotations

import torch


def _check_series_axis(tensor: torch.Tensor, name: str) -> None:
    """``tensor`` must be ``[n]`` or ``[B, n]`` with ``n >= 1`` (``B`` may be 0)."""
    if tensor.dim() not in (1, 2) or tensor.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [n] or [B, n] with n >= 1, got {tuple(tensor.shape)}")


def _as_profile_index(index, name: str = "index") -> torch.Tensor:
    """``index`` as an int64 tensor of shape ``[n]`` or ``[B, n]`` with entries in ``[-1, n - 1]``.

    Raises:
        ValueError: If ``index`` is not an integer (non-bool) tensor, has the wrong shape, or has
            an entry outside ``[-1, n - 1]``. The range check is skipped under ``torch.vmap``
            (it needs a data-dependent branch).
    """
    index = torch.as_tensor(index)
    if index.dtype == torch.bool or index.is_floating_point() or index.is_complex():
        raise ValueError(f"{name} must be an integer tensor, got {index.dtype}")
    _check_series_axis(index, name)
    if not torch._C._functorch.is_batchedtensor(index) and index.numel() > 0:
        low, high = index.amin().item(), index.amax().item()
        if low < -1 or high >= index.shape[-1]:
            raise ValueError(f"{name} entries must lie in [-1, {index.shape[-1] - 1}], got range [{low}, {high}]")
    return index.to(torch.int64)
