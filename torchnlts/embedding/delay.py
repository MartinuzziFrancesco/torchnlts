"""Exact integer-lag Takens delay embedding.

Gradient class: exact (pure indexing/reshape).
Reference: Takens, F. (1981), "Detecting strange attractors in
turbulence", Lecture Notes in Mathematics 898, Springer, 366-381.
Forward-offset convention matches DelayEmbeddings.jl's `embed`.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch


def delay_embedding(trajectory: torch.Tensor, embedding_dim: int, delay: int = 1) -> torch.Tensor:
    """Takens delay embedding of a trajectory.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        embedding_dim: Number of delayed copies per point.
        delay: Lag, in time steps.

    Returns:
        Embedded trajectory of shape ``[N, embedding_dim * D]`` or
        ``[B, N, embedding_dim * D]``, where
        ``N = T - (embedding_dim - 1) * delay`` and row ``n`` concatenates
        ``trajectory[n], trajectory[n + delay], ...``.
    """
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    batched, ndim = _ensure_batched(trajectory)
    batch_size, num_points, dim = batched.shape
    num_embedded = num_points - (embedding_dim - 1) * delay
    if num_embedded < 1:
        raise ValueError(
            f"trajectory of length {num_points} is too short for "
            f"embedding_dim={embedding_dim}, delay={delay} "
            f"(needs >= {(embedding_dim - 1) * delay + 1})"
        )
    starts = torch.arange(num_embedded, device=batched.device)
    offsets = torch.arange(embedding_dim, device=batched.device) * delay
    gather_index = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [num_embedded, embedding_dim]
    embedded = batched[:, gather_index, :]  # [batch_size, num_embedded, embedding_dim, dim]
    embedded = embedded.reshape(batch_size, num_embedded, embedding_dim * dim)
    return _restore_batch(embedded, ndim)
