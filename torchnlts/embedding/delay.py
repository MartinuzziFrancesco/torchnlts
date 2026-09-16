"""Exact integer-lag Takens delay embedding.

Gradient class: **exact**. Pure indexing and reshape of the input; every
output entry is exactly one input entry, so autodiff is the ordinary
identity-scatter gradient everywhere, with no boundary to document.

Takens, F. (1981), "Detecting strange attractors in turbulence", in
Dynamical Systems and Turbulence, Lecture Notes in Mathematics 898,
Springer, 366-381: an m-dimensional delay-coordinate reconstruction
``X_n = (x_n, x_{n+tau}, ..., x_{n+(m-1)tau})`` is, for generic tau and
m > 2*(box-counting dimension of the attractor), diffeomorphic to the
original attractor. Forward-offset convention (using x_n, x_{n+tau}, ...
rather than x_n, x_{n-tau}, ...) matches DelayEmbeddings.jl's `embed`.
"""

from __future__ import annotations

import torch

from torchnlts._shape import ensure_batched, restore_batch


def delay_embedding(x: torch.Tensor, m: int, tau: int = 1) -> torch.Tensor:
    """Takens delay embedding of a trajectory.

    Args:
        x: Trajectory of shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        m: Embedding dimension (number of delayed copies per point).
        tau: Delay (lag), in time steps.

    Returns:
        Embedded trajectory of shape ``[N, m * D]`` (unbatched input) or
        ``[B, N, m * D]`` (batched input), where ``N = T - (m - 1) * tau``
        and row ``n`` is the concatenation of
        ``x[n], x[n + tau], ..., x[n + (m - 1) * tau]``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    xb, ndim = ensure_batched(x)
    b, t, d = xb.shape
    n = t - (m - 1) * tau
    if n < 1:
        raise ValueError(f"trajectory of length {t} is too short for m={m}, tau={tau} (needs >= {(m - 1) * tau + 1})")
    starts = torch.arange(n, device=xb.device)
    offsets = torch.arange(m, device=xb.device) * tau
    gather_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [n, m]
    embedded = xb[:, gather_idx, :]  # [b, n, m, d]
    embedded = embedded.reshape(b, n, m * d)
    return restore_batch(embedded, ndim)
