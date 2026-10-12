"""Fractional (optionally learnable) lag delay embedding, generalizing ``delay_embedding``.

Reference: Laakso, T. I. et al. (1996), "Splitting the unit delay: tools for fractional delay filter design", IEEE
Signal Processing Magazine 13(1), 30-60. Linear interpolation between the two nearest integer-lag samples is the
simplest fractional-delay filter in their taxonomy; integer lags reproduce ``delay_embedding`` exactly.

Gradient class: piecewise. The interpolation is continuous in every lag and its derivative equals the local finite
difference of the trajectory, but the derivative jumps when a lag crosses an integer.

No Julia equivalent exists: DelayEmbeddings.jl's ``embed`` accepts integer lags only.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _restore_batch
from torchnlts.convert import as_tensor


def fractional_delay_embedding(trajectory, delays) -> torch.Tensor:
    """Delay embedding at arbitrary (non-integer, non-uniform, optionally learnable) lags, by linear interpolation.

    Reference: Laakso et al. (1996), IEEE Signal Process. Mag. 13(1), 30-60.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        delays: 1-D tensor of ``embedding_dim`` finite, non-negative lags in any order and spacing. Pass a tensor
            with ``requires_grad=True`` to optimize the lags.

    Returns:
        Shape ``[N, embedding_dim * D]`` or ``[B, N, embedding_dim * D]`` with
        ``N = floor(T - 1 - max(delays)) + 1``. Row ``n`` interpolates ``trajectory`` at times ``n + delays``. The
        dtype is the promotion of the default dtype, ``delays`` and ``trajectory``.

    Raises:
        ValueError: If ``delays`` is not a non-empty 1-D tensor of finite, non-negative lags, or the trajectory is
            too short for the largest lag.
    """
    batched, ndim = _ensure_batched(trajectory)
    batch_size, num_points, dim = batched.shape
    delays = as_tensor(delays)
    if delays.dim() != 1:
        raise ValueError(f"delays must be 1D, got shape {tuple(delays.shape)}")
    if delays.numel() < 1:
        raise ValueError("delays must have at least one lag")
    # Interpolation needs a floating dtype; promote across delays, trajectory and the default so no lag is truncated
    # and the index arange keeps integer precision.
    target_dtype = torch.get_default_dtype()
    if delays.is_floating_point():
        target_dtype = torch.promote_types(target_dtype, delays.dtype)
    if batched.is_floating_point():
        target_dtype = torch.promote_types(target_dtype, batched.dtype)
    delays = delays.to(device=batched.device, dtype=target_dtype)
    if not torch.isfinite(delays).all():
        raise ValueError("delays must be finite")
    if torch.any(delays < 0):
        raise ValueError("delays must be non-negative")

    # Limitation: max_delay.item() syncs the host and fixes num_embedded, so torch.compile recompiles and vmap
    # cannot batch over delays; a fixed-size padded output would lift this.
    max_delay = delays.detach().max().item()
    num_embedded = math.floor((num_points - 1) - max_delay) + 1
    if num_embedded < 1:
        raise ValueError(f"trajectory of length {num_points} is too short for delays up to {max_delay}")

    starts = torch.arange(num_embedded, device=batched.device, dtype=delays.dtype)
    times = starts.unsqueeze(1) + delays.unsqueeze(0)  # [num_embedded, embedding_dim]
    floor_times = torch.floor(times)
    lower_index = floor_times.long()
    # frac is 0 at integer delays, but upper_index can pass the last index; clamp it.
    upper_index = (lower_index + 1).clamp(max=num_points - 1)
    frac = (times - floor_times).unsqueeze(-1)  # [num_embedded, embedding_dim, 1]

    lower = batched[:, lower_index, :]  # [batch_size, num_embedded, embedding_dim, dim]
    upper = batched[:, upper_index, :]
    embedded = lower * (1 - frac) + upper * frac
    embedded = embedded.reshape(batch_size, num_embedded, delays.shape[0] * dim)
    return _restore_batch(embedded, ndim)
