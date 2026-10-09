"""Fractional/learnable-lag delay embedding, generalizing delay_embedding.

Gradient class: piecewise. Linear interpolation is continuous in the
lag everywhere, and its derivative w.r.t. a lag equals the local finite
difference of the trajectory almost everywhere -- but that derivative
jumps whenever a lag crosses an exact integer (the pair of interpolated
samples changes). Gradcheck avoids that boundary by using half-integer
lags.

No Julia equivalent exists: DelayEmbeddings.jl's `embed` only accepts
integer lags (checked its method signatures directly). Reference:
Laakso, T. I. et al. (1996), "Splitting the unit delay: tools for
fractional delay filter design", IEEE Signal Processing Magazine 13(1),
30-60 -- linear interpolation between the two nearest integer-lag
samples, the simplest fractional-delay filter in their taxonomy.
Validated here against that formula directly (hand-derived) and against
:func:`torchnlts.embedding.delay_embedding` for integer lags,
where the two must agree exactly.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _restore_batch
from torchnlts.convert import as_tensor


def fractional_delay_embedding(trajectory, delays) -> torch.Tensor:
    """Delay embedding at arbitrary (non-integer, non-uniform, and
    optionally learnable) lags, via linear interpolation.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        delays: 1D tensor of ``embedding_dim`` non-negative lags (need
            not be integers, sorted, or evenly spaced). Pass a tensor
            with ``requires_grad=True`` to optimize the lags themselves.

    Returns:
        Embedded trajectory of shape ``[N, embedding_dim * D]`` or
        ``[B, N, embedding_dim * D]``, row ``n`` interpolating
        ``trajectory`` at times ``n + delays``.
    """
    batched, ndim = _ensure_batched(trajectory)
    batch_size, num_points, dim = batched.shape
    delays = as_tensor(delays)
    if delays.dim() != 1:
        raise ValueError(f"delays must be 1D, got shape {tuple(delays.shape)}")
    if delays.numel() < 1:
        raise ValueError("delays must have at least one lag")
    # Interpolation needs a floating dtype regardless of what batched or
    # delays came in as: casting delays to an integer trajectory's dtype
    # would silently truncate fractional lags (and drop their gradient).
    # Promote across delays' own dtype (if already floating), the
    # trajectory's dtype (if floating), and the global default, so
    # neither a float64 trajectory nor a float64 delays tensor ever
    # loses precision to a lower-precision partner, and `starts`'s
    # arange (dtype-matched to delays below) never lands in a dtype too
    # coarse to represent consecutive integers, corrupting indices.
    # torch.promote_types (not a bit-width comparison) correctly sends a
    # float16/bfloat16 mix to float32 rather than picking one of them.
    target_dtype = torch.get_default_dtype()
    if delays.is_floating_point():
        target_dtype = torch.promote_types(target_dtype, delays.dtype)
    if batched.is_floating_point():
        target_dtype = torch.promote_types(target_dtype, batched.dtype)
    delays = delays.to(device=batched.device, dtype=target_dtype)
    if torch.any(delays < 0):
        raise ValueError("delays must be non-negative")

    # Limitation: max_delay's .item() forces a host sync and bakes
    # num_embedded into the trace as a Python int, so torch.compile
    # recompiles whenever delays' value changes and vmap can't batch over
    # per-sample delays. Fixed-size output (a max-lag bound plus
    # padding/masking) would remove this limit; not built since nothing
    # here yet needs compile/vmap over a *batch of different* delays.
    max_delay = delays.detach().max().item()
    num_embedded = math.floor((num_points - 1) - max_delay) + 1
    if num_embedded < 1:
        raise ValueError(f"trajectory of length {num_points} is too short for delays up to {max_delay}")

    starts = torch.arange(num_embedded, device=batched.device, dtype=delays.dtype)
    times = starts.unsqueeze(1) + delays.unsqueeze(0)  # [num_embedded, embedding_dim]
    floor_times = torch.floor(times)
    lower_index = floor_times.long()
    # At an exact-integer delay, frac is 0 and upper_index's sample never
    # actually contributes -- but it can still land one past the last
    # valid index, so clamp it rather than index out of bounds.
    upper_index = (lower_index + 1).clamp(max=num_points - 1)
    frac = (times - floor_times).unsqueeze(-1)  # [num_embedded, embedding_dim, 1]

    lower = batched[:, lower_index, :]  # [batch_size, num_embedded, embedding_dim, dim]
    upper = batched[:, upper_index, :]
    embedded = lower * (1 - frac) + upper * frac
    embedded = embedded.reshape(batch_size, num_embedded, delays.shape[0] * dim)
    return _restore_batch(embedded, ndim)
