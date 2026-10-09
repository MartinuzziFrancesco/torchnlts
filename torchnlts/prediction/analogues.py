"""Method of analogues: forecast by finding earlier stretches of the series that resemble the most recent
context and reading off what followed them.

Reference: Lorenz, E.N. (1969), "Atmospheric predictability as revealed by naturally occurring analogues",
J. Atmos. Sci. 26(4), 636-646. Similarity is the z-normalized Euclidean distance of Yeh et al. (2016),
"Matrix Profile I", IEEE ICDM (see :func:`~torchnlts.matrixprofile.matrix_profile`), so analogues match in
shape, not level or amplitude, and their continuations are mapped back into the context's scale.

Limitation: the distance matrix is formed one chunk of origins at a time (about ``_CHUNK_ELEMENTS`` entries),
so memory is O(T) but time is O(O * T * context_length); an FFT-based sliding dot product (MASS) is the upgrade
path.

Gradient class: piecewise (the selection of analogues is discrete; distances and continuations are smooth in
the series for a fixed selection, with kinks at ties and at constant windows). The starts have none.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _as_int, _ensure_batched, _output_dtype, _restore_batch, _work_dtype
from torchnlts.matrixprofile.profile import _znormalized_windows
from torchnlts.neighbors.distance import pairwise_distance

# Distance-matrix entries (batch * origins * candidate starts) formed at once.
_CHUNK_ELEMENTS = 2**22


def _window_mean_std(windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean, population std and constant flag of ``windows`` ``[..., L]`` along the last axis.
    The mean is relative to the first value (``mean - windows[..., 0]``, exact for a large offset); mean and
    std are computed on a power-of-two rescaled copy so the squares stay in range."""
    constant = windows.amax(dim=-1) == windows.amin(dim=-1)
    shifted = windows - windows[..., :1]
    # Power-of-two rescaling (exact) so the squares neither underflow nor overflow.
    with torch.no_grad():
        exponent = torch.frexp(shifted.abs().amax(dim=-1, keepdim=True))[1]
    scaled = torch.ldexp(shifted, -exponent)
    mean = scaled.mean(dim=-1)
    variance = ((scaled - mean[..., None]) ** 2).mean(dim=-1)
    std = torch.sqrt(torch.where(constant, torch.ones_like(variance), variance))
    exponent = exponent.squeeze(-1)
    return torch.ldexp(mean, exponent), torch.ldexp(std, exponent), constant


def analogues(
    series, origins, context_length: int, horizon: int, num_analogues: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest analogues of the context before each forecast origin, with their continuations.

    Reference: Lorenz (1969), J. Atmos. Sci. 26(4), 636.

    Gradient class: piecewise (distances and continuations; selection is discrete); starts have none.

    For origin ``t`` the context is ``series[t - context_length : t]``. A candidate start ``s`` is admissible if
    its window and its continuation end before the origin, ``s + context_length + horizon <= t``, and both are
    finite. Candidates are ranked by the z-normalized Euclidean distance to the context (a constant window is
    mapped to zero, as in :func:`~torchnlts.matrixprofile.matrix_profile`). The ``num_analogues`` nearest are
    taken greedily in ascending distance, removing candidates within ``ceil(context_length / 4)`` of each pick.
    The continuation ``series[s + context_length : s + context_length + horizon]`` is mapped into the context's
    scale, ``(c - mean(window)) / std(window) * std(context) + mean(context)`` (population std); if either
    window is constant the std ratio is 1 (shift only).

    Args:
        series: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        origins: Integer forecast origins ``t``, shape ``[O]`` (a tensor or sequence), shared across the
            batch, each in ``[context_length, T]``.
        context_length: Window length, ``>= 2``.
        horizon: Continuation length, ``>= 1``.
        num_analogues: Number of analogues ``k`` per origin, ``>= 1``.

    Returns:
        ``(distances, starts, continuations)`` of shape ``[O, k]``, ``[O, k]`` (int64) and
        ``[O, k, horizon]``, with a leading ``B`` for batched input. Floating outputs have the input's
        dtype. A missing analogue (fewer admissible candidates than ``k``, or a non-finite context) has
        distance ``inf``, start ``-1`` and continuation ``NaN``.

    Raises:
        ValueError: If the input is not univariate or is complex, an argument is out of range, or an origin
            lies outside ``[context_length, T]``.
    """
    tensor, ndim = _ensure_batched(series)
    dtype = _output_dtype(tensor)
    if tensor.shape[-1] != 1:
        raise ValueError(f"analogues only supports a univariate (D=1) series, got feature dim {tensor.shape[-1]}")
    context_length = _as_int(context_length, "context_length")
    horizon = _as_int(horizon, "horizon")
    num_analogues = _as_int(num_analogues, "num_analogues")
    num_points = tensor.shape[-2]
    if context_length < 2 or context_length > num_points:
        raise ValueError(f"context_length must be in [2, T={num_points}], got {context_length}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if num_analogues < 1:
        raise ValueError(f"num_analogues must be >= 1, got {num_analogues}")
    origins = torch.as_tensor(origins, device=tensor.device)
    if origins.numel() == 0 and origins.dim() == 1:
        origins = origins.to(torch.int64)
    if origins.dim() != 1 or origins.dtype == torch.bool or origins.is_floating_point() or origins.is_complex():
        raise ValueError("origins must be a 1-D integer tensor or sequence")
    origins = origins.to(torch.int64)
    if origins.numel() > 0 and (origins.amin() < context_length or origins.amax() > num_points):
        raise ValueError(f"origins must lie in [context_length={context_length}, T={num_points}]")

    values = tensor.squeeze(-1).to(_work_dtype(tensor.dtype))
    batch, num_origins = values.shape[0], origins.numel()
    num_starts = num_points - context_length + 1
    zone = math.ceil(context_length / 4)
    normalized, window_finite = _znormalized_windows(values.detach(), context_length)
    # A start's continuation is usable only if all its horizon values are finite.
    nonfinite_count = torch.cat(
        [torch.zeros_like(values[:, :1], dtype=torch.int64), (~torch.isfinite(values)).cumsum(dim=-1)], dim=-1
    )
    start = torch.arange(num_starts, device=values.device)
    cont_end = (start + context_length + horizon).clamp(max=num_points)
    continuation_finite = nonfinite_count[:, cont_end] - nonfinite_count[:, start + context_length] == 0
    candidate_ok = window_finite & continuation_finite  # [B, n]
    step = torch.arange(horizon, device=values.device)

    out_distance, out_start, out_continuation = [], [], []
    chunk = max(1, _CHUNK_ELEMENTS // max(1, batch * num_starts))
    for first in range(0, num_origins, chunk):
        origin = origins[first : first + chunk]  # [C]
        context_start = origin - context_length
        with torch.no_grad():
            distance = pairwise_distance(normalized[:, context_start], normalized)  # [B, C, n]
            admissible = candidate_ok[:, None, :] & (
                start[None, None, :] <= (origin - context_length - horizon)[None, :, None]
            )
            admissible = admissible & window_finite[:, context_start][:, :, None]
            distance = torch.where(
                admissible & torch.isfinite(distance), distance, torch.full_like(distance, float("inf"))
            )
            picks = []
            for _ in range(num_analogues):
                best, pick = distance.min(dim=-1)
                pick = torch.where(torch.isfinite(best), pick, torch.full_like(pick, -1))
                picks.append(pick)
                near = (start[None, None, :] - pick[..., None]).abs() <= zone
                distance = distance.masked_fill(near | (pick[..., None] < 0), float("inf"))
            pick = torch.stack(picks, dim=-1)  # [B, C, k]
        missing = pick < 0
        safe = pick.clamp(min=0)
        context = values[:, (context_start[:, None] + torch.arange(context_length, device=values.device))]  # [B, C, L]
        window_index = safe[..., None] + torch.arange(context_length, device=values.device)  # [B, C, k, L]
        batch_index = torch.arange(batch, device=values.device)[:, None, None, None]
        window = values[batch_index, window_index]
        window = torch.where(missing[..., None], torch.zeros_like(window), window)
        context = torch.where(missing.all(dim=-1)[..., None], torch.zeros_like(context), context)
        # Distance of the selected pairs, differentiable through both windows.
        window_z, _ = _znormalized_windows(window.reshape(-1, context_length), context_length)
        context_z, _ = _znormalized_windows(context.reshape(-1, context_length), context_length)
        window_z = window_z.reshape(*window.shape)
        context_z = context_z.reshape(*context.shape)
        chosen = torch.linalg.vector_norm(window_z - context_z[:, :, None, :], dim=-1)
        chosen = torch.where(missing, torch.full_like(chosen, float("inf")), chosen)
        window_mean, window_std, window_constant = _window_mean_std(window)
        context_mean, context_std, context_constant = _window_mean_std(context)
        ratio = torch.where(
            window_constant | context_constant[:, :, None],
            torch.ones_like(window_std),
            context_std[:, :, None] / window_std,
        )
        # Missing picks gather from start 0, which can run past T; clamp, they are masked below.
        ahead_index = ((safe + context_length)[..., None] + step).clamp(max=num_points - 1)
        ahead = values[batch_index, ahead_index]  # [B, C, k, H]
        ahead = torch.where(missing[..., None], torch.zeros_like(ahead), ahead)
        # (c - mean(window)) * ratio + mean(context), each mean written relative to the window's first value.
        mapped = (ahead - window[..., :1] - window_mean[..., None]) * ratio[..., None]
        mapped = mapped + (context[:, :, None, :1] + context_mean[:, :, None, None])
        mapped = torch.where(missing[..., None], torch.full_like(mapped, float("nan")), mapped)
        out_distance.append(chosen)
        out_start.append(pick)
        out_continuation.append(mapped)

    def join(parts, trailing):
        return torch.cat(parts, dim=1) if parts else values.new_empty((batch, 0, *trailing))

    distances = join(out_distance, (num_analogues,)).to(dtype)
    starts = (
        join(out_start, (num_analogues,)) if out_start else torch.empty((batch, 0, num_analogues), dtype=torch.int64)
    )
    continuations = join(out_continuation, (num_analogues, horizon)).to(dtype)
    return _restore_batch(distances, ndim), _restore_batch(starts, ndim), _restore_batch(continuations, ndim)
