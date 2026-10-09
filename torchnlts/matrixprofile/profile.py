"""Matrix profile of a scalar series: for each length-``window`` subsequence, the
z-normalized Euclidean distance to its nearest non-trivial neighbor, and that
neighbor's start index.

Reference: Yeh et al. (2016), "Matrix Profile I", IEEE ICDM (distance, profile,
profile index); Zhu et al. (2016), "Matrix Profile II" (STOMP). A subsequence is
trivially matched to starts within ``exclusion_zone`` of its own
(``|i - j| <= exclusion_zone``). Yeh et al. use ``window / 2``; the default
``ceil(window / 4)`` follows stumpy. The
distance is ``||z(a) - z(b)||``, ``z(a) = (a - mean(a)) / std(a)`` (population
std), equal to ``sqrt(2 * window * (1 - rho))`` for Pearson correlation ``rho``.
A constant subsequence (window max == min exactly, as in stumpy; scale
invariant) is mapped to ``z = 0``: its distance is
0 to another constant subsequence and ``sqrt(window)`` to any other (stumpy's
convention). A subsequence containing NaN/inf is unavailable: it is nobody's
neighbor, and its own profile is ``inf`` with index ``-1``.

Limitation: the full ``[n, n]`` distance matrix is formed, O(T^2) memory and
O(T^2 * window) time; STOMP's O(T) memory diagonal recurrence is the upgrade
path.

Gradient class: piecewise (the minimum over distances, which are smooth in the
series; kinks at ties and at constant subsequences). The index is none.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _output_dtype,
    _restore_batch,
    _validate_matching_batch,
    _work_dtype,
)
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import apply_theiler_window


def _znormalized_windows(values: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding z-normalized windows ``[B, n, window]`` of ``values`` ``[B, T]``
    and the ``[B, n]`` mask of windows that are finite. Constant and non-finite
    windows are zero (finite, so they cannot put NaN into backward)."""
    windows = values.unfold(-1, window, 1)
    finite = torch.isfinite(windows).all(dim=-1)
    windows = torch.where(finite[..., None], windows, torch.zeros_like(windows))
    constant = (windows.amax(dim=-1) == windows.amin(dim=-1))[..., None]
    # Per-window power-of-two scaling (exact) and a shift by the first value (differences of nearby
    # values are exact) before the mean: z-normalization is invariant to both, and neither overflows
    # near the dtype's maximum nor loses precision for data with a large offset.
    magnitude = windows.detach().abs().amax(dim=-1, keepdim=True)
    magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    # frexp's exponent keeps the scale finite even at the dtype's maximum (log2 would round up to overflow).
    windows = torch.ldexp(windows, -torch.frexp(magnitude)[1])
    windows = windows - windows[..., :1]
    centered = windows - windows.mean(dim=-1, keepdim=True)
    # Dividing by the (detached) peak first keeps the variance away from under/overflow at any scale.
    peak = centered.detach().abs().amax(dim=-1, keepdim=True)
    constant = constant | (peak == 0)
    scaled = centered / torch.where(constant, torch.ones_like(peak), peak)
    variance = (scaled * scaled).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(torch.where(constant, torch.ones_like(variance), variance))
    return torch.where(constant, torch.zeros_like(scaled), scaled / scale), finite


def _validate_window(window, num_points: int) -> int:
    window = _as_int(window, "window")
    if window < 2 or window > num_points:
        raise ValueError(f"window must be in [2, T={num_points}], got {window}")
    return window


def matrix_profile(
    trajectory, window: int, exclusion_zone: int | None = None, other=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrix profile and matrix profile index of a scalar series.

    Reference: Yeh et al. (2016), IEEE ICDM (Matrix Profile I).

    Gradient class: piecewise (see module docstring); the index has none.

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        window: Subsequence length ``m``, in ``[2, T]``.
        exclusion_zone: Self-join only: starts with ``|i - j| <= exclusion_zone``
            are not neighbors. Default ``ceil(window / 4)``; 0 excludes only
            ``j = i``.
        other: Optional second series (same batching as ``trajectory``). If
            given, an AB-join: each subsequence of ``trajectory`` is matched
            against all subsequences of ``other`` (no exclusion zone;
            passing ``exclusion_zone`` too is an error).

    Returns:
        ``(profile, index)``, each of shape ``[n]`` or ``[B, n]`` with
        ``n = T - window + 1``. ``profile`` is in the input's floating dtype,
        ``index`` is int64 and indexes subsequences of ``other`` for an
        AB-join. Entries with no available neighbor (no candidate outside the
        zone, or a subsequence containing NaN/inf) are ``inf`` and ``-1``.

    Raises:
        ValueError: If ``exclusion_zone`` is given with ``other``, the feature dim is not 1, the input is complex,
            ``window`` is out of range, or ``exclusion_zone`` is negative.
    """
    tensor, ndim = _ensure_batched(trajectory)
    dtype = _output_dtype(tensor)
    if tensor.shape[-1] != 1:
        raise ValueError(f"matrix_profile only supports a univariate (D=1) series, got feature dim {tensor.shape[-1]}")
    window = _validate_window(window, tensor.shape[-2])
    work_dtype = _work_dtype(tensor.dtype)
    queries, query_finite = _znormalized_windows(tensor.squeeze(-1).to(work_dtype), window)
    if other is None:
        if exclusion_zone is None:
            exclusion_zone = math.ceil(window / 4)
        exclusion_zone = _as_int(exclusion_zone, "exclusion_zone")
        if exclusion_zone < 0:
            raise ValueError(f"exclusion_zone must be >= 0, got {exclusion_zone}")
        candidates, candidate_finite = queries, query_finite
        distance = pairwise_distance(queries, candidates)
        distance = apply_theiler_window(distance, exclusion_zone)
    else:
        if exclusion_zone is not None:
            raise ValueError("exclusion_zone applies to a self-join only; do not pass it with other")
        other_tensor, other_ndim = _ensure_batched(other)
        if other_tensor.is_complex():
            raise ValueError(f"complex input is not supported, got {other_tensor.dtype}")
        _validate_matching_batch(tensor, other_tensor, ndim, other_ndim)
        window = _validate_window(window, other_tensor.shape[-2])
        candidates, candidate_finite = _znormalized_windows(other_tensor.squeeze(-1).to(work_dtype), window)
        distance = pairwise_distance(queries, candidates)
    available = query_finite[..., :, None] & candidate_finite[..., None, :]
    distance = torch.where(available, distance, torch.full_like(distance, float("inf")))
    profile, index = distance.min(dim=-1)
    index = torch.where(torch.isfinite(profile), index, torch.full_like(index, -1))
    return _restore_batch(profile.to(dtype), ndim), _restore_batch(index, ndim)
