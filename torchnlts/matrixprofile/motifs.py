"""Top-k motifs and discords of a self-join matrix profile.

Reference: Yeh et al. (2016), IEEE ICDM (Matrix Profile I): a motif is the
pair with the smallest profile value (subsequence ``i`` and its neighbor
``index[i]``), a discord the subsequence with the largest. After each pick,
starts within ``exclusion_zone`` of the picked starts are removed from the
candidates for later picks (stumpy.motifs with ``max_matches=2``); a removed
start can still appear as a later pair's neighbor. The default
``ceil(window / 4)`` follows stumpy; Yeh et al. use ``window / 2``.

Gradient class: none (selection of indices; values are the profile's).
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _as_int
from torchnlts.matrixprofile._validate import _as_profile_index, _check_series_axis


def _zone_mask(num_points: int, center: torch.Tensor, exclusion_zone: int) -> torch.Tensor:
    """``[B, n]`` mask of starts with ``|start - center| <= exclusion_zone`` for ``center`` ``[B]``."""
    start = torch.arange(num_points, device=center.device)
    return (start[None, :] - center[:, None]).abs() <= exclusion_zone


def _prepare(profile, window: int, num_picks: int, exclusion_zone: int | None):
    profile = torch.as_tensor(profile)
    if not profile.is_floating_point():
        profile = profile.to(torch.get_default_dtype())
    _check_series_axis(profile, "profile")
    num_picks = _as_int(num_picks, "num_picks")
    if num_picks < 1:
        raise ValueError(f"num_picks must be >= 1, got {num_picks}")
    window = _as_int(window, "window")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if exclusion_zone is None:
        exclusion_zone = math.ceil(window / 4)
    exclusion_zone = _as_int(exclusion_zone, "exclusion_zone")
    if exclusion_zone < 0:
        raise ValueError(f"exclusion_zone must be >= 0, got {exclusion_zone}")
    return profile.detach().reshape(-1, profile.shape[-1]), num_picks, exclusion_zone


def motifs(profile, index, window: int, *, num_picks: int = 1, exclusion_zone: int | None = None):
    """Top motif pairs of a self-join matrix profile, most similar first.

    Reference: Yeh et al. (2016), IEEE ICDM (Matrix Profile I).

    Gradient class: none.

    Args:
        profile: Shape ``[n]`` or ``[B, n]``, from
            :func:`~torchnlts.matrixprofile.matrix_profile` (self-join).
        index: Matrix profile index, same shape as ``profile``.
        window: Subsequence length the profile was computed with.
        num_picks: Number of motifs to return.
        exclusion_zone: Starts within this distance of a picked pair's two
            starts are removed. Default ``ceil(window / 4)``.

    Returns:
        ``(distances, pairs)``: ``distances`` of shape ``[num_picks]`` (or
        ``[B, num_picks]``) and int64 ``pairs`` of shape ``[num_picks, 2]``
        (or ``[B, num_picks, 2]``) holding ``(i, index[i])``. Missing picks
        (no finite profile value left) are ``nan`` and ``-1``.

    Raises:
        ValueError: If ``profile`` is not of shape ``[n]`` or ``[B, n]`` (``n >= 1``), ``index`` is not an integer
            tensor of the same shape with entries in ``[-1, n - 1]``, ``window`` or ``num_picks`` is not an integer
            ``>= 1``, or ``exclusion_zone`` is not an integer ``>= 0``.
    """
    original = torch.as_tensor(profile)
    flat, num_picks, exclusion_zone = _prepare(profile, window, num_picks, exclusion_zone)
    flat_index = _as_profile_index(index).to(flat.device)
    if flat_index.shape != original.shape:
        raise ValueError(f"index shape {tuple(flat_index.shape)} must match profile shape {tuple(original.shape)}")
    flat_index = flat_index.reshape(flat.shape)
    remaining = torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float("inf")))
    rows = torch.arange(flat.shape[0], device=flat.device)
    distances, pairs = [], []
    for _ in range(num_picks):
        best = remaining.argmin(dim=-1)
        value = remaining[rows, best]
        valid = torch.isfinite(value)
        neighbor = flat_index[rows, best]
        distances.append(torch.where(valid, value, torch.full_like(value, float("nan"))))
        pairs.append(torch.where(valid[:, None], torch.stack([best, neighbor], dim=-1), -1))
        removed = _zone_mask(flat.shape[-1], best, exclusion_zone) | _zone_mask(
            flat.shape[-1], neighbor, exclusion_zone
        )
        remaining = remaining.masked_fill(removed & valid[:, None], float("inf"))
    distances = torch.stack(distances, dim=-1)
    pairs = torch.stack(pairs, dim=-2)
    if original.dim() == 1:
        return distances[0], pairs[0]
    return distances, pairs


def discords(profile, window: int, *, num_picks: int = 1, exclusion_zone: int | None = None):
    """Top discords (largest finite profile values), most anomalous first.

    Reference: Yeh et al. (2016), IEEE ICDM (Matrix Profile I).

    Gradient class: none.

    Args:
        profile: Shape ``[n]`` or ``[B, n]``. Non-finite entries are skipped.
        window: Subsequence length the profile was computed with.
        num_picks: Number of discords to return.
        exclusion_zone: Starts within this distance of a picked start are
            removed. Default ``ceil(window / 4)``.

    Returns:
        ``(values, indices)``, each of shape ``[num_picks]`` (or
        ``[B, num_picks]``); indices are int64. Missing picks are ``nan`` and ``-1``.

    Raises:
        ValueError: If ``profile`` is not of shape ``[n]`` or ``[B, n]`` (``n >= 1``), ``window`` or ``num_picks`` is
            not an integer ``>= 1``, or ``exclusion_zone`` is not an integer ``>= 0``.
    """
    original = torch.as_tensor(profile)
    flat, num_picks, exclusion_zone = _prepare(profile, window, num_picks, exclusion_zone)
    remaining = torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float("-inf")))
    rows = torch.arange(flat.shape[0], device=flat.device)
    values, indices = [], []
    for _ in range(num_picks):
        best = remaining.argmax(dim=-1)
        value = remaining[rows, best]
        valid = torch.isfinite(value)
        values.append(torch.where(valid, value, torch.full_like(value, float("nan"))))
        indices.append(torch.where(valid, best, torch.full_like(best, -1)))
        removed = _zone_mask(flat.shape[-1], best, exclusion_zone)
        remaining = remaining.masked_fill(removed & valid[:, None], float("-inf"))
    values = torch.stack(values, dim=-1)
    indices = torch.stack(indices, dim=-1)
    if original.dim() == 1:
        return values[0], indices[0]
    return values, indices
