"""FLUSS semantic segmentation from a matrix profile index.

Reference: Gharghabi et al. (2017), "Matrix Profile VIII", IEEE ICDM, Tables I-II.
Arc curve ``AC[j]`` = number of nearest-neighbor arcs ``(i, index[i])`` crossing
position ``j`` (``min(i, index[i]) <= j < max(i, index[i])``; entries with
``index = -1`` have no arc). Corrected arc curve
``CAC = min(AC / IAC, 1)`` with the idealized arc curve of a random series
``IAC[j] = 2 j (n - j) / n``; ``CAC`` is set to 1 within ``exclusion_factor * window``
of both ends. Regime boundaries are found by repeatedly taking the
smallest CAC and setting CAC to 1 within ``exclusion_factor * window`` of it.
Differs from stumpy's default, whose IAC is a fitted beta density (stumpy
reproduces this module's CAC with ``custom_iac`` set to the parabola).

Gradient class: none (counts and argmins of integer indices).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int
from torchnlts.matrixprofile._validate import _as_profile_index, _check_series_axis


def _validate(window, exclusion_factor) -> tuple[int, int]:
    window = _as_int(window, "window")
    exclusion_factor = _as_int(exclusion_factor, "exclusion_factor")
    if window < 1 or exclusion_factor < 0:
        raise ValueError(f"need window >= 1 and exclusion_factor >= 0, got {window} and {exclusion_factor}")
    return window, exclusion_factor


def corrected_arc_curve(index, window: int, exclusion_factor: int = 5) -> torch.Tensor:
    """Corrected arc curve (CAC) of a matrix profile index.

    Reference: Gharghabi et al. (2017), IEEE ICDM, Table I.

    Gradient class: none.

    Args:
        index: Matrix profile index of a self-join, shape ``[n]`` or ``[B, n]``;
            ``-1`` entries add no arc.
        window: Length scale (about one period) for the edge exclusion zones.
        exclusion_factor: CAC is 1 within ``exclusion_factor * window`` of both ends;
            ``0`` disables this.

    Returns:
        CAC in ``[0, 1]``, same shape as ``index``, default floating dtype.
        Low values mark likely regime boundaries.

    Raises:
        ValueError: If ``index`` is not an integer tensor of shape ``[n]`` or ``[B, n]``
            (``n >= 1``) with entries in ``[-1, n - 1]``, or ``window``/``exclusion_factor`` is invalid.
    """
    window, exclusion_factor = _validate(window, exclusion_factor)
    index = _as_profile_index(index)
    flat = index.reshape(-1, index.shape[-1])
    num_points = flat.shape[-1]
    position = torch.arange(num_points, device=flat.device).expand_as(flat)
    neighbor = torch.where(flat < 0, position, flat)
    low, high = torch.minimum(position, neighbor), torch.maximum(position, neighbor)
    # Out of place (vmap-compatible): +1 at each arc's low end, -1 at its high end.
    counts = torch.zeros(flat.shape[0], num_points + 1, dtype=torch.int64, device=flat.device)
    counts = counts.scatter_add(1, low, torch.ones_like(low)).scatter_add(1, high, -torch.ones_like(high))
    arcs = counts.cumsum(dim=-1)[:, :num_points].to(torch.get_default_dtype())
    ideal = 2 * position[:1].to(arcs.dtype) * (num_points - position[:1]) / num_points
    # Zero denominator as in stumpy: 1e-10, so zero arcs give 0 and any arc saturates at 1.
    curve = (arcs / torch.where(ideal > 0, ideal, torch.full_like(ideal, 1e-10))).clamp(max=1.0)
    edge = min(window * exclusion_factor, num_points)
    outside = (position < edge) | (position >= num_points - edge)
    curve = torch.where(outside, torch.ones_like(curve), curve)
    return curve.reshape(index.shape)


def regime_boundaries(arc_curve, window: int, num_regimes: int = 2, exclusion_factor: int = 5) -> torch.Tensor:
    """Regime boundary positions from a corrected arc curve.

    Reference: Gharghabi et al. (2017), IEEE ICDM, Table II (regime extracting algorithm).

    Gradient class: none.

    Args:
        arc_curve: Corrected arc curve, shape ``[n]`` or ``[B, n]``.
        window: Length scale of the exclusion zone around each boundary found.
        num_regimes: Number of regimes; ``num_regimes - 1`` boundaries are returned.
        exclusion_factor: CAC is set to 1 within ``exclusion_factor * window`` of each boundary.

    Returns:
        int64 boundary positions in the order found (lowest CAC first), shape
        ``[num_regimes - 1]`` or ``[B, num_regimes - 1]``. A boundary with no
        evidence (the smallest remaining CAC is 1, e.g. all ones or everything
        already cleared) is ``-1``.
    """
    window, exclusion_factor = _validate(window, exclusion_factor)
    num_regimes = _as_int(num_regimes, "num_regimes")
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2, got {num_regimes}")
    curve = torch.as_tensor(arc_curve)
    _check_series_axis(curve, "arc_curve")
    remaining = curve.detach().reshape(-1, curve.shape[-1]).clone()
    position = torch.arange(remaining.shape[-1], device=remaining.device)
    found = []
    for _ in range(num_regimes - 1):
        best = remaining.argmin(dim=-1)
        found.append(torch.where(remaining.amin(dim=-1) < 1.0, best, torch.full_like(best, -1)))
        within = (position[None, :] >= best[:, None] - exclusion_factor * window) & (
            position[None, :] < best[:, None] + exclusion_factor * window
        )
        remaining = remaining.masked_fill(within, 1.0)
    boundaries = torch.stack(found, dim=-1)
    return boundaries[0] if curve.dim() == 1 else boundaries
