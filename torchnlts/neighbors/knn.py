"""Hard k-nearest-neighbor search over trajectory points.

Gradient class: **piecewise**. Autodiff flows correctly through the
*distance values* of the selected neighbors (``torch.topk`` gradients
route to the selected entries, zero elsewhere), but which points are
selected is a discrete decision: at a Theiler-window boundary or a tie
between the k-th and (k+1)-th nearest neighbor, an infinitesimal change
in the trajectory can swap which point is "in" the neighborhood, causing
a jump discontinuity rather than a smooth change. Use
:mod:`torchnlts.neighbors.soft_knn` (not yet implemented) where a
differentiable neighbor weighting is required through that boundary.

Standard phase-space neighbor search as used throughout nonlinear time
series analysis (e.g. Kantz & Schreiber, "Nonlinear Time Series
Analysis", 2nd ed., Cambridge University Press, 2004, Ch. 3), combining
:func:`torchnlts.neighbors.distance.pairwise_distance` and
:func:`torchnlts.neighbors.theiler.apply_theiler_window`.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import apply_theiler_window


def knn_search(
    x: torch.Tensor,
    k: int,
    theiler: int = 0,
    y: torch.Tensor | None = None,
    p: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest neighbors of every point in a trajectory.

    Args:
        x: Trajectory of shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        k: Number of neighbors to return per point.
        theiler: Theiler window (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
            Only applied for the self-search case (``y is None``); a
            window over two different point sets is not meaningful.
        y: Optional second trajectory to search into, same convention
            as ``x``. Defaults to ``x`` (self-search).
        p: Minkowski norm order, forwarded to ``pairwise_distance``.

    Returns:
        ``(distances, indices)``, each of shape ``[..., T, k]``, sorted
        ascending by distance. Excluded/out-of-range neighbors (when
        ``k`` exceeds the number of valid candidates after the Theiler
        window, or exceeds the number of points being searched) come
        back with distance ``inf`` and index ``-1``.
    """
    d = pairwise_distance(x, y, p=p)
    if y is None:
        d = apply_theiler_window(d, theiler, fill_value=float("inf"))
    t = d.shape[-1]
    if k > t:
        d = torch.nn.functional.pad(d, (0, k - t), value=float("inf"))
    values, indices = torch.topk(d, k, dim=-1, largest=False, sorted=True)
    # Any non-finite slot has no valid candidate -- whether because k
    # exceeds the number of points (padding) or because every remaining
    # candidate was Theiler-excluded -- so its index is meaningless.
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    return values, indices
