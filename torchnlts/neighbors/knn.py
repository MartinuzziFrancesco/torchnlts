"""Hard k-nearest-neighbor search over trajectory points.

Gradient class: piecewise (distance values differentiate normally;
which points are selected is a discrete, non-differentiable decision).
Reference: Kantz & Schreiber, "Nonlinear Time Series Analysis", 2nd ed.,
Cambridge University Press, 2004, Ch. 3.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import apply_theiler_window


def knn_search(
    trajectory: torch.Tensor,
    num_neighbors: int,
    theiler_window: int = 0,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest neighbors of every point in a trajectory.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        num_neighbors: Number of neighbors to return per point.
        theiler_window: Theiler window, applied only for the self-search
            case (``other is None``).
        other: Optional second trajectory to search into, same
            convention as ``trajectory``. Defaults to a self-search.
        norm_order: Minkowski norm order, forwarded to ``pairwise_distance``.

    Returns:
        ``(distances, indices)``, each of shape ``[..., T, num_neighbors]``,
        sorted ascending by distance. Slots with no valid candidate
        (``num_neighbors`` exceeds what's available, or Theiler-excluded)
        come back with distance ``inf`` and index ``-1``.
    """
    distance = pairwise_distance(trajectory, other, norm_order=norm_order)
    if other is None:
        distance = apply_theiler_window(distance, theiler_window, fill_value=float("inf"))
    num_points = distance.shape[-1]
    if num_neighbors > num_points:
        distance = torch.nn.functional.pad(distance, (0, num_neighbors - num_points), value=float("inf"))
    values, indices = torch.topk(distance, num_neighbors, dim=-1, largest=False, sorted=True)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    return values, indices
