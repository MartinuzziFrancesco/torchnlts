"""Hard k-nearest-neighbor search over trajectory points.

Reference: Kantz & Schreiber, "Nonlinear Time Series Analysis", 2nd ed., Cambridge University Press, 2004, Ch. 3.

Gradient class: piecewise.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import apply_theiler_window


def knn_search(
    trajectory: torch.Tensor,
    num_neighbors: int,
    *,
    theiler_window: int = 0,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``num_neighbors`` nearest neighbors of every point of a trajectory.

    Reference: Kantz & Schreiber (2004), Ch. 3.

    Gradient class: piecewise (distances differentiate normally; which points are selected is discrete).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        num_neighbors: Neighbors per point, an integer ``>= 1``.
        theiler_window: Theiler window, applied only for a self-search (``other is None``).
        other: Optional second trajectory to search into, same convention as ``trajectory``. Default: a self-search.
        norm_order: Minkowski norm order, forwarded to :func:`~torchnlts.neighbors.pairwise_distance`.

    Returns:
        ``(distances, indices)``, each of shape ``[..., T, num_neighbors]``, sorted ascending by distance. A slot
        with no valid candidate (``num_neighbors`` exceeds the available points, the Theiler window excludes them,
        or the point has a NaN/inf coordinate) has distance ``inf`` and index ``-1``.

    Raises:
        ValueError: If ``num_neighbors`` is not an integer ``>= 1``, ``theiler_window`` is not an integer (self-search
            only), ``norm_order`` is invalid, or the trajectories do not match in shape.
    """
    num_neighbors = _as_int(num_neighbors, "num_neighbors")
    if num_neighbors < 1:
        raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    if other is None:
        distance = apply_theiler_window(distance, theiler_window, fill_value=float("inf"))
    # Pairs involving a non-finite point are NaN; rank them as unavailable.
    distance = torch.where(torch.isnan(distance), float("inf"), distance)
    num_points = distance.shape[-1]
    if num_neighbors > num_points:
        distance = torch.nn.functional.pad(distance, (0, num_neighbors - num_points), value=float("inf"))
    values, indices = torch.topk(distance, num_neighbors, dim=-1, largest=False, sorted=True)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    return values, indices


def _gather_neighbors(target: torch.Tensor, index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Rows of ``target`` ``[..., T, D]`` at ``index`` ``[..., T]``, shape ``[..., T, D]``.

    Where ``valid`` ``[..., T]`` is ``False`` (``knn_search``'s ``-1`` sentinel), row 0 is gathered instead, so
    callers must mask the result themselves.
    """
    safe_index = torch.where(valid, index, torch.zeros_like(index))
    return torch.take_along_dim(target, safe_index.unsqueeze(-1), dim=-2)
