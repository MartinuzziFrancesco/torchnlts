"""Hard radius (epsilon-ball) neighbor search.

Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1.

Gradient class: none (Heaviside indicator).
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import theiler_window_mask


def radius_neighbors(
    trajectory: torch.Tensor,
    radius: float,
    *,
    theiler_window: int = 0,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Boolean neighbor mask for points within ``radius``.

    Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1. The smooth relaxation is
    :func:`~torchnlts.neighbors.trajectory_kernel`.

    Gradient class: none (Heaviside indicator).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        radius: Inclusion radius; a pair is a neighbor when its distance is ``<= radius``.
        theiler_window: Theiler window, applied only for a self-search (``other is None``).
        other: Optional second trajectory to search into, same convention as ``trajectory``. Default: a self-search.
        norm_order: Minkowski norm order, forwarded to :func:`~torchnlts.neighbors.pairwise_distance`.

    Returns:
        Bool tensor of shape ``[..., T, T']``; ``False`` for every pair involving a point with a NaN/inf coordinate.

    Raises:
        ValueError: If ``theiler_window`` is not an integer (self-search only), ``norm_order`` is invalid, or the
            trajectories do not match in shape.
    """
    distance = pairwise_distance(trajectory, other=other, norm_order=norm_order)
    mask = distance <= radius
    if other is None:
        # AND the Theiler mask in directly: filling with inf and comparing fails because inf <= inf is True.
        mask = mask & theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    return mask
