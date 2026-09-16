"""Hard radius (epsilon-ball) neighbor search.

Gradient class: none (hard Heaviside indicator, no gradient).
Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), eq. 1;
differentiable relaxation lives separately in
:mod:`torchnlts.neighbors.kernels`.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import theiler_window_mask


def radius_neighbors(
    trajectory: torch.Tensor,
    radius: float,
    theiler_window: int = 0,
    other: torch.Tensor | None = None,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Boolean neighbor mask for points within ``radius``.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        radius: Inclusion radius, ``<=`` (inclusive).
        theiler_window: Theiler window, applied only for the self-search
            case (``other is None``).
        other: Optional second trajectory to search into, same
            convention as ``trajectory``. Defaults to a self-search.
        norm_order: Minkowski norm order, forwarded to ``pairwise_distance``.

    Returns:
        Bool tensor of shape ``[..., T, T']``.
    """
    distance = pairwise_distance(trajectory, other, norm_order=norm_order)
    mask = distance <= radius
    if other is None:
        # AND the Theiler mask in directly instead of filling excluded
        # entries with inf and comparing: inf <= inf is True in IEEE 754.
        mask = mask & theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    return mask
