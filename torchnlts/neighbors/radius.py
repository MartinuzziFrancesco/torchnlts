"""Hard radius (epsilon-ball) neighbor search.

Gradient class: **none**. A hard inclusion test is a Heaviside
indicator, ``|X_i - X_j| <= r``; it produces a bool tensor with no
gradient (undefined at the ``d == r`` boundary, zero elsewhere) and is
not worth relaxing here. It is exactly the building block of the
Grassberger & Procaccia (1983) correlation sum (their eq. 1, the
Heaviside step ``Theta(r - |X_i - X_j|)``, inclusive at ``d == r``); a
differentiable relaxation belongs in a separately-named function built
on :mod:`torchnlts.neighbors.kernels` (``TrajectoryKernel``), never a
hidden smoothing of this one.
"""

from __future__ import annotations

import torch

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import theiler_window_mask


def radius_neighbors(
    x: torch.Tensor,
    r: float,
    theiler: int = 0,
    y: torch.Tensor | None = None,
    p: float = 2.0,
) -> torch.Tensor:
    """Boolean neighbor mask for points within radius ``r``.

    Args:
        x: Trajectory of shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        r: Inclusion radius (``|X_i - X_j| <= r`` counts as a neighbor).
        theiler: Theiler window, applied only for the self-search case
            (``y is None``); see :mod:`torchnlts.neighbors.theiler`.
        y: Optional second trajectory to search into, same convention
            as ``x``. Defaults to ``x`` (self-search).
        p: Minkowski norm order, forwarded to ``pairwise_distance``.

    Returns:
        Bool tensor of shape ``[..., T_x, T_y]``, ``True`` where the
        pair is within radius ``r`` (and not Theiler-excluded).
    """
    d = pairwise_distance(x, y, p=p)
    mask = d <= r
    if y is None:
        # AND the Theiler mask in directly rather than filling excluded
        # entries with inf and comparing: inf <= inf is True in IEEE 754,
        # so that approach silently un-excludes Theiler pairs at r=inf.
        mask = mask & theiler_window_mask(d.shape[-1], theiler, device=d.device)
    return mask
