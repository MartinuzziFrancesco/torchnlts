"""Pairwise distance between points of a (batch of) trajectory.

Gradient class: **exact** (ordinary PyTorch autodiff via ``torch.cdist``).
The Minkowski p-norm is non-differentiable at zero distance (coincident
points), so the gradient at ``x_i == x_j`` is a subgradient, not a true
derivative; this only affects the diagonal of a self-distance matrix and
any pair of exactly-coincident points. See :func:`test_pairwise_distance`
in ``tests/test_distance.py`` for the gradcheck, which is computed on
distinct points to avoid that singularity.

This is the base primitive every downstream measure in torchnlts is built
on: Grassberger & Procaccia (1983), "Characterization of strange
attractors", Physica D 9(1-2), define the correlation sum directly in
terms of pairwise Euclidean distances between delay-embedded vectors
(their eq. 1-2, ``theta(r - |X_i - X_j|)``); Theiler-window masking,
k-nearest-neighbor search, and kernel-weighted (soft) neighborhoods all
consume this same distance matrix.
"""

from __future__ import annotations

import torch

from torchnlts._shape import ensure_batched, restore_batch


def pairwise_distance(x: torch.Tensor, y: torch.Tensor | None = None, p: float = 2.0) -> torch.Tensor:
    """Pairwise Minkowski distance between points of a trajectory.

    Args:
        x: Trajectory of shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        y: Optional second trajectory, same shape convention as ``x``
            (and same batch size ``B`` if batched). Defaults to ``x``,
            giving the self-distance matrix.
        p: Minkowski norm order (``2`` = Euclidean, ``1`` = Manhattan,
            ``float('inf')`` = Chebyshev/sup norm).

    Returns:
        Distance matrix of shape ``[T_x, T_y]`` (unbatched input) or
        ``[B, T_x, T_y]`` (batched input).
    """
    xb, ndim = ensure_batched(x)
    yb = xb if y is None else ensure_batched(y)[0]
    # torch.cdist's default compute_mode switches p=2 to a matmul-based
    # formula (||x||^2 + ||y||^2 - 2xy) once T > 25; that form catastrophically
    # cancels in float32 for points sharing a large offset (e.g. a trajectory
    # far from the origin), corrupting nearby-point distances. Force the
    # direct (non-matmul) computation so correctness doesn't depend on T.
    d = torch.cdist(xb, yb, p=p, compute_mode="donot_use_mm_for_euclid_dist")
    return restore_batch(d, ndim)
