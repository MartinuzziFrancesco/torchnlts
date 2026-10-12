"""Hard false-nearest-neighbors (FNN) fraction at a given embedding dimension.

Reference: Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403, eq. 4-5. A neighbor pair found in dimension ``d``
is false if adding one more delay coordinate either grows its distance by more than the factor ``distance_tol``
(eq. 4) or makes it larger than ``attractor_tol`` times the attractor size (eq. 5).

Gradient class: none (hard threshold comparisons, boolean output).

Conventions:

- Eq. 4 is evaluated as ``high**2 > low**2 * (1 + distance_tol**2)``, equivalent to Kennel's
  ``sqrt(high**2 - low**2) / low > distance_tol`` without a division, so ``low == 0`` needs no special case.
- The defaults (``distance_tol=10``, ``attractor_tol=2``) and the returned fraction match DelayEmbeddings.jl's
  ``delay_fnn``, whose docstring says "number of" but which divides by the point count.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _restore_batch,
    _work_dtype,
)
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import _gather_neighbors, knn_search


def hard_fnn(
    trajectory,
    embedding_dim: int,
    *,
    delay: int = 1,
    theiler_window: int = 0,
    distance_tol: float = 10.0,
    attractor_tol: float = 2.0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Fraction of false nearest neighbors at ``embedding_dim``.

    Reference: Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403, eq. 4-5.

    Gradient class: none (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        embedding_dim: Dimension ``d >= 1`` to test; ``d`` and ``d + 1`` are embedded.
        delay: Lag, forwarded to :func:`~torchnlts.embedding.delay_embedding`.
        theiler_window: Theiler window of the neighbor search (``|i - j| <= theiler_window`` is excluded).
        distance_tol: Threshold on the relative distance growth (eq. 4).
        attractor_tol: Threshold on the distance relative to the attractor size (eq. 5), the L2 norm of the
            per-channel standard deviations.
        norm_order: Minkowski norm order, ``>= 0``, for the neighbor search and the distances.

    Returns:
        Fraction in ``[0, 1]`` of the points whose nearest neighbor is false, shape ``[]`` (or ``[B]``), in the
        input's floating dtype (the default dtype for integer input). Points whose Theiler window excludes every
        candidate count in neither the numerator nor the denominator. ``NaN`` for a row containing NaN/inf, and
        when every point is excluded.

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an
            integer, ``norm_order`` is invalid, the input is complex, or the trajectory is too short.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    values, nonfinite = _mask_nonfinite_rows(batched.to(_work_dtype(batched.dtype)))
    embedded_low = delay_embedding(values, embedding_dim, delay=delay)
    embedded_high = delay_embedding(values, embedding_dim + 1, delay=delay)
    num_common = embedded_high.shape[-2]
    embedded_low = embedded_low[..., :num_common, :]

    low_distance, neighbor_index = knn_search(
        embedded_low, num_neighbors=1, theiler_window=theiler_window, norm_order=norm_order
    )
    low_distance = low_distance[..., 0]
    neighbor_index = neighbor_index[..., 0]
    valid = neighbor_index != -1

    neighbor_points = _gather_neighbors(embedded_high, neighbor_index, valid)
    high_distance = torch.linalg.vector_norm(embedded_high - neighbor_points, ord=norm_order, dim=-1)

    # Per-channel std combined by an L2 norm, not norm_order: Kennel's R_A is a sum-of-squares size,
    # and flattening across channels would mix each channel's mean into the spread.
    attractor_size = torch.linalg.vector_norm(values.std(dim=1, correction=0), dim=-1, keepdim=True)  # [B, 1]

    exceeds_relative_growth = high_distance**2 > low_distance**2 * (1 + distance_tol**2)
    exceeds_attractor_fraction = high_distance > attractor_tol * attractor_size
    is_false = (exceeds_relative_growth | exceeds_attractor_fraction) & valid

    fraction = is_false.sum(dim=-1).to(low_distance.dtype) / valid.sum(dim=-1).to(low_distance.dtype)
    return _restore_batch(torch.where(nonfinite, float("nan"), fraction).to(out_dtype), ndim)
