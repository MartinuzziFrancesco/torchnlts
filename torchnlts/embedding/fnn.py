"""Hard false-nearest-neighbors (FNN) fraction at a given embedding dimension.

Gradient class: none (hard threshold comparisons, boolean output).
A differentiable relaxation would be a separately-named ``soft_fnn``
built on :mod:`torchnlts.neighbors.kernels`, never a hidden swap here --
not yet needed, so not added.

Reference: Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403,
eq. 4-5. A neighbor pair found at embedding dimension ``d`` is false if
adding one more delay coordinate reveals the pair was never really
close: either its distance grows too much relative to itself (eq. 4,
reformulated here as ``high**2 > low**2 * (1 + distance_tol**2)``,
algebraically equivalent to Kennel's ``sqrt(high**2-low**2)/low >
distance_tol`` but division-free, so ``low == 0`` needs no special
case), or it's already large relative to the attractor's overall size
(eq. 5). Defaults (``distance_tol=10, attractor_tol=2``) match
DelayEmbeddings.jl's ``delay_fnn``, validated against it directly. Its
own docstring says "number of" false neighbors, but it actually
normalizes by the point count -- this function returns that same
fraction, not a raw count.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched
from torchnlts.convert import as_tensor
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import _gather_neighbors, knn_search


def hard_fnn(
    trajectory,
    embedding_dim: int,
    delay: int = 1,
    theiler_window: int = 0,
    distance_tol: float = 10.0,
    attractor_tol: float = 2.0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Count of false nearest neighbors at ``embedding_dim``.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        embedding_dim: Embedding dimension to test.
        delay: Lag, forwarded to ``delay_embedding``.
        theiler_window: Theiler window, forwarded to ``knn_search``.
        distance_tol: Threshold on the neighbor's relative distance
            growth (Kennel's eq. 4).
        attractor_tol: Threshold on the neighbor's distance relative to
            the attractor size (Kennel's eq. 5).
        norm_order: Minkowski norm order for both the neighbor search
            and the added-coordinate distance.

    Returns:
        Scalar (or ``[B]``) fraction of false nearest neighbors, in
        ``[0, 1]``, matching DelayEmbeddings.jl's ``delay_fnn`` (its
        docstring says "number of", but it actually normalizes by the
        point count -- confirmed by direct comparison). Points whose
        Theiler window excludes every candidate are excluded from both
        the numerator and denominator.
    """
    trajectory = as_tensor(trajectory)
    embedded_low = delay_embedding(trajectory, embedding_dim, delay)
    embedded_high = delay_embedding(trajectory, embedding_dim + 1, delay)
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

    batched_trajectory, ndim = _ensure_batched(trajectory)
    # Per-channel std, then combined via an L2 norm across channels --
    # not ``norm_order``, since Kennel's R_A is a
    # sum-of-squares attractor size regardless of the neighbor-search
    # norm. Computing std on data flattened across time *and* channel
    # would mix each channel's mean into every other channel's spread.
    per_channel_std = batched_trajectory.std(dim=1, correction=0)
    attractor_size = torch.linalg.vector_norm(per_channel_std, dim=-1)
    attractor_size = attractor_size.unsqueeze(-1) if ndim == 3 else attractor_size.squeeze(0)

    exceeds_relative_growth = high_distance**2 > low_distance**2 * (1 + distance_tol**2)
    exceeds_attractor_fraction = high_distance > attractor_tol * attractor_size
    is_false = (exceeds_relative_growth | exceeds_attractor_fraction) & valid

    return is_false.sum(dim=-1).to(low_distance.dtype) / valid.sum(dim=-1).to(low_distance.dtype)
