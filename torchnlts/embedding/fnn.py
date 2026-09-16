"""Hard false-nearest-neighbors (FNN) count at a given embedding dimension.

Gradient class: none (hard threshold comparisons, boolean output).
A differentiable relaxation would be a separately-named ``soft_fnn``
built on :mod:`torchnlts.neighbors.kernels`, never a hidden swap here.
Not yet needed, so not added.

Reference: Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403,
eq. 4-5. A neighbor pair found at embedding dimension ``d`` is false if
adding one more delay coordinate reveals the pair was never really
close: either its distance grows too much relative to itself (eq. 4,
reformulated here as ``high**2 > low**2 * (1 + distance_tol**2)``,
algebraically equivalent to Kennel's ``sqrt(high**2-low**2)/low >
distance_tol`` but division-free, so ``low == 0`` needs no special
case), or it's already large relative to the attractor's overall size
(eq. 5). Defaults (``distance_tol=10, attractor_tol=2``) and the
"count, not fraction" return convention match DelayEmbeddings.jl's
``delay_fnn``, validated against it directly.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched
from torchnlts.convert import as_tensor
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import knn_search


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
    safe_index = torch.where(valid, neighbor_index, torch.zeros_like(neighbor_index))

    index_expanded = safe_index.unsqueeze(-1).expand(*safe_index.shape, embedded_high.shape[-1])
    neighbor_points = torch.gather(embedded_high, dim=-2, index=index_expanded)
    high_distance = torch.linalg.vector_norm(embedded_high - neighbor_points, ord=norm_order, dim=-1)

    batched_trajectory, ndim = _ensure_batched(trajectory)
    attractor_size = batched_trajectory.reshape(batched_trajectory.shape[0], -1).std(dim=-1, correction=0)
    attractor_size = attractor_size.unsqueeze(-1) if ndim == 3 else attractor_size.squeeze(0)

    exceeds_relative_growth = high_distance**2 > low_distance**2 * (1 + distance_tol**2)
    exceeds_attractor_fraction = high_distance > attractor_tol * attractor_size
    is_false = (exceeds_relative_growth | exceeds_attractor_fraction) & valid

    return is_false.sum(dim=-1).to(low_distance.dtype) / valid.sum(dim=-1).to(low_distance.dtype)
