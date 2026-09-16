"""Cao's method for the minimum embedding dimension of a scalar time series.

Gradient class: none (nearest-neighbor search is a discrete decision).

Reference: Cao, L. (1997), "Practical method for determining the
minimum embedding dimension of a scalar time series", Physica D 110,
43-50, eq. 1-3 (``cao_e1``) and eq. 5 (``cao_e2``). ``cao_e1`` saturates
near 1 once the embedding dimension is large enough (find the smallest
``embedding_dim`` where it stops changing); ``cao_e2`` distinguishes a
deterministic signal from noise (which stays away from 1 far longer,
as ``embedding_dim`` grows, for noise -- it does *not`` sit near 1 at
small ``embedding_dim`` even for pure noise, an easy-to-misread finite-
sample property of eq. 5 itself, not a bug).

``cao_e1`` matches DelayEmbeddings.jl's ``delay_afnn`` exactly,
including two of its conventions that aren't obvious from Cao's paper
alone (found by reading its source, not just its docstring): a
coincident nearest neighbor (distance 0) falls back to the *second*
nearest neighbor rather than being excluded, and the average divides by
``num_common - 1``, not ``num_common``.

``cao_e2`` does *not* match DelayEmbeddings.jl's ``stochastic_indicator``:
that function's denominator is the embedded length at each dimension
rather than the number of terms actually summed (an asymmetric
tree/query truncation leaves those two counts different), which looks
like an implementation quirk rather than a deliberate reading of eq. 5.
``cao_e2`` instead follows eq. 5
directly and is validated against an independent brute-force transcription
of it plus the paper's own random-vs-deterministic qualitative claim.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import knn_search


def _cao_step(
    trajectory,
    embedding_dim: int,
    delay: int,
    theiler_window: int,
    norm_order: float,
    new_coordinate_only: bool,
) -> torch.Tensor:
    """Average ratio comparing each point's nearest-neighbor distance at
    ``embedding_dim`` against the same pair's distance at
    ``embedding_dim + 1``: the numerator of Cao's E(d) (full vector norm,
    matching ``delay_afnn``) or E*(d) (only the newly added coordinate,
    ``new_coordinate_only=True``), depending on which is wanted.
    """
    batched_trajectory, _ = _ensure_batched(trajectory)
    dim = batched_trajectory.shape[-1]
    embedded_low = delay_embedding(trajectory, embedding_dim, delay)
    embedded_high = delay_embedding(trajectory, embedding_dim + 1, delay)
    num_common = embedded_high.shape[-2]
    embedded_low = embedded_low[..., :num_common, :]

    distances, indices = knn_search(embedded_low, num_neighbors=2, theiler_window=theiler_window, norm_order=norm_order)
    # A coincident nearest neighbor (distance 0) falls back to the
    # second-nearest, per delay_afnn, rather than being excluded.
    use_second = distances[..., 0] == 0
    neighbor_index = torch.where(use_second, indices[..., 1], indices[..., 0])
    low_distance = torch.where(use_second, distances[..., 1], distances[..., 0])
    valid = (neighbor_index != -1) & (low_distance > 0)
    safe_index = torch.where(valid, neighbor_index, torch.zeros_like(neighbor_index))

    target = embedded_high[..., -dim:] if new_coordinate_only else embedded_high
    index_expanded = safe_index.unsqueeze(-1).expand(*safe_index.shape, target.shape[-1])
    neighbor_target = torch.gather(target, dim=-2, index=index_expanded)
    high_distance = torch.linalg.vector_norm(target - neighbor_target, ord=norm_order, dim=-1)

    safe_low = torch.where(valid, low_distance, torch.ones_like(low_distance))
    ratio = torch.where(valid, high_distance / safe_low, torch.zeros_like(high_distance))
    # E1's denominator matches delay_afnn's num_common - 1 exactly; E2
    # has no Julia convention to match (see module docstring), so it
    # uses the mathematically direct choice: the number of valid terms.
    denominator = num_common - 1 if not new_coordinate_only else valid.sum(dim=-1).to(ratio.dtype)
    return ratio.sum(dim=-1) / denominator


def cao_e1(trajectory, embedding_dim: int, delay: int = 1, theiler_window: int = 0, norm_order: float = 2.0) -> torch.Tensor:
    """Cao's E1(d) = E(d+1) / E(d): saturates near 1 once ``embedding_dim``
    is large enough to unfold the attractor.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        embedding_dim: Dimension ``d`` to evaluate (compares ``d``,
            ``d + 1``, and ``d + 2``).
        delay: Lag, forwarded to ``delay_embedding``.
        theiler_window: Theiler window, forwarded to ``knn_search``.
        norm_order: Minkowski norm order.

    Returns:
        Scalar (or ``[B]``) E1(d).
    """
    e_low = _cao_step(trajectory, embedding_dim, delay, theiler_window, norm_order, new_coordinate_only=False)
    e_high = _cao_step(trajectory, embedding_dim + 1, delay, theiler_window, norm_order, new_coordinate_only=False)
    return e_high / e_low


def cao_e2(trajectory, embedding_dim: int, delay: int = 1, theiler_window: int = 0, norm_order: float = 2.0) -> torch.Tensor:
    """Cao's E2(d) = E*(d+1) / E*(d): use alongside ``cao_e1`` to confirm
    its conclusion isn't an artifact of the signal being random -- a
    random signal's E2 approaches 1 much more slowly as ``embedding_dim``
    grows than a deterministic signal's does, not necessarily near 1 at
    any single small ``embedding_dim`` (see module docstring).

    Args: same as :func:`cao_e1`.

    Returns:
        Scalar (or ``[B]``) E2(d).
    """
    e_low = _cao_step(trajectory, embedding_dim, delay, theiler_window, norm_order, new_coordinate_only=True)
    e_high = _cao_step(trajectory, embedding_dim + 1, delay, theiler_window, norm_order, new_coordinate_only=True)
    return e_high / e_low
