"""Cao's method for the minimum embedding dimension of a scalar time series.

Reference: Cao, L. (1997), "Practical method for determining the minimum embedding dimension of a scalar time series",
Physica D 110, 43-50, eq. 1-3 (``cao_e1``) and eq. 4-5 (``cao_e2``).

Gradient class: none (nearest-neighbor search is a discrete decision).

Conventions:

- ``cao_e1`` matches DelayEmbeddings.jl's ``delay_afnn``: a coincident nearest neighbor (distance 0) falls back to the
  second nearest, and the mean divides by one less than the number of reference points.
- ``cao_e2`` uses the same neighbor rule and averages over the points with a valid neighbor. DelayEmbeddings.jl's
  ``stochastic_indicator`` divides by the embedded length instead of the number of summed terms and leaves the last
  ``delay + 1`` points out of the neighbor search; the results differ by about 0.5% at 500 points, 0.1% at 2000
  points and 5% at 100.
- For ``D > 1`` the added coordinate is the new ``D``-dimensional block, compared with ``norm_order``.
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


def _cao_step(
    values: torch.Tensor,
    embedding_dim: int,
    delay: int,
    theiler_window: int,
    norm_order: float,
    new_coordinate_only: bool,
) -> torch.Tensor:
    """``[B]`` E(d), or E*(d) with ``new_coordinate_only``, of ``values`` ``[B, T, D]``."""
    dim = values.shape[-1]
    embedded_low = delay_embedding(values, embedding_dim, delay=delay)
    embedded_high = delay_embedding(values, embedding_dim + 1, delay=delay)
    num_common = embedded_high.shape[-2]
    embedded_low = embedded_low[..., :num_common, :]

    distances, indices = knn_search(embedded_low, num_neighbors=2, theiler_window=theiler_window, norm_order=norm_order)
    # A coincident nearest neighbor (distance 0) falls back to the second-nearest, as in delay_afnn.
    use_second = distances[..., 0] == 0
    neighbor_index = torch.where(use_second, indices[..., 1], indices[..., 0])
    low_distance = torch.where(use_second, distances[..., 1], distances[..., 0])
    valid = (neighbor_index != -1) & (low_distance > 0)

    target = embedded_high[..., -dim:] if new_coordinate_only else embedded_high
    neighbor_target = _gather_neighbors(target, neighbor_index, valid)
    high_distance = torch.linalg.vector_norm(target - neighbor_target, ord=norm_order, dim=-1)
    if new_coordinate_only:
        return torch.where(valid, high_distance, 0).sum(dim=-1) / valid.sum(dim=-1)
    ratio = high_distance / torch.where(valid, low_distance, 1)
    # delay_afnn divides by num_common - 1.
    return torch.where(valid, ratio, 0).sum(dim=-1) / (num_common - 1)


def _cao_ratio(
    trajectory, embedding_dim: int, delay: int, theiler_window: int, norm_order: float, new_coordinate_only: bool
) -> torch.Tensor:
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    values, nonfinite = _mask_nonfinite_rows(batched.to(_work_dtype(batched.dtype)))
    e_low = _cao_step(values, embedding_dim, delay, theiler_window, norm_order, new_coordinate_only)
    e_high = _cao_step(values, embedding_dim + 1, delay, theiler_window, norm_order, new_coordinate_only)
    ratio = torch.where(nonfinite, float("nan"), e_high / e_low)
    return _restore_batch(ratio.to(out_dtype), ndim)


def cao_e1(
    trajectory, embedding_dim: int, *, delay: int = 1, theiler_window: int = 0, norm_order: float = 2.0
) -> torch.Tensor:
    """Cao's E1(d) = E(d + 1) / E(d), which saturates near 1 once ``embedding_dim`` unfolds the attractor.

    E(d) is the mean over points of the nearest-neighbor distance in dimension ``d + 1`` divided by the distance of
    the same pair in dimension ``d``.

    Reference: Cao (1997), Physica D 110, 43-50, eq. 1-3.

    Gradient class: none (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        embedding_dim: Dimension ``d >= 1``; ``d``, ``d + 1`` and ``d + 2`` are embedded.
        delay: Lag, forwarded to :func:`~torchnlts.embedding.delay_embedding`.
        theiler_window: Theiler window of the neighbor search (``|i - j| <= theiler_window`` is excluded).
        norm_order: Minkowski norm order, ``>= 0``.

    Returns:
        E1(d), shape ``[]`` (or ``[B]``), in the input's floating dtype (the default dtype for integer input).
        ``NaN`` for a row containing NaN/inf, and when no point has a valid neighbor (for example every candidate
        is excluded by ``theiler_window``).

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an
            integer, ``norm_order`` is invalid, the input is complex, or the trajectory is too short.
    """
    return _cao_ratio(trajectory, embedding_dim, delay, theiler_window, norm_order, new_coordinate_only=False)


def cao_e2(
    trajectory, embedding_dim: int, *, delay: int = 1, theiler_window: int = 0, norm_order: float = 2.0
) -> torch.Tensor:
    """Cao's E2(d) = E*(d + 1) / E*(d), which separates deterministic signals from noise.

    E*(d) is the mean over points of the distance between the coordinate added after dimension ``d`` and the same
    coordinate of the point's nearest neighbor found in dimension ``d``. E2 is about 1 for every ``d`` on random data
    and differs from 1 for some ``d`` on deterministic data.

    Reference: Cao (1997), Physica D 110, 43-50, eq. 4-5.

    Gradient class: none (see module docstring).

    Args: same as :func:`~torchnlts.embedding.cao_e1`.

    Returns:
        E2(d), shape ``[]`` (or ``[B]``), in the input's floating dtype (the default dtype for integer input).
        ``NaN`` for a row containing NaN/inf, and when no point has a valid neighbor.

    Raises:
        ValueError: As :func:`~torchnlts.embedding.cao_e1`.
    """
    return _cao_ratio(trajectory, embedding_dim, delay, theiler_window, norm_order, new_coordinate_only=True)
