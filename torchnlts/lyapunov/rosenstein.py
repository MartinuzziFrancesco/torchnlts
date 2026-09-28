"""Rosenstein's largest Lyapunov exponent estimator, from observed data.

Reference: Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993), "A
practical method for calculating largest Lyapunov exponents from small
data sets", Physica D 65(1-2), 117-134: delay-embed the trajectory, find
each point's single nearest neighbor (excluding temporally correlated
points via a Theiler window), then track the average log-divergence
E(k) = <ln ||X_{i+k} - X_{j(i)+k}||>_i as the pair is evolved forward k
steps. For a chaotic system this grows linearly, E(k) ~ lambda_1 * k +
E(0), over a well-defined early-k region; the largest Lyapunov exponent
is the slope of that region (no automatic scaling-region detection --
the caller picks the fit range, same convention as
:func:`torchnlts.dimension.correlation.correlation_dimension`).

Matches ChaosTools.jl's more general ``lyapunov_from_data`` at
``ntype=NeighborNumber(1)``, ``distance=Euclidean()``. That function's
Parlitz-due generalization also accepts a radius-neighborhood
(``ntype=WithinRange``) in place of a single nearest neighbor, structurally
resembling Kantz's method -- but it is *not* a match for Kantz's own
1994 formula even there (it averages the log of each neighbor's distance
rather than the log of the mean neighbor distance the paper defines; see
:func:`torchnlts.lyapunov.kantz.kantz_divergence`'s module docstring for
the confirmed disagreement). This codebase keeps Rosenstein and Kantz as
separate functions, ``rosenstein_divergence`` here and ``kantz_divergence``
there, matching this repo's package layout and the two papers' distinct
presentations. ``rosenstein_lyapunov`` below fits either curve -- the OLS
slope step doesn't care which method produced ``divergence``.
"""

from __future__ import annotations

import operator

import torch

from torchnlts._shape import _as_sweep, _ensure_batched, _ols_slope, _output_dtype, _work_dtype
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import _gather_neighbors, knn_search
from torchnlts.neighbors.theiler import _require_self_exclusion


def rosenstein_divergence(
    trajectory,
    k_values,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Rosenstein's average log-divergence curve E(k).

    Gradient class: piecewise (built on :func:`torchnlts.neighbors.knn.knn_search`'s
    nearest-neighbor selection: the evolved-point distances differentiate
    normally w.r.t. the trajectory once a neighbor is picked, but the
    pick itself is a discrete, non-differentiable decision that can flip
    at an infinitesimal perturbation).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        k_values: Evolution steps to evaluate E(k) at. Scalar or ``[K]``
            tensor to sweep.
        embedding_dim: Delay-embedding dimension.
        delay: Embedding lag.
        theiler_window: Theiler window over embedded-point start indices,
            for the nearest-neighbor search only (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`). The
            paper recommends excluding points within the series' mean
            period; this generalizes that into a plain window, left to
            the caller to size appropriately (default ``0`` excludes
            only the self-match).
        norm_order: Minkowski norm order for both the neighbor search and
            the evolved-point divergence.

    Returns:
        E(k), shape ``[]``/``[K]`` for an unbatched trajectory or
        ``[B]``/``[K, B]`` for a batched one. ``NaN`` if no reference
        point has both a valid nearest neighbor (Theiler window didn't
        exclude every candidate) and enough remaining trajectory length
        for both to evolve ``max(k_values)`` steps forward -- the same
        reference-point set, sized by the *largest* requested ``k``, is
        used for every entry of a swept ``k_values`` (matching
        ChaosTools.jl's ``lyapunov_from_data``, whose default
        ``refstates = 1:(length(R) - ks[end])`` does the same, confirmed
        by reading its source): this keeps every point of the E(k) curve
        comparable, evaluated over one consistent population of
        reference points, rather than a population that changes (grows)
        as ``k`` shrinks. ``-inf`` at a ``k`` where any valid
        reference point's evolved divergence happens to be exactly
        ``0`` (e.g. exact duplicate points in ``trajectory``), matching
        ``log(0)`` rather than silently flooring it to a large finite
        value.

    Raises:
        ValueError: If ``k_values`` is empty, has more than one
            dimension, or contains a negative, non-finite, or
            non-integer-valued entry, or if ``theiler_window < 0``
            (self-matches must be excluded unconditionally, or the
            "nearest neighbor" is always the point itself at distance 0,
            evolving trivially with it).
    """
    # float64, not torch.as_tensor's default dtype (float32 unless
    # changed globally): a plain Python float/list argument would
    # otherwise silently lose precision before validation even runs,
    # e.g. -1e-50 rounds to -0.0 in float32 (passing the >= 0 check) and
    # 1.00000001 rounds to exactly 1.0 (passing the integer check).
    k_values_tensor = torch.as_tensor(k_values, dtype=torch.float64)
    if k_values_tensor.numel() == 0:
        raise ValueError("k_values must not be empty")
    if k_values_tensor.dim() > 1:
        raise ValueError(f"k_values must be scalar or 1D, got shape {tuple(k_values_tensor.shape)}")
    if torch.any(~torch.isfinite(k_values_tensor)):
        raise ValueError("k_values must be finite")
    if torch.any(k_values_tensor < 0):
        raise ValueError("k_values must be non-negative (evolution steps forward in time)")
    if torch.any(k_values_tensor != k_values_tensor.round()):
        raise ValueError("k_values must be integer-valued (they index discrete time steps)")
    _require_self_exclusion(theiler_window, "Rosenstein's Lyapunov estimator")

    embedding_dim = operator.index(embedding_dim)
    delay = operator.index(delay)
    theiler_window = operator.index(theiler_window)
    # Limitation: delay_embedding's `torch.arange(embedding_dim) * delay`
    # raises OverflowError converting an astronomically large Python int
    # `delay` to a tensor scalar, even when embedding_dim=1 makes the
    # actual multiplication trivially 0 regardless of delay's value (a
    # one-copy embedding has no delayed offsets, so its output never
    # depends on delay -- confirmed: num_embedded = num_points -
    # (embedding_dim-1)*delay = num_points exactly at embedding_dim=1).
    # Worked around locally here rather than fixed at the root in
    # delay_embedding, since delay_embedding is an already-shipped,
    # already-tested shared primitive used across the package, and
    # this is far outside any realistic delay value. Upgrade path: guard
    # the tensor-scalar conversion in delay_embedding itself if another
    # caller hits the same crash with a non-degenerate embedding_dim.
    embedding_delay = 1 if embedding_dim == 1 and delay >= 1 else delay
    embedded = delay_embedding(trajectory, embedding_dim, embedding_delay)
    batched, ndim = _ensure_batched(embedded)
    dtype = _output_dtype(batched)
    batched = batched.to(_work_dtype(dtype))
    # Limitation: same OverflowError class, this time in
    # theiler_window_mask's `abs() > window` comparison against an
    # astronomically large Python int. Larger-than-the-trajectory
    # windows all exclude every pair identically, so clamping to
    # num_points changes nothing about the actual result while avoiding
    # the crash. Same upgrade path as above: fix at the root in
    # theiler_window_mask if this recurs elsewhere.
    theiler_window = min(theiler_window, batched.shape[-2])
    _, neighbor_index = knn_search(batched, num_neighbors=1, theiler_window=theiler_window, norm_order=norm_order)
    neighbor_index = neighbor_index[..., 0]
    neighbor_found = neighbor_index >= 0
    safe_neighbor_index = torch.where(neighbor_found, neighbor_index, torch.zeros_like(neighbor_index))

    batch_size, num_points = batched.shape[0], batched.shape[-2]
    start_index = torch.arange(num_points, device=batched.device)

    k_list = k_values_tensor.reshape(-1).tolist()
    max_k = int(max(k_list))

    # One reference-point mask for the whole sweep, sized by max_k (see
    # Returns): a point still qualifies for a smaller k in the sweep
    # whenever it qualifies for max_k, since k <= max_k keeps both the
    # reference and its neighbor's evolved index in bounds a fortiori.
    # `max_k >= num_points` is handled as its own case rather than by
    # `start_index + max_k`, since max_k is an arbitrary Python int here
    # (from k_values, not bounded by the trajectory) and could overflow
    # int64 index arithmetic -- silently wrapping to a small/negative
    # value that would pass the `< num_points` bounds check it's
    # supposed to fail.
    if max_k >= num_points:
        valid = torch.zeros(batch_size, num_points, dtype=torch.bool, device=batched.device)
    else:
        reference_in_bounds = (start_index + max_k) < num_points
        neighbor_in_bounds = (safe_neighbor_index + max_k) < num_points
        valid = reference_in_bounds.unsqueeze(0) & neighbor_in_bounds & neighbor_found
    valid_count = valid.sum(dim=-1)

    curve = []
    for k in k_list:
        # Out-of-range sweeps have an all-false mask; keep even their
        # unused gather indices within integer range.
        k = min(int(k), num_points)
        reference_index = (start_index + k).expand(batch_size, -1)
        neighbor_target_index = safe_neighbor_index + k
        reference_evolved = _gather_neighbors(batched, reference_index, valid)
        neighbor_evolved = _gather_neighbors(batched, neighbor_target_index, valid)
        divergence = (reference_evolved - neighbor_evolved).norm(p=norm_order, dim=-1)
        # At an invalid (masked) position, both gathers fall back to the
        # same dummy row 0, so divergence is exactly 0 there -- an
        # artifact of the safe-index substitution, not real data. Left
        # alone, log(0)'s 1/0 gradient would poison the backward pass
        # through torch.where's *unselected* branch (its forward value
        # is correctly ignored below, but autograd still needs a finite
        # local gradient there); substituting 1 (log(1) = 0, and its
        # gradient 1/1 = 1 is finite) avoids that without touching any
        # valid position, including a genuinely-zero one (see Returns).
        # A genuine zero at a valid position keeps its -inf forward value
        # (see Returns) but is also given a safe operand, so log(0)'s
        # infinite gradient cannot reach a parameter shared with other rows.
        usable = valid & (divergence > 0)
        safe_divergence = torch.where(usable, divergence, torch.ones_like(divergence))
        log_divergence = torch.where(valid & ~usable, torch.full_like(divergence, float("-inf")), torch.log(safe_divergence))
        masked_log_divergence = torch.where(valid, log_divergence, torch.zeros_like(log_divergence))
        curve.append(masked_log_divergence.sum(dim=-1) / valid_count)

    result = torch.stack(curve, dim=0) if k_values_tensor.dim() > 0 else curve[0]
    # Not `_restore_batch` (which squeezes dim 0): a swept result is
    # `[K, B]`, batch trailing rather than leading, matching
    # `correlation_sum`'s own sweep convention -- so the batch axis to
    # drop for an unbatched trajectory is always the *last* one here.
    return (result.squeeze(-1) if ndim != 3 else result).to(dtype)


def rosenstein_lyapunov(k_values: torch.Tensor, divergence: torch.Tensor) -> torch.Tensor:
    """Largest Lyapunov exponent: OLS slope of E(k) vs k.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``divergence`` -- it inherits
    :func:`rosenstein_divergence`'s ``piecewise`` from there).

    This is a plain regression over every ``(k, E(k))`` pair given, with
    no automatic scaling-region detection -- the caller picks the ``k``
    range, matching :func:`torchnlts.dimension.correlation.correlation_dimension`'s
    same convention (and FractalDimensions.jl's ``slopefit`` with
    ``ignore_saturation=false``); ChaosTools.jl's own
    ``lyapunov_from_data`` doesn't fit a slope at all -- it only returns
    the E(k) curve, and its docs separately suggest (not perform) a
    ``linear_region`` call as a follow-up.

    The slope is w.r.t. the raw ``k`` (sample steps), not ``k * dt``: if
    ``trajectory`` was sampled at interval ``dt``, divide the result by
    ``dt`` to convert to a per-unit-time Lyapunov exponent (the paper's
    ``E(k) ~ lambda_1 * k * dt + E(0)``, see the module docstring).

    Reference: Rosenstein, Collins & De Luca (1993), eq. 12-13 and
    surrounding discussion (the logarithmic divergence and its
    least-squares slope; eq. 4 is the earlier trajectory-matrix
    reconstruction step, not this formula).

    Args:
        k_values: Shape ``[K]``, the sweep ``divergence`` was evaluated
            at (see :func:`rosenstein_divergence`).
        divergence: Shape ``[K]`` (unbatched trajectory) or ``[K, B]``
            (batched), as returned by :func:`rosenstein_divergence` for
            the same sweep.

    Returns:
        Largest Lyapunov exponent estimate(s), shape ``[]`` for an
        unbatched ``divergence`` or ``[B]`` for a batched one. ``NaN`` if
        ``k_values`` has fewer than two distinct values, or
        propagated from any ``NaN`` in ``divergence`` itself.

    Raises:
        ValueError: If the inputs do not have the documented shapes
            with matching sweep lengths.
    """
    return _ols_slope(_as_sweep(k_values, divergence.device), divergence)
