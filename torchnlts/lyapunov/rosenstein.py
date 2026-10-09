"""Rosenstein's largest Lyapunov exponent estimator, from observed data.

Reference: Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993), "A
practical method for calculating largest Lyapunov exponents from small
data sets", Physica D 65(1-2), 117-134: delay-embed the trajectory, find
each point's single nearest neighbor (excluding temporally correlated
points via a Theiler window), then track the average log-divergence
E(k) = <ln ||X_{i+k} - X_{j(i)+k}||>_i as the pair is evolved forward k
steps. For a chaotic system this grows linearly, E(k) ~ lambda_1 * k +
E(0), over a well-defined early-k region; the largest Lyapunov exponent
is the slope of that region (:func:`~torchnlts.lyapunov.rosenstein_lyapunov` fits the
range given; :func:`torchnlts.scaling.scaling_slope` detects it).

Matches ChaosTools.jl's more general ``lyapunov_from_data`` at
``ntype=NeighborNumber(1)``, ``distance=Euclidean()``. That function's
Parlitz-due generalization also accepts a radius-neighborhood
(``ntype=WithinRange``) in place of a single nearest neighbor, structurally
resembling Kantz's method -- but it is *not* a match for Kantz's own
1994 formula even there (it averages the log of each neighbor's distance
rather than the log of the mean neighbor distance the paper defines; see
:func:`torchnlts.lyapunov.kantz_divergence`'s module docstring for
the confirmed disagreement). This codebase keeps Rosenstein and Kantz as
separate functions, ``rosenstein_divergence`` here and ``kantz_divergence``
there, matching this repo's package layout and the two papers' distinct
presentations. ``rosenstein_lyapunov`` below fits either curve -- the OLS
slope step doesn't care which method produced ``divergence``.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _as_sweep,
    _ensure_batched,
    _is_complex,
    _mask_nonfinite_rows,
    _ols_slope,
    _output_dtype,
    _work_dtype,
)
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import _validate_norm_order
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

    Gradient class: piecewise (built on :func:`torchnlts.neighbors.knn_search`'s
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
            :func:`torchnlts.neighbors.theiler_window_mask`). The
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
        value. ``NaN`` for any row containing NaN/inf.

    Raises:
        ValueError: If ``k_values`` is complex, empty, has more than one
            dimension, or contains a negative, non-finite, or
            non-integer-valued entry, or if ``theiler_window < 0``
            (self-matches must be excluded unconditionally, or the
            "nearest neighbor" is always the point itself at distance 0,
            evolving trivially with it), or if ``norm_order`` is not a
            real number >= 0.
    """
    # float64, not the float32 default, so Python float arguments keep their precision until validated.
    if _is_complex(k_values):
        raise ValueError(f"k_values must be real, got {k_values!r}")
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
    theiler_window = _as_int(theiler_window, "theiler_window")
    _require_self_exclusion(theiler_window, "Rosenstein's Lyapunov estimator")
    norm_order = _validate_norm_order(norm_order)

    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    # Limitation: delay_embedding raises OverflowError for a huge int delay at embedding_dim=1; guard it there.
    embedding_delay = 1 if embedding_dim == 1 and delay >= 1 else delay
    # Masked before embedding: samples outside every embedded point can hold NaN/inf, which make a row NaN.
    raw, ndim = _ensure_batched(trajectory)
    dtype = _output_dtype(raw)
    raw, nonfinite = _mask_nonfinite_rows(raw.to(_work_dtype(dtype)))
    batched = delay_embedding(raw, embedding_dim, embedding_delay)
    # Limitation: theiler_window_mask overflows on a huge int window; clamped to num_points, same excluded pairs.
    theiler_window = min(theiler_window, batched.shape[-2])
    _, neighbor_index = knn_search(batched, num_neighbors=1, theiler_window=theiler_window, norm_order=norm_order)
    neighbor_index = neighbor_index[..., 0]
    neighbor_found = neighbor_index >= 0
    safe_neighbor_index = torch.where(neighbor_found, neighbor_index, torch.zeros_like(neighbor_index))

    batch_size, num_points = batched.shape[0], batched.shape[-2]
    start_index = torch.arange(num_points, device=batched.device)

    k_list = k_values_tensor.reshape(-1).tolist()
    max_k = int(max(k_list))

    # One reference mask sized by max_k; max_k >= num_points is separate since start_index + max_k can overflow int64.
    if max_k >= num_points:
        valid = torch.zeros(batch_size, num_points, dtype=torch.bool, device=batched.device)
    else:
        reference_in_bounds = (start_index + max_k) < num_points
        neighbor_in_bounds = (safe_neighbor_index + max_k) < num_points
        valid = reference_in_bounds.unsqueeze(0) & neighbor_in_bounds & neighbor_found
    valid_count = valid.sum(dim=-1)

    curve = []
    for k in k_list:
        # Out-of-range sweeps have an all-false mask; keep their unused gather indices in integer range.
        k = min(int(k), num_points)
        reference_index = (start_index + k).expand(batch_size, -1)
        neighbor_target_index = safe_neighbor_index + k
        reference_evolved = _gather_neighbors(batched, reference_index, valid)
        neighbor_evolved = _gather_neighbors(batched, neighbor_target_index, valid)
        difference = reference_evolved - neighbor_evolved
        # Divide differences by their largest magnitude (detached) before the norm_order power to avoid
        # overflow and underflow; skipped below norm_order=1, where powers cannot overflow (norm_order=0 counts nonzeros).
        largest = difference.detach().abs().amax(dim=-1, keepdim=True)
        largest = torch.where(
            (largest > 0) & torch.isfinite(largest) & (norm_order >= 1), largest, torch.ones_like(largest)
        )
        divergence = (difference / largest).norm(p=norm_order, dim=-1) * largest.squeeze(-1)
        # Masked positions gather the same dummy row, so divergence is 0 there; substitute 1 so log's backward is
        # finite in torch.where's unselected branch. A genuine zero at a valid position keeps -inf, with a safe operand.
        usable = valid & (divergence > 0)
        safe_divergence = torch.where(usable, divergence, torch.ones_like(divergence))
        log_divergence = torch.where(
            valid & ~usable, torch.full_like(divergence, float("-inf")), torch.log(safe_divergence)
        )
        masked_log_divergence = torch.where(valid, log_divergence, torch.zeros_like(log_divergence))
        curve.append(masked_log_divergence.sum(dim=-1) / valid_count)

    result = torch.stack(curve, dim=0) if k_values_tensor.dim() > 0 else curve[0]
    result = torch.where(nonfinite, float("nan"), result)
    # Not _restore_batch: a swept result is [K, B], so the batch axis to drop is the last one.
    return (result.squeeze(-1) if ndim != 3 else result).to(dtype)


def rosenstein_lyapunov(k_values: torch.Tensor, divergence: torch.Tensor) -> torch.Tensor:
    """Largest Lyapunov exponent: OLS slope of E(k) vs k.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``divergence`` -- it inherits
    :func:`~torchnlts.lyapunov.rosenstein_divergence`'s ``piecewise`` from there).

    This is a plain regression over every ``(k, E(k))`` pair given
    (FractalDimensions.jl's ``slopefit`` with ``ignore_saturation=false``);
    see :func:`torchnlts.scaling.scaling_slope` for automatic
    scaling-region detection (pass ``k_values``, ``divergence``).

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
            at (see :func:`~torchnlts.lyapunov.rosenstein_divergence`).
        divergence: Shape ``[K]`` (unbatched trajectory) or ``[K, B]``
            (batched), as returned by :func:`~torchnlts.lyapunov.rosenstein_divergence` for
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
