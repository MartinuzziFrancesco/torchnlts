"""Rosenstein's largest Lyapunov exponent estimator, from observed data.

Reference: Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993), "A practical method for calculating largest
Lyapunov exponents from small data sets", Physica D 65(1-2), 117-134. The trajectory is delay-embedded, each point's
single nearest neighbor is found outside a Theiler window, and the average log-divergence
``E(k) = <ln ||X_{i+k} - X_{j(i)+k}||>_i`` is tracked as the pair is evolved ``k`` steps. For a chaotic system ``E(k)``
grows linearly, ``E(k) ~ lambda_1 * k + E(0)``, over an early range of ``k``; the largest Lyapunov exponent is the
slope there (:func:`~torchnlts.lyapunov.rosenstein_lyapunov` fits a given range,
:func:`torchnlts.scaling.scaling_slope` detects one).

Matches ChaosTools.jl's ``lyapunov_from_data`` at ``ntype=NeighborNumber(1)``, ``distance=Euclidean()``. For a sweep of
``k`` the reference points are fixed by the largest ``k`` (ChaosTools.jl's default
``refstates = 1:(length(R) - ks[end])``), so every point of the curve averages over the same population. Its radius
neighborhood (``ntype=WithinRange``) is not Kantz's estimator; see :func:`~torchnlts.lyapunov.kantz_divergence`.

Gradient class: piecewise (the nearest neighbor is a discrete choice; the evolved distances are smooth once it is made).
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _as_int_list,
    _as_sweep,
    _ensure_batched,
    _mask_nonfinite_rows,
    _ols_slope,
    _output_dtype,
    _restore_sweep,
    _work_dtype,
)
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import _validate_norm_order
from torchnlts.neighbors.knn import _gather_neighbors, knn_search
from torchnlts.neighbors.theiler import _require_self_exclusion


def rosenstein_divergence(
    trajectory,
    k_values,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Rosenstein's average log-divergence curve ``E(k)``.

    Reference: Rosenstein, Collins & De Luca (1993), Physica D 65(1-2), 117-134.

    Gradient class: piecewise (see the module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        k_values: Evolution steps ``k >= 0`` to evaluate at: an integer-valued number or a 1-D sequence of them
            (``[4.0, 8.0]`` is accepted).
        embedding_dim: Delay-embedding dimension.
        delay: Embedding lag.
        theiler_window: Embedded points with ``|i - j| <= theiler_window`` are not neighbors (the paper suggests the
            mean period). Must be ``>= 0``, or a point would be its own nearest neighbor.
        norm_order: Minkowski norm order of the neighbor search and of the evolved-point divergence.

    Returns:
        ``E(k)`` with shape ``[]`` or ``[K]`` for an unbatched trajectory, ``[B]`` or ``[K, B]`` for a batched one
        (``[K]`` and ``[K, B]`` when ``k_values`` is a sequence), in the input's floating dtype (the default dtype for
        integer input; half precision is computed in float32). The reference points are those that have a neighbor
        and, with it, can be evolved ``max(k_values)`` steps; the set is the same for every ``k``, and ``NaN`` if it is
        empty. ``-inf`` at a ``k`` where a reference point's evolved divergence is exactly 0 (e.g. duplicate points),
        as ``log(0)``. ``NaN`` for a row containing NaN/inf.

    Raises:
        ValueError: If ``k_values`` is not a non-negative integer or a non-empty 1-D sequence of them,
            ``theiler_window`` is negative or not an integer, ``embedding_dim`` or ``delay`` is not a positive
            integer or too large for the trajectory, ``norm_order`` is not a real number ``>= 0``, or the trajectory
            is complex.
    """
    k_list, swept = _as_int_list(k_values, "k_values", minimum=0)
    theiler_window = _require_self_exclusion(theiler_window, "Rosenstein's Lyapunov estimator")
    norm_order = _validate_norm_order(norm_order)

    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    # Masked before embedding: samples outside every embedded point can hold NaN/inf, which make a row NaN.
    raw, ndim = _ensure_batched(trajectory)
    dtype = _output_dtype(raw)
    raw, nonfinite = _mask_nonfinite_rows(raw.to(_work_dtype(dtype)))
    batched = delay_embedding(raw, embedding_dim, delay=delay)
    _, neighbor_index = knn_search(batched, num_neighbors=1, theiler_window=theiler_window, norm_order=norm_order)
    neighbor_index = neighbor_index[..., 0]
    neighbor_found = neighbor_index >= 0
    safe_neighbor_index = torch.where(neighbor_found, neighbor_index, torch.zeros_like(neighbor_index))

    batch_size, num_points = batched.shape[0], batched.shape[-2]
    start_index = torch.arange(num_points, device=batched.device)
    max_k = max(k_list)

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
        k = min(k, num_points)
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

    result = torch.where(nonfinite, float("nan"), torch.stack(curve, dim=0))  # [K, B]
    return _restore_sweep(result, swept, ndim).to(dtype)


def rosenstein_lyapunov(k_values: torch.Tensor, divergence: torch.Tensor) -> torch.Tensor:
    """Largest Lyapunov exponent: the ordinary-least-squares slope of ``E(k)`` against ``k``.

    The fit uses every ``(k, E(k))`` pair given (FractalDimensions.jl's ``slopefit`` with
    ``ignore_saturation=false``); :func:`torchnlts.scaling.scaling_slope` detects the scaling range instead. It
    also fits a :func:`~torchnlts.lyapunov.kantz_divergence` curve. The slope is per sample step: divide by the
    sampling interval ``dt`` for a per-unit-time exponent.

    Reference: Rosenstein, Collins & De Luca (1993), Physica D 65(1-2), 117-134, eq. 12-13.

    Gradient class: exact (closed-form slope; autodiff continues through whatever produced ``divergence``, so
    :func:`~torchnlts.lyapunov.rosenstein_divergence`'s piecewise class carries over).

    Args:
        k_values: Shape ``[K]``, the sweep ``divergence`` was evaluated at.
        divergence: Shape ``[K]`` (unbatched trajectory) or ``[K, B]`` (batched), as returned by
            :func:`~torchnlts.lyapunov.rosenstein_divergence` for the same sweep.

    Returns:
        The exponent(s), shape ``[]`` or ``[B]``. ``NaN`` if ``k_values`` has fewer than two distinct values or
        ``divergence`` contains a ``NaN``.

    Raises:
        ValueError: If the inputs do not have the documented shapes with matching sweep lengths.
    """
    return _ols_slope(_as_sweep(k_values, divergence.device), divergence)
