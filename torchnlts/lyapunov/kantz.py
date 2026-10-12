"""Kantz's largest Lyapunov exponent estimator, from observed data.

Reference: Kantz, H. (1994), "A robust method to estimate the maximal Lyapunov exponent of a time series", Phys.
Lett. A 185(1), 77-87, eq. 1-2; Kantz & Schreiber, "Nonlinear Time Series Analysis", 2nd ed., Cambridge University
Press, 2004, Ch. 5. The trajectory is delay-embedded, each point's full epsilon-neighborhood is found outside a Theiler
window (Rosenstein's estimator uses one nearest neighbor), and
``S(k) = <ln( (1/|U_i|) * sum_{j in U_i} ||X_{i+k} - X_{j+k}|| )>_i`` is tracked as the neighborhood is evolved ``k``
steps: the log of the *mean* neighborhood distance, averaged over reference points. The slope of ``S(k)`` over an
early range of ``k`` estimates the largest Lyapunov exponent, through
:func:`~torchnlts.lyapunov.rosenstein_lyapunov`.

Differs from ChaosTools.jl's ``lyapunov_from_data`` at ``ntype=WithinRange(radius)``, which takes the log of each
neighbor's distance before averaging (Parlitz's generalization): the mean of logs instead of the log of the mean.
By Jensen's inequality the log of the mean is the larger, and the two curves differ by more than rounding.

Gradient class: piecewise (neighborhood membership is a hard threshold; the evolved distances are smooth once it is
fixed).
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _as_int_list,
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _real_scalar,
    _restore_sweep,
)
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, theiler_window_mask


def kantz_divergence(
    trajectory,
    k_values,
    radius: float,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kantz's average log-neighborhood-divergence curve ``S(k)``.

    Reference: Kantz (1994), Phys. Lett. A 185(1), 77-87, eq. 1-2.

    Gradient class: piecewise (see the module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        k_values: Evolution steps ``k >= 0`` to evaluate at: an integer-valued number or a 1-D sequence of them
            (``[4.0, 8.0]`` is accepted).
        radius: Neighborhood radius ``epsilon >= 0``, inclusive (``<=``). Not swept: call again to check the
            estimate's sensitivity to it.
        embedding_dim: Delay-embedding dimension.
        delay: Embedding lag.
        theiler_window: Embedded points with ``|i - j| <= theiler_window`` are not neighbors. Must be ``>= 0``, or a
            point would be its own neighbor.
        norm_order: Minkowski norm order of the neighborhood search and of the evolved-point distances.

    Returns:
        ``S(k)`` with shape ``[]`` or ``[K]`` for an unbatched trajectory, ``[B]`` or ``[K, B]`` for a batched one
        (``[K]`` and ``[K, B]`` when ``k_values`` is a sequence), in the input's floating dtype (the default dtype for
        integer input; half precision is computed in float32). The reference points are those with a nonempty
        neighborhood among the points that can be evolved ``max(k_values)`` steps; ``NaN`` if there are none. ``-inf``
        at a ``k`` where any reference point's mean neighborhood distance is exactly 0 (e.g. a neighborhood of
        duplicate points), as ``log(0)``. ``NaN`` for a row containing NaN/inf.

    Raises:
        ValueError: If ``k_values`` is not a non-negative integer or a non-empty 1-D sequence of them, ``radius`` is
            not a real number ``>= 0``, ``embedding_dim`` or ``delay`` is not a positive integer or too large for the
            trajectory, ``theiler_window`` is negative or not an integer, or the trajectory is complex.
    """
    k_list, swept = _as_int_list(k_values, "k_values", minimum=0)
    theiler_window = _require_self_exclusion(theiler_window, "Kantz's Lyapunov estimator")
    try:
        radius_value = _real_scalar(radius, "radius")
    except ValueError:
        raise ValueError(f"radius must be a real number >= 0, got {radius!r}") from None
    if not radius_value >= 0:
        raise ValueError(f"radius must be a real number >= 0, got {radius!r}")

    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    # Zero non-finite rows so backward can't put NaN into other rows; their results are NaN.
    raw, ndim = _ensure_batched(trajectory)
    dtype = _output_dtype(raw)
    raw, nonfinite = _mask_nonfinite_rows(raw)
    batched = delay_embedding(raw, embedding_dim, delay=delay)
    num_points = batched.shape[-2]
    max_k = max(k_list)

    # The distance at lag k is the shifted slice [i+k, j+k] of the full pairwise matrix computed below.
    distance = pairwise_distance(batched, norm_order=norm_order)
    neighbor_mask = (distance <= radius_value) & theiler_window_mask(num_points, theiler_window, device=batched.device)

    num_usable = 0 if max_k >= num_points else num_points - max_k
    active_neighbor_mask = neighbor_mask[..., :num_usable, :num_usable]
    neighbor_count = active_neighbor_mask.sum(dim=-1)
    row_valid = neighbor_count > 0
    safe_neighbor_count = neighbor_count.clamp(min=1).to(distance.dtype)
    row_valid_count = row_valid.sum(dim=-1)

    curve = []
    for k in k_list:
        k = min(k, num_points)
        distance_at_k = distance[..., k : k + num_usable, k : k + num_usable]
        masked_distance = torch.where(active_neighbor_mask, distance_at_k, torch.zeros_like(distance_at_k))
        mean_distance = masked_distance.sum(dim=-1) / safe_neighbor_count
        # Use a dummy operand of 1 at rows with no neighbors or zero mean distance so log's gradient stays finite.
        usable = row_valid & (mean_distance > 0)
        safe_mean_distance = torch.where(usable, mean_distance, torch.ones_like(mean_distance))
        log_mean_distance = torch.where(row_valid & ~usable, float("-inf"), torch.log(safe_mean_distance))
        masked_log = torch.where(row_valid, log_mean_distance, torch.zeros_like(log_mean_distance))
        curve.append(masked_log.sum(dim=-1) / row_valid_count)

    result = torch.where(nonfinite, float("nan"), torch.stack(curve, dim=0))  # [K, B]
    return _restore_sweep(result, swept, ndim).to(dtype)
