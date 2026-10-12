"""Grassberger-Procaccia correlation sum: hard threshold and smooth kernel-weighted variants.

Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), 189-208, eq. 1:
C(r) = lim_{N->inf} (2 / (N(N-1))) sum_{i<j} Theta(r - ||x_i - x_j||), with the Theiler (1986) window excluding
temporally correlated pairs from both the count and the normalization. At a fixed radius C(r) is the recurrence rate
(:func:`~torchnlts.recurrence.recurrence_rate`) of the matching recurrence matrix, but a whole C(r) sweep here comes
from one pairwise-distance computation.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_sweep, _broadcast_over_matrix, _ols_slope, _output_dtype, _safe_log
from torchnlts.convert import as_tensor
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.neighbors.theiler import theiler_window_mask


def _nan_where_nonfinite(total: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """``total`` with ``NaN`` for every batch row whose distance matrix has
    ``NaN`` (``pairwise_distance`` marks pairs with a non-finite point)."""
    nonfinite = torch.isnan(distance).flatten(-2).any(dim=-1)
    return torch.where(nonfinite, torch.full_like(total, float("nan")), total)


def correlation_sum(trajectory, radius, *, theiler_window: int = 0, norm_order: float = 2.0) -> torch.Tensor:
    """Grassberger-Procaccia correlation sum C(r): fraction of point pairs within radius r.

    Reference: Grassberger & Procaccia (1983), Physica D 9, 189, eq. 1. Matches FractalDimensions.jl's
    ``correlationsum(X, eps; w=theiler_window)``; pairs with ``|i - j| <= theiler_window`` are excluded
    (RecurrenceAnalysis.jl's ``theiler`` keyword is off by one from this).

    Gradient class: none (Heaviside threshold).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        radius: Scalar or ``[R]`` tensor of radii to evaluate C(r) at.
        theiler_window: Theiler window (see :func:`torchnlts.neighbors.theiler_window_mask`).
        norm_order: Minkowski norm order.

    Returns:
        C(r) in ``[0, 1]``, shape ``[]``/``[R]`` unbatched or ``[B]``/``[R, B]`` batched (scalar or ``[R]``
        ``radius``), in the trajectory's float dtype (the default dtype for integer input). ``NaN`` if
        ``theiler_window`` excludes every pair and for any row containing NaN/inf.

    Raises:
        ValueError: If ``theiler_window`` is not an integer, ``norm_order`` is not a real number ``>= 0``, or the
            trajectory is complex or has an invalid shape.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    radius = torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    within_radius = distance <= _broadcast_over_matrix(radius, distance)
    total = (within_radius & mask).sum(dim=(-2, -1)).to(distance.dtype) / mask.sum()
    return _nan_where_nonfinite(total, distance).to(_output_dtype(as_tensor(trajectory)))


def soft_correlation_sum(
    trajectory,
    temperature,
    *,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted correlation sum, a smooth stand-in for the hard threshold.

    ``temperature`` plays the role of ``radius`` (as in :func:`torchnlts.recurrence.soft_recurrence_matrix`), with
    pair weights from :func:`torchnlts.neighbors.trajectory_kernel`.

    Gradient class: smooth (a differentiable relaxation of :func:`~torchnlts.dimension.correlation_sum`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        temperature: Scalar or ``[R]`` tensor of kernel bandwidths.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window, forwarded to the point mask.
        norm_order: Minkowski norm order.

    Returns:
        Weighted correlation sum in ``[0, 1]``; shapes, dtype and ``NaN`` cases as in
        :func:`~torchnlts.dimension.correlation_sum`.

    Raises:
        ValueError: If ``theiler_window`` is not an integer, ``kind`` is unknown, ``temperature`` is not a positive
            real number or tensor, ``norm_order`` is not a real number ``>= 0``, or the trajectory is complex or has
            an invalid shape.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    temperature = torch.as_tensor(temperature, dtype=distance.dtype, device=distance.device)
    weight = trajectory_kernel(distance, _broadcast_over_matrix(temperature, distance), kind=kind)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    total = torch.where(mask, weight, torch.zeros_like(weight)).sum(dim=(-2, -1)) / mask.sum()
    return _nan_where_nonfinite(total, distance).to(_output_dtype(as_tensor(trajectory)))


def correlation_dimension(radius: torch.Tensor, correlation_sum: torch.Tensor) -> torch.Tensor:
    """Correlation dimension D2: ordinary-least-squares slope of log C(r) vs log r.

    Reference: Grassberger & Procaccia (1983), eq. 1 and the surrounding discussion. A plain regression over every
    ``(radius, correlation_sum)`` pair, matching FractalDimensions.jl's
    ``slopefit(x, y, LinearRegression(); ignore_saturation=false)`` (its defaults, ``LargestLinearRegion`` and
    ``ignore_saturation=true``, can drop endpoints). For automatic scaling-region detection use
    :func:`torchnlts.scaling.scaling_slope` (pass ``log(radius)``, ``log(correlation_sum)``). For the soft variant pass
    the temperature sweep as ``radius``.

    Gradient class: exact (closed-form OLS slope), with the gradient reaching ``correlation_sum`` that of its producer
    (none for :func:`~torchnlts.dimension.correlation_sum`, smooth for
    :func:`~torchnlts.dimension.soft_correlation_sum`).

    Args:
        radius: Shape ``[R]``, the sweep the correlation sum was evaluated at.
        correlation_sum: Shape ``[R]`` (unbatched) or ``[R, B]`` (batched), as returned by
            :func:`~torchnlts.dimension.correlation_sum` or :func:`~torchnlts.dimension.soft_correlation_sum` for the
            same sweep.

    Returns:
        D2, shape ``[]`` for an unbatched ``correlation_sum`` or ``[B]`` for a batched one, in its float dtype.
        ``NaN`` if ``radius`` has fewer than two distinct values or any ``correlation_sum`` entry is non-positive
        (``log`` of it is ``-inf``/``NaN``, e.g. a fully Theiler-excluded radius).

    Raises:
        ValueError: If ``radius``/``correlation_sum`` do not have the documented shapes with matching sweep lengths.
    """
    log_radius = _safe_log(_as_sweep(radius, correlation_sum.device))
    return _ols_slope(log_radius, _safe_log(correlation_sum)).to(_output_dtype(correlation_sum))
