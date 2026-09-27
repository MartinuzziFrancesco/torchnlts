"""Grassberger-Procaccia correlation sum: hard threshold and smooth
kernel-weighted variants.

Reference: Grassberger & Procaccia (1983), Physica D 9(1-2), 189-208, eq.
1: C(r) = lim_{N->inf} (2 / (N(N-1))) sum_{i<j} Theta(r - ||x_i - x_j||),
with the Theiler (1986) window correction excluding temporally correlated
pairs from both the count and the normalization.

The correlation sum at a fixed radius is the same quantity RQA calls the
"recurrence rate" (:func:`torchnlts.recurrence.rqa.recurrence_rate`) of
the corresponding recurrence matrix -- both are "fraction of
Theiler-windowed pairs within radius r". This module doesn't reuse
recurrence_rate directly (its ``[T, T]``/``[B, T, T]`` contract can't
express a whole radius *sweep* in one call), but broadcasts ``radius``
against the distance matrix directly so a full C(r) curve -- the input a
dimension estimate needs -- comes from one pairwise-distance computation.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _broadcast_over_matrix, _ols_slope, _output_dtype, _safe_log
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.neighbors.theiler import theiler_window_mask


def correlation_sum(trajectory, radius, theiler_window: int = 0, norm_order: float = 2.0) -> torch.Tensor:
    """Grassberger-Procaccia correlation sum C(r): fraction of point pairs within radius r.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Reference: Grassberger & Procaccia (1983), eq. 1. Matches
    FractalDimensions.jl's ``correlationsum(X, eps; w=theiler_window,
    q=2)`` (Theiler window ``w`` uses the same ``|i - j| <= w`` exclusion
    convention as this codebase's ``theiler_window``, unlike
    RecurrenceAnalysis.jl's off-by-one ``theiler`` keyword -- see
    :func:`torchnlts.recurrence.rqa.recurrence_rate`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        radius: Scalar or ``[R]`` tensor of radii to evaluate C(r) at.
        theiler_window: Theiler window (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
        norm_order: Minkowski norm order.

    Returns:
        Correlation sum(s) in ``[0, 1]``, shape ``[]``/``[R]`` for an
        unbatched trajectory or ``[B]``/``[R, B]`` for a batched one
        (matching whether ``radius`` is scalar or ``[R]``), or ``NaN`` if
        ``theiler_window`` excludes every pair.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    radius = torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    within_radius = distance <= _broadcast_over_matrix(radius, distance)
    return (within_radius & mask).sum(dim=(-2, -1)).to(distance.dtype) / mask.sum()


def soft_correlation_sum(
    trajectory,
    temperature,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted correlation sum, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`correlation_sum`, built on
    :func:`torchnlts.neighbors.kernels.trajectory_kernel` -- ``temperature``
    plays the role of ``radius``'s smooth analogue, same convention as
    :func:`torchnlts.recurrence.matrix.soft_recurrence_matrix`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        temperature: Scalar or ``[R]`` tensor of kernel bandwidths.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window, forwarded to the point mask.
        norm_order: Minkowski norm order.

    Returns:
        Weighted correlation sum(s) in ``[0, 1]``, same shape convention
        as :func:`correlation_sum`.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    temperature = torch.as_tensor(temperature, dtype=distance.dtype, device=distance.device)
    weight = trajectory_kernel(distance, _broadcast_over_matrix(temperature, distance), kind=kind)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    return torch.where(mask, weight, torch.zeros_like(weight)).sum(dim=(-2, -1)) / mask.sum()


def correlation_dimension(radius: torch.Tensor, correlation_sum: torch.Tensor) -> torch.Tensor:
    """Correlation dimension D2: ordinary-least-squares slope of log C(r) vs log r.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``correlation_sum`` -- it inherits
    :func:`correlation_sum`'s ``none`` or :func:`soft_correlation_sum`'s
    ``smooth`` from there).

    This is a plain regression over every ``(radius, correlation_sum)``
    pair given, with no automatic scaling-region detection -- the
    caller picks the radii, matching FractalDimensions.jl's
    ``slopefit(x, y, LinearRegression(); ignore_saturation=false)``
    (``ignore_saturation`` defaults to ``true`` there even under
    ``LinearRegression`` and can silently drop endpoints, so it must be
    turned off to reproduce this function's all-point fit) rather than
    ``slopefit``'s own default, ``LargestLinearRegion``. Matches
    :func:`torchnlts.dimension.correlation.soft_correlation_sum`'s note
    that ``temperature`` plays the role of ``radius``'s smooth analogue:
    pass the temperature sweep here for the soft variant's dimension
    estimate.

    Reference: Grassberger & Procaccia (1983), eq. 1 and surrounding
    discussion (D2 as the log-log slope of the correlation sum).

    Args:
        radius: Shape ``[R]``, the sweep the correlation sum was
            evaluated at (or the matching temperature sweep for the
            soft variant).
        correlation_sum: Shape ``[R]`` (unbatched trajectory) or
            ``[R, B]`` (batched trajectory), as returned by
            :func:`correlation_sum` / :func:`soft_correlation_sum` for
            the same sweep.

    Returns:
        Slope estimate(s), shape ``[]`` for an unbatched
        ``correlation_sum`` or ``[B]`` for a batched one. ``NaN`` if
        ``radius`` has fewer than two distinct values. ``NaN`` also if any ``correlation_sum`` entry is
        non-positive (``log`` of it is ``-inf``/``NaN``, e.g. a fully
        Theiler-excluded radius).

    Raises:
        ValueError: If ``radius``/``correlation_sum`` do not have the documented
            shapes with matching sweep lengths.
    """
    log_radius = _safe_log(torch.as_tensor(radius, dtype=correlation_sum.dtype, device=correlation_sum.device))
    return _ols_slope(log_radius, _safe_log(correlation_sum)).to(_output_dtype(correlation_sum))
