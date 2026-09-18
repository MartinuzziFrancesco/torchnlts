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

from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.neighbors.theiler import theiler_window_mask


def _broadcast_over_matrix(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar or ``[R]`` radius/temperature sweep to broadcast
    against ``matrix``'s trailing ``[..., T, T]`` dims as a leading ``R``
    axis, e.g. ``[R]`` -> ``[R, 1, 1]`` for an unbatched ``[T, T]`` matrix.
    An already-scalar (0-dim) ``value`` reshapes to all-ones, which
    broadcasts the same as leaving it scalar.
    """
    return value.view(value.shape + (1,) * matrix.dim())


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
