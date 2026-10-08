"""Recurrence matrix: hard threshold and smooth kernel-weighted variants.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329, eq. 1: R_ij = Theta(epsilon - ||x_i - x_j||).

The hard variant reuses :func:`torchnlts.neighbors.distance.pairwise_distance`
and is validated against RecurrenceAnalysis.jl's ``RecurrenceMatrix``. The
soft variant reuses :func:`torchnlts.neighbors.kernels.trajectory_kernel` as
its base primitive -- ``temperature`` plays the role of
``epsilon``'s smooth analogue, not a relaxed threshold at a fixed epsilon.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _is_complex
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel


_SCALES = ("max", "mean", "std")


def _nan_quantile(values: torch.Tensor, probability: float, dim: int, diagonal: bool = False) -> torch.Tensor:
    """Linear-interpolation quantile (R type 7, Julia's default) of ``values``
    along ``dim``, ignoring NaN; ``NaN`` where nothing is finite. With
    ``diagonal``, the probability is raised by one over the finite count (a
    self-distance column has one guaranteed zero) and capped at 1. Sort and
    gather only, so ``torch.vmap`` works.

    Limitation: index arithmetic in float64, float32 on devices without it
    (MPS), where it can pick the wrong neighbour above ~1e6 entries; upgrade
    path: exact integer arithmetic on the probability's binary fraction."""
    ordered = values.movedim(dim, -1).sort(dim=-1).values
    count = (~torch.isnan(ordered)).sum(dim=-1, keepdim=True)
    work = torch.float32 if values.device.type == "mps" else torch.float64
    count_work = count.to(work)
    probability = torch.full_like(count_work, probability)
    if diagonal:
        probability = (probability + 1.0 / count_work.clamp(min=1.0)).clamp(max=1.0)
    # Julia's formula, so a quantile landing exactly on an order statistic
    # reproduces it bit for bit: aleph = (n - 1) p + 1, 1-based.
    aleph = count_work * probability + (1.0 - probability)
    last = (count - 1).clamp(min=0)
    lower = (aleph.trunc().long() - 1).clamp(min=0).minimum(last)
    upper = (lower + 1).minimum(last)
    weight = (aleph - 1.0 - lower.to(work)).clamp(0.0, 1.0).to(values.dtype)
    low = ordered.gather(-1, lower)
    high = ordered.gather(-1, upper)
    result = low + weight * (high - low)
    return torch.where(count > 0, result, torch.full_like(result, float("nan"))).squeeze(-1)


def _distance_scale(distance: torch.Tensor, scale: str, self_distance: bool) -> torch.Tensor:
    """``max`` / ``mean`` / ``std`` of the finite entries of ``distance`` ``[..., T, T']``,
    shape ``[..., 1, 1]``. As in RecurrenceAnalysis.jl, ``max`` and ``mean`` of
    a self-distance matrix exclude the zero diagonal; ``std`` (sample) does not.
    Computed on distances divided by a detached power of two near their
    largest, so sums and squares neither overflow nor underflow."""
    finite = ~torch.isnan(distance)
    largest = torch.where(finite, distance, torch.zeros_like(distance)).detach().amax(dim=(-2, -1), keepdim=True)
    largest = torch.where(torch.isfinite(largest) & (largest > 0), largest, torch.ones_like(largest))
    unit = torch.exp2(torch.floor(torch.log2(largest)))
    scaled = distance / unit
    zeroed = torch.where(finite, scaled, torch.zeros_like(scaled))
    count = finite.sum(dim=(-2, -1), keepdim=True)
    if scale == "max":
        value = torch.where(finite, scaled, torch.full_like(scaled, float("-inf"))).amax(dim=(-2, -1), keepdim=True)
        value = torch.where(count > 0, value, torch.full_like(value, float("nan")))
    elif scale == "mean":
        if self_distance:
            count = count - torch.diagonal(finite, dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
        value = zeroed.sum(dim=(-2, -1), keepdim=True) / count
    else:
        mean = zeroed.sum(dim=(-2, -1), keepdim=True) / count
        squares = torch.where(finite, (scaled - mean) ** 2, torch.zeros_like(scaled))
        value = (squares.sum(dim=(-2, -1), keepdim=True) / (count - 1)).sqrt()
    return value * unit


def _rate(value, name: str) -> float:
    """``value`` as a float in ``(0, 1]``; ``ValueError`` otherwise."""
    try:
        if _is_complex(value):
            raise ValueError
        rate = float(value)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError(f"{name} must be a real number in (0, 1], got {value!r}") from None
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return rate


def _threshold_matrix(
    distance: torch.Tensor,
    epsilon,
    recurrence_rate,
    local_rate,
    scale,
    self_distance: bool,
) -> torch.Tensor:
    """Boolean ``distance <= threshold`` for the threshold option in use.

    Raises:
        ValueError: If not exactly one of ``epsilon``, ``recurrence_rate``,
            ``local_rate`` is given, ``scale`` is given without ``epsilon``
            or is not one of ``"max"``, ``"mean"``, ``"std"``, or a rate is
            outside ``(0, 1]``.
    """
    if sum(option is not None for option in (epsilon, recurrence_rate, local_rate)) != 1:
        raise ValueError("give exactly one of epsilon, recurrence_rate, local_rate")
    if scale is not None and (epsilon is None or scale not in _SCALES):
        raise ValueError(f"scale must be one of {_SCALES} and requires epsilon, got scale={scale!r}, epsilon={epsilon!r}")
    if recurrence_rate is not None:
        flat = distance.flatten(-2)
        threshold = _nan_quantile(flat, _rate(recurrence_rate, "recurrence_rate"), dim=-1)[..., None, None]
    elif local_rate is not None:
        threshold = _nan_quantile(distance, _rate(local_rate, "local_rate"), dim=-2, diagonal=self_distance)[..., None, :]
    elif scale is not None:
        threshold = epsilon * _distance_scale(distance, scale, self_distance)
    else:
        threshold = epsilon
    return distance <= threshold


def hard_recurrence_matrix(
    trajectory,
    epsilon: float | None = None,
    norm_order: float = 2.0,
    *,
    recurrence_rate: float | None = None,
    local_rate: float | None = None,
    scale: str | None = None,
) -> torch.Tensor:
    """Boolean recurrence matrix at a fixed, rate-derived or relative distance threshold.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        epsilon: Recurrence threshold. Points ``i``, ``j`` recur when
            their distance is ``<= epsilon``, matching RecurrenceAnalysis.jl's
            ``RecurrenceMatrix`` (including the always-recurrent diagonal,
            except at points with a NaN/inf coordinate, which recur with nothing).
            With ``scale``, a ratio of that statistic of the distances
            (``RecurrenceThresholdScaled``).
        norm_order: Minkowski norm order.
        recurrence_rate: Instead of ``epsilon``: the threshold is this
            quantile of all pairwise distances, giving about this fraction of
            recurrences (``GlobalRecurrenceRate``). In ``(0, 1]``.
        local_rate: Instead of ``epsilon``: each column has its own threshold
            giving it about this fraction of recurrences besides the diagonal
            (``LocalRecurrenceRate``; quantile-derived, ties and finite size
            can add recurrences); the matrix is not symmetric. In ``(0, 1]``.
            A rate of 1 is accepted (Julia rejects it) and saturates the
            diagonal-corrected probability at 1, where Julia's can exceed one.
        scale: ``"max"``, ``"mean"`` or ``"std"``; makes ``epsilon`` a ratio of
            that statistic of the pairwise distances.

    Returns:
        Bool tensor of shape ``[T, T]`` or ``[B, T, T]``; each batch row gets
        its own threshold, computed over its finite distances.

    Raises:
        ValueError: If not exactly one of ``epsilon``, ``recurrence_rate``,
            ``local_rate`` is given, ``scale`` is invalid or given without
            ``epsilon``, or a rate is outside ``(0, 1]``.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    return _threshold_matrix(distance, epsilon, recurrence_rate, local_rate, scale, self_distance=True)


def soft_recurrence_matrix(
    trajectory,
    temperature: torch.Tensor | float,
    kind: str = "gaussian",
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kernel-weighted recurrence matrix, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`hard_recurrence_matrix`, built on
    :func:`torchnlts.neighbors.kernels.trajectory_kernel` rather than a
    hidden default swap).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        temperature: Kernel bandwidth, forwarded to ``trajectory_kernel``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        norm_order: Minkowski norm order.

    Returns:
        Weight tensor in ``(0, 1]``, same shape as ``hard_recurrence_matrix``;
        ``0`` for pairs involving a point with a NaN/inf coordinate.
    """
    distance = pairwise_distance(trajectory, norm_order=norm_order)
    return trajectory_kernel(distance, temperature, kind=kind)
