"""Automatic scaling-region detection for the log-log (or semi-log) slope
estimators.

Reference: the sequential ``LargestLinearRegion`` algorithm of
FractalDimensions.jl (``linear_regions``, ``linear_region``, ``slopefit``)
with ``dxi = 1``. Unlike Julia (``ignore_saturation = true`` by default),
``saturation=None`` here: no segments are trimmed unless a threshold is
given. A zero-width segment (repeated sweep point) has no slope and never
joins a region. Consecutive points define segments with slope
``dy / dx``; a segment starts a new region when its slope differs from the
slope of the segment that started the current region by more than
``tolerance`` (relative, ``|a - b| <= tolerance * max(|a|, |b|)``).
Neighbouring regions share their boundary point, and the region covering
the most points (the first one on ties) is the scaling region.

What to pass as ``(sweep, values)`` so the slope equals each estimator's
result over the detected region:

==================================  ==========================  ==========================
Estimator                           ``sweep``                   ``values``
==================================  ==========================  ==========================
``correlation_dimension``           ``log(radius)``             ``log(correlation_sum)``
``dfa_exponent``                    ``log(window_size)``        ``log(fluctuation)``
``higuchi_dimension``               ``-log(k)``                 ``log(length)``
``generalized_dimension``           ``-log(box_size)``          ``entropy``
``rosenstein_lyapunov``             ``k_values``                ``divergence``
``variogram_dimension``             ``log(lag)``                ``log(variogram)``
==================================  ==========================  ==========================

``variogram_dimension`` returns ``2 - slope / power``; apply that to the
slope. Limitation: only the ``dxi = 1`` scan of FractalDimensions.jl is
implemented (no ``dxi`` stride, no ``:overlap`` method); add a stride by
averaging segment slopes over ``dxi`` points.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_sweep, _is_complex, _mask_nonfinite_rows, _ols_slope, _output_dtype, _work_dtype


def _real_scalar(value, name: str) -> float:
    """``value`` as a Python float; ``ValueError`` for strings, ``None``, complex or non-scalars."""
    if isinstance(value, (str, bytes)) or value is None or _is_complex(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None


def _prepare(sweep, values, tolerance, saturation):
    """Validated ``(sweep [R], values [R, B], was_batched, nonfinite [B])`` in the work dtype."""
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    sweep = _as_sweep(sweep, values.device)
    if sweep.dim() != 1 or values.dim() not in (1, 2) or sweep.shape[0] != values.shape[0]:
        raise ValueError(f"expected a sweep [R] and values [R] or [R, B] with matching R, got {tuple(sweep.shape)} and {tuple(values.shape)}")
    if sweep.shape[0] < 2:
        raise ValueError(f"need at least 2 sweep points, got {sweep.shape[0]}")
    if not _real_scalar(tolerance, "tolerance") > 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance!r}")
    if saturation is not None and not _real_scalar(saturation, "saturation") >= 0:
        raise ValueError(f"saturation must be >= 0 or None, got {saturation!r}")
    work_dtype = torch.promote_types(_work_dtype(values.dtype), _work_dtype(sweep.dtype))
    batched = values.dim() == 2
    values = values.to(work_dtype).reshape(values.shape[0], -1)
    values, nonfinite = _mask_nonfinite_rows(values.T)
    return sweep.to(work_dtype), values.T, batched, nonfinite


def _segment_labels(sweep, values, tolerance, saturation) -> torch.Tensor:
    """Region label of each segment, ``[R - 1, B]``, ``-1`` for ignored ones (no gradient)."""
    sweep, values = sweep.detach(), values.detach()
    width = (sweep[1:] - sweep[:-1]).unsqueeze(-1)
    # A zero-width segment has no slope (NaN, as in FractalDimensions.jl).
    slope = torch.where(width == 0, torch.full_like(values[1:], float("nan")), (values[1:] - values[:-1]) / width)
    num_segments = slope.shape[0]
    position = torch.arange(num_segments, device=slope.device).unsqueeze(-1)
    if saturation is None:
        active = torch.ones_like(slope, dtype=torch.bool)
    else:
        # Segments between the first and last one that move by more than
        # ``saturation``; with none, every segment stays.
        big = (values[1:] - values[:-1]).abs() > saturation
        first = torch.where(big, position, num_segments).amin(dim=0)
        last = torch.where(big, position, -1).amax(dim=0)
        active = ~big.any(dim=0) | ((position >= first) & (position <= last))
    previous = slope[0]
    seen = torch.zeros_like(slope[0], dtype=torch.bool)
    label = torch.full_like(slope[0], -1, dtype=torch.long)
    labels = []
    for index in range(num_segments):
        current, on = slope[index], active[index]
        # Non-finite slopes (zero-width segments) are never close to anything.
        close = torch.isfinite(current) & torch.isfinite(previous) & ((current == previous) | ((current - previous).abs() <= tolerance * torch.maximum(current.abs(), previous.abs())))
        # The first active segment opens region 0; a later one opens a new
        # region when it is not close to the opening segment's slope.
        opens = on & (~seen | ~close)
        label = torch.where(on & ~seen, torch.zeros_like(label), label + (opens & seen).long())
        previous = torch.where(opens, current, previous)
        seen = seen | on
        labels.append(torch.where(on, label, torch.full_like(label, -1)))
    return torch.stack(labels)


def linear_regions(sweep, values, tolerance: float = 0.25, saturation: float | None = None) -> torch.Tensor:
    """Segment ``values`` against ``sweep`` into linear regions.

    Reference: FractalDimensions.jl ``linear_regions`` (``dxi = 1``).

    Gradient class: none (integer labels).

    Args:
        sweep: Shape ``[R]``, the x-axis (see the module docstring for
            what each estimator uses).
        values: Shape ``[R]`` or ``[R, B]``, the y-axis.
        tolerance: Relative slope tolerance for staying in a region.
        saturation: If given, ignore the leading and trailing segments
            whose ``|dy|`` is at most this (FractalDimensions.jl's
            ``ignore_saturation`` with ``sat_threshold``); ``None``
            keeps every segment.

    Returns:
        Long tensor of shape ``[R - 1]`` or ``[R - 1, B]``: the region
        index (from 0) of each segment between consecutive sweep points,
        ``-1`` for ignored segments and for every segment of a row with a
        non-finite entry or sweep.

    Raises:
        ValueError: If the shapes differ, ``R < 2``, ``tolerance <= 0``
            or ``saturation < 0``.
    """
    sweep, values, batched, nonfinite = _prepare(sweep, values, tolerance, saturation)
    sweep_ok = torch.isfinite(sweep).all()
    labels = _segment_labels(sweep, values, tolerance, saturation)
    labels = torch.where(nonfinite | ~sweep_ok, torch.full_like(labels, -1), labels)
    return labels if batched else labels.squeeze(-1)


def _largest(labels: torch.Tensor):
    """``(start, stop)`` point indices ``[B]`` of the most populated label of ``labels`` ``[R - 1, B]``."""
    ids = torch.arange(labels.shape[0], device=labels.device).view(-1, 1, 1)
    counts = (labels.unsqueeze(0) == ids).sum(dim=1)  # [label, B]
    best = counts.argmax(dim=0)
    start = (labels.unsqueeze(0) == ids).long().argmax(dim=1).gather(0, best.unsqueeze(0)).squeeze(0)
    stop = start + counts.gather(0, best.unsqueeze(0)).squeeze(0) + 1
    return start, stop


def largest_linear_region(sweep, values, tolerance: float = 0.25, saturation: float | None = None):
    """Start and stop indices of the largest linear region per row.

    Reference: FractalDimensions.jl ``linear_region`` / ``LargestLinearRegion``.

    Gradient class: none (integer indices).

    Args:
        sweep: Shape ``[R]``.
        values: Shape ``[R]`` or ``[R, B]``.
        tolerance: See :func:`linear_regions`.
        saturation: See :func:`linear_regions`.

    Returns:
        ``(start, stop)`` long tensors, shape ``[]`` or ``[B]``, so the
        region is ``values[start:stop]``. ``(-1, -1)`` for a row with a
        non-finite entry or sweep.

    Raises:
        ValueError: As :func:`linear_regions`.
    """
    labels = linear_regions(sweep, values, tolerance, saturation)
    batched = labels.dim() == 2
    labels = labels if batched else labels.unsqueeze(-1)
    start, stop = _largest(labels)
    invalid = (labels < 0).all(dim=0)
    start, stop = torch.where(invalid, -1, start), torch.where(invalid, -1, stop)
    return (start, stop) if batched else (start.squeeze(0), stop.squeeze(0))


def scaling_slope(sweep, values, tolerance: float = 0.25, saturation: float | None = None) -> torch.Tensor:
    """OLS slope of ``values`` against ``sweep`` over the largest linear region.

    Reference: FractalDimensions.jl ``slopefit(x, y, LargestLinearRegion())``.

    Gradient class: exact for the chosen region (closed-form OLS slope;
    the region choice is piecewise constant in ``values`` and carries no
    gradient).

    Args:
        sweep: Shape ``[R]``.
        values: Shape ``[R]`` or ``[R, B]``.
        tolerance: See :func:`linear_regions`.
        saturation: See :func:`linear_regions`.

    Returns:
        Slope(s), shape ``[]`` or ``[B]``, in ``values``' floating dtype
        (half precision is computed in float32). ``NaN`` for a row with a
        non-finite entry or sweep, whose gradient is zero.

    Raises:
        ValueError: As :func:`linear_regions`.
    """
    out_dtype = _output_dtype(values if torch.is_tensor(values) else torch.as_tensor(values))
    sweep, values, batched, nonfinite = _prepare(sweep, values, tolerance, saturation)
    labels = _segment_labels(sweep, values, tolerance, saturation)
    start, stop = _largest(labels)
    point = torch.arange(sweep.shape[0], device=sweep.device).unsqueeze(-1)
    slope = _ols_slope(sweep, values, (point >= start) & (point < stop))
    # Rows with a non-finite entry, or any non-finite sweep point, are NaN.
    invalid = nonfinite | ~torch.isfinite(sweep).all()
    slope = torch.where(invalid, torch.full_like(slope, float("nan")), slope).to(out_dtype)
    return slope if batched else slope.squeeze(0)
