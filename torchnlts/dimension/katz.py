"""Katz fractal dimension of a scalar waveform, from its curve length and
planar extent; no delay embedding or sweep.

Reference: Katz, M.J. (1988), "Fractals and the analysis of waveforms",
Computers in Biology and Medicine 18(3), 145-156:
``D = log10(n) / (log10(n) + log10(d / L))``, n the number of steps, L the
total curve length, d the largest distance from the first point. Katz
measures both in the plane of (x, y) point pairs; this module places
sample i at (i, x_i), i.e. unit index spacing, so D depends on the
amplitude scale. ``amplitude_only=True`` drops the index term, the
variant used by antropy's ``katz_fd`` and mne-features.

Gradient class: piecewise (``hypot``/``abs`` and a ``max``; kinks at ties
for the farthest point and, with ``amplitude_only``, at zero steps).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch


def _validate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(f"katz_dimension only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")
    if tensor.is_complex():
        raise ValueError("katz_dimension does not support complex input")
    if not torch.isfinite(tensor).all():
        raise ValueError("katz_dimension requires a finite trajectory (no NaN/inf)")


def _curve_length(values: torch.Tensor, index_step: torch.Tensor | None) -> torch.Tensor:
    """L: sum of successive-point distances, ``[B, T]`` -> ``[B]``.
    ``index_step`` is the per-row index spacing ``[B]``, ``None`` for
    amplitude differences only."""
    step = values[..., 1:] - values[..., :-1]
    step_length = step.abs() if index_step is None else torch.hypot(index_step[:, None].expand_as(step), step)
    return step_length.sum(dim=-1)


def _planar_extent(values: torch.Tensor, index_step: torch.Tensor | None) -> torch.Tensor:
    """d: largest distance from the first point, ``[B, T]`` -> ``[B]``."""
    amplitude_gap = values[..., 1:] - values[..., :1]
    if index_step is None:
        distance_from_start = amplitude_gap.abs()
    else:
        index_gap = torch.arange(1, values.shape[-1], dtype=values.dtype, device=values.device) * index_step[:, None]
        distance_from_start = torch.hypot(index_gap, amplitude_gap)
    return distance_from_start.amax(dim=-1)


def katz_dimension(trajectory, amplitude_only: bool = False) -> torch.Tensor:
    """Katz fractal dimension ``log10(n) / (log10(n) + log10(d / L))``.

    Reference: Katz (1988), Comput. Biol. Med. 18(3), 145.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        amplitude_only: Measure L and d on amplitude differences alone
            (antropy's ``katz_fd``) instead of in the (index, amplitude)
            plane.

    Returns:
        D, shape ``[]`` unbatched or ``[B]`` batched. ``NaN`` when the
        formula is undefined: fewer than 3 points, ``L = 0`` (a constant
        series with ``amplitude_only=True``), or a denominator that is zero
        to within rounding (``d / L`` at ``1 / n``, where ``|D|`` would
        exceed about ``1 / (8 * eps)`` and is not determined by the data). A constant series gives ``D = 1`` in the
        plane (a straight line).

    Raises:
        ValueError: If the feature dim is not 1, or the trajectory is
            complex or contains NaN/inf.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _validate(tensor)
    dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
    # Half precision overflows index**2 and log terms; work in float32.
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    values = tensor.squeeze(-1).to(work_dtype)  # [B, T]
    num_points = values.shape[-1]

    if num_points < 3:
        result = torch.full(values.shape[:-1], float("nan"), dtype=work_dtype, device=values.device)
        return _restore_batch(result.to(dtype), ndim)

    # d / L is unchanged when both coordinates are divided by the same
    # factor, so compute in coordinates scaled to at most 1: no distance or
    # sum overflows. The factor is a constant, so detaching it is exact.
    scale = values.detach().abs().amax(dim=-1)
    if not amplitude_only:
        scale = scale.clamp(min=1.0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    values = values / scale[:, None]
    index_step = None if amplitude_only else 1.0 / scale
    length = _curve_length(values, index_step)
    extent = _planar_extent(values, index_step)
    log_num_steps = torch.log10(torch.tensor(float(num_points - 1), dtype=work_dtype, device=values.device))
    # The denominator vanishes at d / L = 1 / n, where D has a pole. Near it
    # D is not determined by the data (the denominator is at rounding level),
    # so such rows, and rows with L = 0, are NaN. They are found first and
    # given safe operands before the division, so their arithmetic cannot
    # put NaN into the gradients of other rows.
    with torch.no_grad():
        zero_length = length == 0
        log_ratio = torch.log10(extent / torch.where(zero_length, torch.ones_like(length), length))
        rounding = 8 * torch.finfo(work_dtype).eps * (log_num_steps + log_ratio.abs())
        degenerate = zero_length | ((log_num_steps + log_ratio).abs() <= rounding)
    safe_extent = torch.where(degenerate, torch.ones_like(extent), extent)
    safe_length = torch.where(degenerate, torch.ones_like(length), length)
    result = log_num_steps / (log_num_steps + torch.log10(safe_extent / safe_length))
    result = torch.where(degenerate, torch.full_like(result, float("nan")), result)
    return _restore_batch(result.to(dtype), ndim)
