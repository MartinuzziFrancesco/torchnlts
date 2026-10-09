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

import math

import torch

from torchnlts._shape import (
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _restore_batch,
    _safe_scale,
)


def _validate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(
            f"katz_dimension only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}"
        )
    if tensor.is_complex():
        raise ValueError("katz_dimension does not support complex input")


def _curve_length(step: torch.Tensor, index_step: torch.Tensor | None) -> torch.Tensor:
    """L from successive differences ``step`` ``[B, T-1]`` -> ``[B]``.
    ``index_step`` is the per-row index spacing ``[B]``, ``None`` for
    amplitude differences only."""
    step_length = step.abs() if index_step is None else torch.hypot(index_step[:, None].expand_as(step), step)
    return step_length.sum(dim=-1)


def _planar_extent(gap: torch.Tensor, index_step: torch.Tensor | None) -> torch.Tensor:
    """d from displacements from the first point ``gap`` ``[B, T-1]`` -> ``[B]``."""
    if index_step is None:
        return gap.abs().amax(dim=-1)
    index_gap = torch.arange(1, gap.shape[-1] + 1, dtype=gap.dtype, device=gap.device) * index_step[:, None]
    return torch.hypot(index_gap, gap).amax(dim=-1)


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
        to within rounding (``|n d - L| / L`` below ``8 * log2(T) * eps``, where
        ``D`` is not determined by the data). A constant series gives ``D = 1`` in the
        plane (a straight line). ``NaN`` for any row containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1, or the trajectory is
            complex.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _validate(tensor)
    dtype = _output_dtype(tensor)
    # float64 so float32/half data is not rejected as undetermined by the pole test.
    work_dtype = torch.float64
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(work_dtype))  # [B, T]
    num_points = values.shape[-1]

    if num_points < 3:
        result = torch.full(values.shape[:-1], float("nan"), dtype=work_dtype, device=values.device)
        return _restore_batch(result.to(dtype), ndim)

    # Scaled to at most 1 by a detached constant (d / L is scale-invariant), so no distance or sum overflows.
    step = values[:, 1:] - values[:, :-1]
    gap = values[:, 1:] - values[:, :1]
    scale = gap.detach().abs().amax(dim=-1)
    if not amplitude_only:
        scale = scale.clamp(min=1.0)
    scale = _safe_scale(scale)
    index_step = None if amplitude_only else 1.0 / scale
    length = _curve_length(step / scale[:, None], index_step)
    extent = _planar_extent(gap / scale[:, None], index_step)
    num_steps = num_points - 1
    log_num_steps = torch.log10(torch.tensor(float(num_steps), dtype=work_dtype, device=values.device))
    # log10(n) + log10(d / L) has a pole at n d = L; use log1p of the gap. Rows at the pole or with L = 0 are NaN,
    # with safe operands (d = L = 1) so they cannot put NaN into other rows' gradients.
    with torch.no_grad():
        zero_length = length == 0
        relative_gap = (extent * num_steps - length) / torch.where(zero_length, torch.ones_like(length), length)
        degenerate = (
            nonfinite | zero_length | (relative_gap.abs() <= 8 * math.log2(num_points) * torch.finfo(work_dtype).eps)
        )
    safe_extent = torch.where(degenerate, torch.ones_like(extent), extent)
    safe_length = torch.where(degenerate, torch.ones_like(length), length)
    denominator = torch.log1p((safe_extent * num_steps - safe_length) / safe_length) / math.log(10)
    result = log_num_steps / denominator
    result = torch.where(degenerate, torch.full_like(result, float("nan")), result)
    return _restore_batch(result.to(dtype), ndim)
