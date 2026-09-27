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


def _curve_length(values: torch.Tensor, amplitude_only: bool) -> torch.Tensor:
    """L: sum of successive-point distances, ``[B, T]`` -> ``[B]``."""
    step = values[..., 1:] - values[..., :-1]
    step_length = step.abs() if amplitude_only else torch.hypot(torch.ones_like(step), step)
    return step_length.sum(dim=-1)


def _planar_extent(values: torch.Tensor, amplitude_only: bool) -> torch.Tensor:
    """d: largest distance from the first point, ``[B, T]`` -> ``[B]``."""
    amplitude_gap = values[..., 1:] - values[..., :1]
    if amplitude_only:
        distance_from_start = amplitude_gap.abs()
    else:
        index_gap = torch.arange(1, values.shape[-1], dtype=values.dtype, device=values.device)
        distance_from_start = torch.hypot(index_gap.expand_as(amplitude_gap), amplitude_gap)
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
        series with ``amplitude_only=True``), or a zero denominator
        (``d / L = 1 / n``). A constant series gives ``D = 1`` in the
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

    length = _curve_length(values, amplitude_only)
    extent = _planar_extent(values, amplitude_only)
    log_num_steps = torch.log10(torch.tensor(float(num_points - 1), dtype=work_dtype, device=values.device))
    # Degenerate rows are replaced by safe values before the log/division so
    # their undefined arithmetic cannot reach the backward pass of other rows.
    zero_length = length == 0
    log_ratio = torch.log10(torch.where(zero_length, torch.ones_like(extent), extent) / torch.where(zero_length, torch.ones_like(length), length))
    denominator = log_num_steps + log_ratio
    # d / L = 1 / n makes the denominator zero; test it on d * n == L since
    # log10 rounding can leave a tiny nonzero value (n = 3: -5.6e-17).
    degenerate = zero_length | (denominator == 0) | (extent * (num_points - 1) == length)
    result = log_num_steps / torch.where(degenerate, torch.ones_like(denominator), denominator)
    result = torch.where(degenerate, torch.full_like(result, float("nan")), result)
    return _restore_batch(result.to(dtype), ndim)
