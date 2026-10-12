"""Higuchi fractal dimension: roughness of a scalar time series' graph, estimated directly from the data (no delay
embedding).

Reference: Higuchi (1988), Physica D 31(2), 277-283. For each offset m = 1, ..., k the series is subsampled at stride k
starting from m; the curve length L_m(k) is the sum of consecutive absolute differences times
``(N - 1) / (floor((N - m) / k) * k**2)``, and L(k) is the average of L_m(k) over m. The dimension D_H, in [1, 2] in
theory, is the OLS slope of ``log(L(k))`` against ``log(1/k)`` over a sweep of k. The caller picks the sweep (no
automatic scaling-region detection), as in :func:`~torchnlts.dimension.correlation_dimension`.

Conventions: :func:`~torchnlts.dimension.higuchi_length` matches FractalDimensions.jl's ``higuchi_length``, including
its ``k**2`` normalization (Higuchi's own formula). :func:`~torchnlts.dimension.higuchi_dimension` fits every given k,
whereas Julia's ``higuchi_dim`` first selects a linear region of the sweep, so they can differ (a sine wave over
strides 1-20 gives 1.274 here and 1.052 there). A k for which some offset has no valid step gives ``NaN``, as in
Julia; an empty input also gives ``NaN`` here, where Julia returns ``0.0``.

Univariate only: the definition is the roughness of the graph of a scalar function.

Gradient class: piecewise (subtraction, ``abs`` and sums over an index pattern fixed once ``k`` is chosen; kinks at
exact ties between two indexed points).
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int_list,
    _as_sweep,
    _ensure_batched,
    _mask_nonfinite_rows,
    _ols_slope,
    _output_dtype,
    _require_univariate,
    _restore_sweep,
    _safe_log,
    _work_dtype,
)


def _higuchi_length_single_k(values: torch.Tensor, k: int) -> torch.Tensor:
    """L(k) for a single stride ``k``, ``values`` shape ``[B, N]``."""
    num_points = values.shape[-1]
    offset_lengths = []
    for offset in range(k):
        num_steps = (num_points - 1 - offset) // k
        if num_steps < 1:
            # Matches Julia: division by zero here gives NaN for the whole L(k).
            return torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device)
        indices = offset + k * torch.arange(num_steps + 1, device=values.device)
        subsequence = values[..., indices]
        length = (subsequence[..., 1:] - subsequence[..., :-1]).abs().sum(dim=-1)
        normalization = (num_points - 1) / (num_steps * k * k)
        offset_lengths.append(length * normalization)
    return torch.stack(offset_lengths, dim=0).mean(dim=0)


def higuchi_length(trajectory, k) -> torch.Tensor:
    """Higuchi's average curve length L(k) at one or more strides.

    Reference: Higuchi (1988), Physica D 31, 277.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        k: Stride, or a non-empty 1-D sweep of strides; integer-valued numbers ``>= 1`` (``4.0`` is accepted, ``4.5``
            is rejected, not truncated).

    Returns:
        L(k), shape ``[]``/``[K]`` unbatched or ``[B]``/``[K, B]`` batched, in the trajectory's float dtype (the default
        dtype for integer input). ``NaN`` at any ``k`` for which some offset ``m`` in ``0, ..., k - 1`` has no valid
        step (``(T - 1 - m) // k < 1``), which includes every ``k`` for an empty or too-short trajectory, and for any
        row containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1, the trajectory is complex, or ``k`` is empty, complex, non-integer or
            ``< 1``.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor, "higuchi_length")
    dtype = _output_dtype(tensor)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(_work_dtype(dtype)))  # [B, T]

    k_list, swept = _as_int_list(k, "k")

    lengths = torch.stack([_higuchi_length_single_k(values, value) for value in k_list], dim=0)  # [K, B]
    lengths = torch.where(nonfinite, float("nan"), lengths)
    return _restore_sweep(lengths, swept, ndim).to(dtype)


def higuchi_dimension(k, length: torch.Tensor) -> torch.Tensor:
    """Higuchi fractal dimension D_H: OLS slope of ``log(L(k))`` vs ``log(1/k)``.

    Reference: Higuchi (1988), Physica D 31, 277. A plain regression over every ``(k, length)`` pair; for automatic
    scaling-region detection use :func:`torchnlts.scaling.scaling_slope` (pass ``-log(k)``, ``log(length)``).

    Gradient class: exact (closed-form OLS slope), with the gradient reaching ``length`` that of
    :func:`~torchnlts.dimension.higuchi_length`.

    Args:
        k: Shape ``[K]``, the sweep ``length`` was evaluated at.
        length: Shape ``[K]`` (unbatched) or ``[K, B]`` (batched), as returned by
            :func:`~torchnlts.dimension.higuchi_length` for the same sweep.

    Returns:
        D_H, shape ``[]`` for an unbatched ``length`` or ``[B]`` for a batched one, in its float dtype. Not guaranteed
        to lie in ``[1, 2]`` for a finite sweep (an 8-point series at strides 1 and 2 gives 3.263). ``NaN`` if ``k``
        has fewer than two distinct values or any ``length`` entry is non-positive or ``NaN``.

    Raises:
        ValueError: If ``k``/``length`` do not have the documented shapes with matching sweep lengths.
    """
    neg_log_k = -_safe_log(_as_sweep(k, length.device))
    return _ols_slope(neg_log_k, _safe_log(length)).to(_output_dtype(length))
