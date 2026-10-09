"""Complexity estimate and complexity-invariant distance (CID) of univariate series.

Reference: Batista, Keogh, Tataw & de Souza (2014), "CID: an efficient complexity-invariant distance
for time series", Data Mining and Knowledge Discovery 28(3), 634-669:
``CE(Q) = sqrt(sum_i (q_{i+1} - q_i)^2)`` and ``CID(Q, C) = ED(Q, C) * max(CE(Q), CE(C)) / min(CE(Q), CE(C))``.

Gradient class: exact except at zero (the norms are not differentiable there; the zero subgradient is
returned) and, for CID, at ties ``CE(Q) = CE(C)`` (max/min kink). Where exactly one CE is zero the CID
is ``inf``; its gradient is not meaningful there and is returned as zero.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import (
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _power_of_two_scale,
    _restore_batch,
    _work_dtype,
)


def _prepare(trajectory, name: str) -> tuple[torch.Tensor, int]:
    tensor, ndim = _ensure_batched(trajectory)
    if tensor.shape[-1] != 1:
        raise ValueError(f"{name} only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")
    if tensor.is_complex():
        raise ValueError(f"{name} does not support complex input")
    return tensor.squeeze(-1), ndim


def _complexity(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values[:, 1:] - values[:, :-1], dim=-1)


def complexity_estimate(trajectory, normalize: bool = False) -> torch.Tensor:
    """Complexity estimate ``CE = sqrt(sum_i (x_{i+1} - x_i)^2)``, the length of the series' difference vector.

    Reference: Batista et al. (2014), Data Min. Knowl. Discov. 28(3), 634.

    Gradient class: exact except at a constant series (zero subgradient).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas
            Series/DataFrame.
        normalize: Z-normalize each series first (population standard deviation, as in tsfresh's
            ``cid_ce``). A constant series then has ``CE = 0``.

    Returns:
        CE, shape ``[]`` unbatched or ``[B]`` batched, in the input's floating dtype. ``NaN`` for any
        row containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1, or the trajectory is complex.
    """
    values, ndim = _prepare(trajectory, "complexity_estimate")
    dtype = _output_dtype(values)
    values, nonfinite = _mask_nonfinite_rows(values.to(_work_dtype(dtype)))
    # A power-of-two row scale keeps the squared differences in range; CE scales linearly with it.
    scale = _power_of_two_scale(values.unsqueeze(-1))[:, :, 0]
    values = values / scale
    if normalize:
        deviation = values.std(dim=-1, unbiased=False, keepdim=True)
        constant = deviation == 0
        centered = values - values.mean(dim=-1, keepdim=True)
        values = torch.where(constant, torch.zeros_like(values), centered / torch.where(constant, 1, deviation))
        result = _complexity(values)
    else:
        result = _complexity(values) * scale[:, 0]
    result = torch.where(nonfinite, torch.full_like(result, float("nan")), result)
    return _restore_batch(result.to(dtype), ndim)


def complexity_invariant_distance(trajectory, other) -> torch.Tensor:
    """Euclidean distance scaled by the ratio of the larger to the smaller complexity estimate.

    Reference: Batista et al. (2014), Data Min. Knowl. Discov. 28(3), 634.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``.
        other: Same convention, same ``T``. Batch sizes must be equal or one of them 1 (or unbatched),
            in which case that series is paired with every row of the other.

    Returns:
        CID, shape ``[]`` if both inputs are unbatched, else ``[B]``. If both complexity estimates are
        zero the correction factor is 1 (CID = ED); if only one is zero it is ``inf`` (CID = ``inf``
        unless ED = 0). ``NaN`` for any pair with a NaN/inf value.

    Raises:
        ValueError: If a feature dim is not 1, an input is complex, the lengths differ, or the batch
            sizes are incompatible.
    """
    first, first_ndim = _prepare(trajectory, "complexity_invariant_distance")
    second, second_ndim = _prepare(other, "complexity_invariant_distance")
    if first.shape[-1] != second.shape[-1]:
        raise ValueError(f"series must have equal length, got {first.shape[-1]} and {second.shape[-1]}")
    if first.shape[0] != second.shape[0] and 1 not in (first.shape[0], second.shape[0]):
        raise ValueError(f"incompatible batch sizes {first.shape[0]} and {second.shape[0]}")
    dtype = torch.promote_types(_output_dtype(first), _output_dtype(second))
    work_dtype = _work_dtype(dtype)
    first, first_nonfinite = _mask_nonfinite_rows(first.to(work_dtype))
    second, second_nonfinite = _mask_nonfinite_rows(second.to(work_dtype))
    nonfinite = first_nonfinite | second_nonfinite
    # Power-of-two scales keep every norm in range. ED uses one scale for the pair; each CE uses its own
    # series' scale, so a series much smaller than the other keeps a nonzero scaled CE.
    pair_scale = _power_of_two_scale(first.unsqueeze(-1), second.unsqueeze(-1))[:, :, 0]
    first_scale = _power_of_two_scale(first.unsqueeze(-1))[:, :, 0]
    second_scale = _power_of_two_scale(second.unsqueeze(-1))[:, :, 0]
    euclidean = torch.linalg.vector_norm((first - second) / pair_scale, dim=-1)
    first_complexity = _complexity(first / first_scale)
    second_complexity = _complexity(second / second_scale)
    # A zero CE is exact (a constant series). Pairs with a zero CE (both zero: ratio 1; one zero: infinite)
    # use a placeholder ratio of 1 in the arithmetic, so the infinite result is selected afterwards and
    # cannot put NaN into backward.
    first_zero, second_zero = first_complexity == 0, second_complexity == 0
    first_safe = torch.where(first_zero, torch.ones_like(first_complexity), first_complexity)
    second_safe = torch.where(second_zero, torch.ones_like(second_complexity), second_complexity)
    # The scales enter only as integer exponents, applied once at the end, so no intermediate overflows
    # or underflows unless the result does. The larger CE goes in the numerator.
    with torch.no_grad():
        exponents = [
            torch.frexp(scale[:, 0])[1].to(torch.int64) - 1 for scale in (pair_scale, first_scale, second_scale)
        ]
        pair_exponent, first_exponent, second_exponent = exponents
        log_ratio = torch.log2(first_safe / second_safe) + (first_exponent - second_exponent).to(first_safe.dtype)
        first_high = log_ratio >= 0
    mantissa = euclidean * torch.where(first_high, first_safe / second_safe, second_safe / first_safe)
    exponent = pair_exponent + torch.where(
        first_high, first_exponent - second_exponent, second_exponent - first_exponent
    )
    zero_any = first_zero | second_zero
    mantissa = torch.where(zero_any, euclidean, mantissa)
    exponent = torch.where(zero_any, pair_exponent, exponent)
    limit = math.frexp(torch.finfo(work_dtype).max)[1] - 4
    exponent = exponent.clamp(-2 * limit, 2 * limit)
    half = torch.div(exponent, 2, rounding_mode="floor")
    result = mantissa * torch.exp2(half.to(work_dtype)) * torch.exp2((exponent - half).to(work_dtype))
    singular = first_zero != second_zero
    result = torch.where(singular, torch.full_like(result, float("inf")), result)
    result = torch.where(nonfinite, torch.full_like(result, float("nan")), result)
    return _restore_batch(result.to(dtype), max(first_ndim, second_ndim))
