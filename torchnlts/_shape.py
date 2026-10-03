"""Tensor-convention normalization shared by every public function."""

from __future__ import annotations

import functools
import numpy

import torch

from torchnlts.convert import as_tensor


def _ensure_batched(tensor) -> tuple[torch.Tensor, int]:
    """Normalize ``[T]`` / ``[T, D]`` / ``[B, T, D]`` to ``[B, T, D]``.

    ``tensor`` may be a torch.Tensor, numpy array, or pandas
    Series/DataFrame (converted via :func:`torchnlts.convert.as_tensor`).

    Returns:
        The normalized tensor and its original number of dimensions.
    """
    tensor = as_tensor(tensor)
    if tensor.dim() == 1:
        return tensor.view(1, -1, 1), 1
    if tensor.dim() == 2:
        return tensor.unsqueeze(0), 2
    if tensor.dim() == 3:
        return tensor, 3
    raise ValueError(f"expected a tensor of shape [T], [T, D], or [B, T, D], got {tuple(tensor.shape)}")


def _restore_batch(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`_ensure_batched` on a result that kept the batch dim."""
    if ndim == 3:
        return tensor
    if ndim in (1, 2):
        return tensor.squeeze(0)
    raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")


def _restore_shape_preserving(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    """Undo :func:`_ensure_batched` on a result whose shape mirrors the
    input's exactly (e.g. a surrogate generator's ``[T] -> [T]``),
    unlike :func:`_restore_batch` (for a result whose *trailing* shape
    differs from the input's, e.g. delay_embedding's
    ``[T] -> [num_embedded, embedding_dim]``, where only the leading
    batch dim needs undoing): squeeze back every dim ``_ensure_batched``
    added, including the feature dim it synthesizes for a ``[T]`` input.
    """
    if ndim == 1:
        return tensor.squeeze(0).squeeze(-1)
    if ndim == 2:
        return tensor.squeeze(0)
    if ndim == 3:
        return tensor
    raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")


def _broadcast_over_matrix(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar or ``[R]`` radius/temperature sweep to broadcast
    against ``matrix``'s trailing ``[..., T, T]`` dims as a leading ``R``
    axis, e.g. ``[R]`` -> ``[R, 1, 1]`` for an unbatched ``[T, T]`` matrix.
    An already-scalar (0-dim) ``value`` reshapes to all-ones, which
    broadcasts the same as leaving it scalar.
    """
    return value.view(value.shape + (1,) * matrix.dim())


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Computation dtype for ``dtype``: default float for non-float input,
    float32 for half precision, else ``dtype`` itself.

    Raises:
        ValueError: If ``dtype`` is complex (converting would drop the
            imaginary part).
    """
    if dtype.is_complex:
        raise ValueError(f"complex input is not supported, got {dtype}")
    if not dtype.is_floating_point:
        return torch.get_default_dtype()
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _is_complex(value) -> bool:
    """Whether ``value`` (tensor, numpy array, or Python number/sequence) is complex."""
    return value.is_complex() if torch.is_tensor(value) else numpy.iscomplexobj(value)


def _as_sweep(value, device) -> torch.Tensor:
    """Sweep points (radii, strides, lags, ...) as a floating tensor.

    A floating tensor keeps its own dtype (casting it to the values' dtype
    could merge distinct points); anything else becomes float64.

    Raises:
        ValueError: If ``value`` is complex.
    """
    if _is_complex(value):
        raise ValueError(f"sweep must be real, got {value!r}")
    if torch.is_tensor(value):
        return value.to(device) if value.is_floating_point() else value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _as_positive_int_list(value, name: str) -> tuple[list[int], bool]:
    """Validate a positive-integer scalar or non-empty sweep.

    Returns:
        The values as Python ints, and whether ``value`` was a sweep.

    Raises:
        ValueError: If ``value`` is complex, empty, non-finite,
            non-integer, or has a value ``< 1``.
    """
    if _is_complex(value):
        raise ValueError(f"{name} must be integer-valued, got complex {value!r}")
    # float64 so a Python float like 1.00000001 is not rounded to an
    # integer by float32 before the integrality check.
    value_tensor = torch.as_tensor(value, dtype=torch.float64)
    flat = value_tensor.reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(flat).all() or not torch.equal(flat, flat.round()):
        raise ValueError(f"{name} must be integer-valued, got {flat.tolist()}")
    items = [int(item) for item in flat.tolist()]
    if any(item < 1 for item in items):
        raise ValueError(f"{name} must be >= 1, got {items}")
    return items, value_tensor.dim() > 0


def _output_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Result dtype for a computation on ``tensor``: its own floating dtype,
    or the default dtype for non-float input.

    Raises:
        ValueError: If ``tensor`` is complex.
    """
    if tensor.is_complex():
        raise ValueError(f"complex input is not supported, got {tensor.dtype}")
    return tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    """``scale`` with zeros replaced by one, for use as a divisor."""
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _mask_nonfinite_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``values`` ``[B, T]`` with every row containing NaN/inf set to zero,
    and the ``[B]`` mask of those rows. The zeroed rows cannot put NaN into
    backward; callers set their results to ``NaN`` with the mask. No
    data-dependent control flow, so ``torch.vmap`` works."""
    nonfinite = ~torch.isfinite(values).all(dim=-1)
    return torch.where(nonfinite[:, None], torch.zeros_like(values), values), nonfinite


def _safe_log(values: torch.Tensor) -> torch.Tensor:
    """``log(values)``, with every column (last dim of a ``[R, B]`` input,
    or the whole ``[R]`` input) that has a non-positive or non-finite entry
    set to ``NaN``. Invalid entries never reach ``log``, so no infinite
    gradient is produced there. Computed and returned in
    :func:`_work_dtype` (a log of half-precision values cannot resolve
    small ratios)."""
    values = values.to(_work_dtype(values.dtype))
    valid = (values > 0) & torch.isfinite(values)
    column_valid = valid.all(dim=0, keepdim=True)
    safe = torch.where(column_valid, values, torch.ones_like(values))
    return torch.where(column_valid, torch.log(safe), torch.full_like(values, float("nan")))


def _ols_slope(sweep: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Closed-form ordinary-least-squares slope of ``values`` against ``sweep``.

    Args:
        sweep: Shape ``[R]``, the points ``values`` was evaluated at.
        values: Shape ``[R]`` (unbatched) or ``[R, B]`` (batched).

    Returns:
        Slope estimate(s), shape ``[]`` for an unbatched ``values`` or
        ``[B]`` for a batched one, in ``values``' floating dtype (the
        default dtype for integer input; half precision is computed in
        float32). ``NaN`` if ``sweep`` has fewer than two distinct values
        or is non-finite, and for any column of ``values`` with a
        non-finite entry. A ``NaN`` result does not produce a ``NaN``
        gradient for the other columns or for ``sweep`` (for inputs away
        from the dtype's overflow range, see docs/primitives.rst).

    Raises:
        ValueError: If ``sweep``/``values`` do not have the shapes above
            with matching ``R``.
    """
    if sweep.dim() != 1 or values.dim() not in (1, 2) or sweep.shape[0] != values.shape[0]:
        raise ValueError(f"expected a sweep [R] and values [R] or [R, B] with matching R, got {tuple(sweep.shape)} and {tuple(values.shape)}")
    out_dtype = _output_dtype(values)
    # The wider of the two dtypes, so distinct float64 sweep points are not
    # merged when values are float32 (also rejects a complex sweep).
    work_dtype = torch.promote_types(_work_dtype(out_dtype), _work_dtype(sweep.dtype))
    if sweep.shape[0] == 0:
        # Zero-weighted sums keep the result connected to both inputs.
        connected = values.to(out_dtype).sum(dim=0) * 0 + sweep.to(out_dtype).sum() * 0
        return connected + float("nan")
    sweep = sweep.to(work_dtype).view((-1,) + (1,) * (values.dim() - 1))
    values = values.to(work_dtype)
    # Degenerate sweeps and invalid columns are replaced before any
    # arithmetic, so their NaN/inf cannot reach backward through a shared
    # input. Repeated values are checked on sweep itself: centering can
    # leave a rounding-sized nonzero denominator instead of an exact zero.
    # Tensor masks, not Python branches, so vmap still works.
    sweep_valid = torch.isfinite(sweep).all() & ~(sweep == sweep[:1]).all()
    column_valid = torch.isfinite(values).all(dim=0) & sweep_valid
    safe_sweep = torch.where(sweep_valid, sweep, torch.arange(sweep.shape[0], dtype=work_dtype, device=sweep.device).view_as(sweep))
    safe_values = torch.where(column_valid, values, torch.zeros_like(values))
    # Translating by the first point before centering keeps close points far
    # from zero distinct (the rounded mean of 2**24 and 2**24 + 2 is not) and
    # constant values exactly zero.
    safe_sweep = safe_sweep - safe_sweep[:1]
    safe_values = safe_values - safe_values[:1]
    sweep_centered = safe_sweep - safe_sweep.mean(dim=0, keepdim=True)
    values_centered = safe_values - safe_values.mean(dim=0, keepdim=True)
    # Products of centered terms overflow at about the square root of the
    # dtype's range; divide both by their (detached, hence exact) maxima
    # first and put the ratio back at the end.
    sweep_scale = _safe_scale(sweep_centered.detach().abs().amax(dim=0))
    values_scale = _safe_scale(values_centered.detach().abs().amax(dim=0))
    sweep_centered = sweep_centered / sweep_scale
    values_centered = values_centered / values_scale
    slope = (sweep_centered * values_centered).sum(dim=0) / (sweep_centered * sweep_centered).sum(dim=0)
    # Multiply before dividing: the ratio of scales alone can overflow.
    slope = slope * values_scale / sweep_scale
    return torch.where(column_valid, slope, torch.full_like(slope, float("nan"))).to(out_dtype)


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    """Shared shape guard for every function that masks a ``[T, T]``-style
    matrix by time-index pairs (Theiler window, RQA measures, ...):
    without it, a malformed shape surfaces as a confusing broadcast
    ``RuntimeError`` deep inside whichever masking op runs first, instead
    of a clear message at the call site.
    """
    if matrix.dim() not in (2, 3) or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected a [T, T] or [B, T, T] square matrix, got {tuple(matrix.shape)}")


def _validate_matching_batch(
    trajectory: torch.Tensor,
    other: torch.Tensor,
    trajectory_ndim: int,
    other_ndim: int,
    check_feature_dim: bool = True,
) -> None:
    """Shared shape guard for every two-trajectory function built on
    :func:`_ensure_batched` (``pairwise_distance``'s ``other`` argument, and
    everything that forwards it: ``knn_search``, ``radius_neighbors``, the
    cross/joint recurrence matrices, ...).

    Without it, a batch-dim mismatch (one trajectory batched, the other
    not) silently broadcasts to the wrong output shape instead of raising
    -- ``_restore_batch`` decides whether to squeeze using only
    ``trajectory``'s ndim, so a batched ``other`` paired with an unbatched
    ``trajectory`` returns an unintended ``[B, T, T']`` where a squeeze
    would otherwise have been a no-op. A feature-dim or non-broadcastable
    batch-size mismatch instead surfaces as a raw, context-free
    ``RuntimeError`` deep inside ``torch.cdist``.

    Args:
        trajectory: First trajectory, already normalized via ``_ensure_batched``.
        other: Second trajectory, already normalized via ``_ensure_batched``.
        trajectory_ndim: ``trajectory``'s original ndim, as returned by ``_ensure_batched``.
        other_ndim: ``other``'s original ndim, as returned by ``_ensure_batched``.
        check_feature_dim: Whether the two trajectories' feature dims must
            match. ``True`` for a shared-space comparison like cross
            distance (``x`` and ``y`` are points in the same space).
            ``False`` for a per-trajectory comparison like joint recurrence,
            where ``x`` and ``y`` may be states of two different systems of
            different dimensionality, combined only by matching time index.
    """
    trajectory_batched = trajectory_ndim == 3
    other_batched = other_ndim == 3
    if trajectory_batched != other_batched:
        raise ValueError(
            "trajectory and other must both be batched ([B, T, D]) or both "
            f"unbatched ([T] or [T, D]), got ndim {trajectory_ndim} and {other_ndim}"
        )
    if trajectory_batched and trajectory.shape[0] != other.shape[0]:
        raise ValueError(f"batch size mismatch between trajectory ({trajectory.shape[0]}) and other ({other.shape[0]})")
    if check_feature_dim and trajectory.shape[-1] != other.shape[-1]:
        raise ValueError(f"feature dim mismatch between trajectory ({trajectory.shape[-1]}) and other ({other.shape[-1]})")


def _power_of_two_scale(*tensors: torch.Tensor) -> torch.Tensor:
    """Detached per-row power of two within a factor of two of the largest
    finite magnitude (scaled magnitudes are below 2), else 1; shape
    ``[B, 1, 1]`` for ``[B, T, D]``."""
    reference = tensors[0]
    if not all(tensor.shape[-2] * tensor.shape[-1] > 0 for tensor in tensors):
        return torch.ones((1, 1, 1), dtype=reference.dtype, device=reference.device)
    # Finite coordinates only: a NaN/inf point must not disable the scaling
    # of the finite pairs in its row.
    magnitudes = [torch.where(torch.isfinite(tensor), tensor, 0).detach().abs().amax(dim=(-2, -1), keepdim=True) for tensor in tensors]
    scale = functools.reduce(torch.maximum, magnitudes)
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))
    return torch.exp2(torch.floor(torch.log2(scale)))
