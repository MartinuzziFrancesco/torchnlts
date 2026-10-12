"""Generalized (Rényi) dimension spectrum from box counting.

Reference: Russell, Hanson & Ott (1980), Phys. Rev. Lett. 45(14), 1175-1178 (box-counting estimation); Hentschel &
Procaccia (1983), Physica D 8(3), 435-444 (the D_q spectrum, with D_2 the correlation dimension). The state space is
partitioned into boxes of width ``box_size``, and D_q is the slope of the order-q Rényi entropy of the box-occupation
distribution against ``-log(box_size)``: ``H_q(box_size) ~ D_q * -log(box_size)``. Order 0 is the capacity
(box-counting) dimension, order 1 the information dimension, order 2 a box-counting estimate of the correlation
dimension (:func:`~torchnlts.dimension.correlation_dimension` estimates the same quantity from pairwise distances).

Conventions: boxes are left-closed, right-open and anchored at each coordinate's minimum over the trajectory, as in
ComplexityMeasures.jl's ``ValueHistogram`` (probabilities and Rényi entropies agree with
``entropy(Renyi(q=q, base=ℯ), ValueHistogram(ε), X)``). Unlike FractalDimensions.jl's ``generalized_dim``, the caller
supplies the box sizes and every point is fitted (no ``estimate_boxsizes``, no automatic linear-region detection), as
in :func:`~torchnlts.dimension.correlation_dimension`. The pairwise-distance q-order correlation sum (Grassberger 2007,
``q > 1`` only) is not implemented.

Gradient class: none (box assignment and occupied-box counting are discrete).

Limitation: not ``torch.vmap``-compatible, and the box-size sweep and batch rows are Python loops, because the number
of occupied boxes depends on the data and the scale.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_sweep,
    _ensure_batched,
    _ols_slope,
    _output_dtype,
    _real_scalar,
    _restore_sweep,
    _safe_log,
    _work_dtype,
)
from torchnlts.entropy.generalized import renyi_entropy


def box_occupation_probabilities(trajectory, box_size: float) -> torch.Tensor:
    """Relative frequency of the occupied boxes in a fixed-width partition of a trajectory's points.

    Gradient class: none (box assignment and occupied-box counting are discrete).

    Args:
        trajectory: Shape ``[T]`` or ``[T, D]`` (one trajectory; use :func:`~torchnlts.dimension.box_counting_entropy`
            for a batch). A tensor, numpy array, or pandas Series/DataFrame.
        box_size: Box width, the same in every dimension, a finite real number ``> 0``.

    Returns:
        Probabilities of the occupied boxes, shape ``[num_occupied]`` (data-dependent), summing to 1, in the input's
        floating dtype (the default dtype for integer input; half precision is computed in float32). ``[nan]`` if
        ``trajectory`` contains ``NaN``/``inf``, or if a box index would overflow int64 (``(max - min) / box_size``
        reaching ``2**63``).

    Raises:
        ValueError: If ``trajectory`` is batched (``[B, T, D]`` with ``B != 1``) or complex, or ``box_size`` is not a
            finite real number ``> 0``.
    """
    tensor, _ = _ensure_batched(trajectory)
    if tensor.shape[0] != 1:
        raise ValueError(
            f"box_occupation_probabilities only supports a single (unbatched) trajectory, got batch size {tensor.shape[0]}"
        )
    size = _real_scalar(box_size, "box_size")
    if not 0 < size < float("inf"):
        raise ValueError(f"box_size must be a finite real number > 0, got {box_size!r}")
    out_dtype = _output_dtype(tensor)
    points = tensor[0].to(_work_dtype(tensor.dtype))
    if not torch.isfinite(points).all():
        return torch.full((1,), float("nan"), dtype=out_dtype, device=points.device)
    minimum = points.min(dim=0, keepdim=True).values
    scaled = (points - minimum) / size
    # 2**63 is exact in every float dtype; int64's maximum 2**63 - 1 rounds up to it in float64, letting 2**63 through.
    if scaled.abs().max() >= 2**63:
        return torch.full((1,), float("nan"), dtype=out_dtype, device=points.device)
    box_index = torch.floor(scaled).to(torch.int64)
    _, counts = torch.unique(box_index, dim=0, return_counts=True)
    return (counts.to(points.dtype) / counts.sum()).to(out_dtype)


def box_counting_entropy(trajectory, box_size, *, order: float = 1.0) -> torch.Tensor:
    """Order-``order`` Rényi entropy of the box-occupation distribution at one or more box sizes.

    Gradient class: none (box assignment and occupied-box counting are discrete).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        box_size: Box width; a scalar or an ``[S]`` sweep, each a finite real number ``> 0``.
        order: Rényi order ``>= 0``, forwarded to :func:`torchnlts.entropy.renyi_entropy`.

    Returns:
        Entropy in nats, shape ``[]``/``[S]`` unbatched or ``[B]``/``[S, B]`` batched, in the trajectory's float dtype
        (the default dtype for integer input). ``NaN`` where the box occupation is undefined (see
        :func:`~torchnlts.dimension.box_occupation_probabilities`).

    Raises:
        ValueError: If ``order < 0`` or a ``box_size`` is not a finite real number ``> 0``.
    """
    tensor, ndim = _ensure_batched(trajectory)
    dtype = _output_dtype(tensor)
    tensor = tensor.to(_work_dtype(dtype))
    sizes = _as_sweep(box_size, tensor.device)
    swept = sizes.dim() > 0
    size_list = sizes.reshape(-1).tolist()
    batch_size = tensor.shape[0]

    values = torch.empty(len(size_list), batch_size, dtype=tensor.dtype, device=tensor.device)
    for batch_index in range(batch_size):
        for size_index, size in enumerate(size_list):
            probability = box_occupation_probabilities(tensor[batch_index], size)
            values[size_index, batch_index] = renyi_entropy(probability, order)

    return _restore_sweep(values, swept, ndim).to(dtype)


def generalized_dimension(box_size, entropy: torch.Tensor) -> torch.Tensor:
    """Generalized dimension D_q: OLS slope of the Rényi entropy vs ``-log(box_size)``.

    Reference: Russell, Hanson & Ott (1980), Phys. Rev. Lett. 45, 1175. A plain regression over every
    ``(box_size, entropy)`` pair; for automatic scaling-region detection use :func:`torchnlts.scaling.scaling_slope`
    (pass ``-log(box_size)``, ``entropy``).

    Gradient class: exact (closed-form OLS slope), with the gradient reaching ``entropy`` that of its producer
    (none for :func:`~torchnlts.dimension.box_counting_entropy`).

    Args:
        box_size: Shape ``[S]``, the sweep ``entropy`` was evaluated at.
        entropy: Shape ``[S]`` (unbatched) or ``[S, B]`` (batched), as returned by
            :func:`~torchnlts.dimension.box_counting_entropy` for the same sweep.

    Returns:
        D_q, shape ``[]`` for an unbatched ``entropy`` or ``[B]`` for a batched one, in its float dtype. ``NaN`` if
        ``box_size`` has fewer than two distinct values.

    Raises:
        ValueError: If ``box_size``/``entropy`` do not have the documented shapes with matching sweep lengths.
    """
    neg_log_size = -_safe_log(_as_sweep(box_size, entropy.device))
    return _ols_slope(neg_log_size, entropy)
