"""Higuchi fractal dimension: roughness of a scalar time series' own
graph, estimated directly from the data with no delay embedding.

Reference: Higuchi, T. (1988), "Approach to an irregular time series on
the basis of the fractal theory", Physica D 31(2), 277-283: for each
offset m = 1, ..., k, subsample the series at stride k starting from
m, sum the subsequence's consecutive absolute differences, and
normalize by ``(N - 1) / (floor((N - m) / k) * k**2)`` to get that
offset's curve length L_m(k); L(k) is the average of L_m(k) over m.
The dimension D_H, in [1, 2] *in theory* (not a guarantee for any
particular finite-sample sweep -- see :func:`higuchi_dimension`'s own
Returns section), is the OLS slope of ``log(L(k))`` against
``log(1/k)`` over a k sweep -- the same "caller picks the sweep, no
automatic scaling-region detection" convention as
:func:`torchnlts.dimension.correlation.correlation_dimension`.

:func:`higuchi_length` matches FractalDimensions.jl's own
``higuchi_length`` exactly (confirmed both by independent
hand-derivation and by direct Julia execution for a small example):
its ``k**2`` (not ``k``) denominator is Higuchi's own original
formula, confirmed against a secondary source quoting the paper's
equation directly, not a package-specific convention this module
happens to inherit uncritically. :func:`higuchi_dimension` does
*not* generally match Julia's own ``higuchi_dim`` the same way, even
though it happens to on some inputs -- Julia's own function picks an
automatic linear-region subset of the sweep (``linear_region``)
before fitting, while this module always fits every point given (the
same "caller picks the sweep, no automatic scaling-region detection"
convention as ``correlation_dimension``/``generalized_dimension``);
confirmed directly with a real counterexample (a clean sine wave over
strides 1-20 gives ``1.274`` here against Julia's own
region-restricted ``1.052``, not floating-point noise). A ``k`` large
enough that some offset m has no valid
subsequence step (``floor((N - m) / k) < 1``) makes Julia's own
literal formula divide by zero for that one offset -- confirmed
directly to propagate to a ``NaN`` L(k), not an error -- replicated
here explicitly (see :func:`higuchi_length`'s Returns section) rather
than left to an incidental floating-point division.

Univariate only, same reasoning as
:mod:`torchnlts.entropy.permutation`/:mod:`torchnlts.entropy.lempel_ziv`:
Higuchi's own definition is the roughness of the graph of a *scalar*
function x(t), with no stated convention for combining multiple
channels into one curve.

Gradient class: piecewise. Unlike this subpackage's other new
additions (box-counting, permutation/dispersion patterns, Lempel-Ziv
parsing -- all fundamentally discrete/combinatorial, gradient class
``none``), Higuchi's curve length is built entirely from subtraction,
``abs()``, and summation over a *fixed* (data-independent) index
pattern once ``k`` is chosen -- differentiable almost everywhere, the
same "piecewise" caveat (a kink at an exact tie between two indexed
points) already documented for
:func:`torchnlts.lyapunov.rosenstein.rosenstein_divergence`/
:func:`torchnlts.lyapunov.kantz.kantz_divergence`'s own
absolute-difference-based curves.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_positive_int_list, _as_sweep, _ensure_batched, _mask_nonfinite_rows, _ols_slope, _output_dtype, _safe_log, _work_dtype


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(f"higuchi_length only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")


def _higuchi_length_single_k(values: torch.Tensor, k: int) -> torch.Tensor:
    """L(k) for a single stride ``k``, ``values`` shape ``[B, N]``."""
    num_points = values.shape[-1]
    offset_lengths = []
    for offset in range(k):
        num_steps = (num_points - 1 - offset) // k
        if num_steps < 1:
            # Matches Julia's own natural failure mode for this offset
            # (a literal division by zero there propagates to NaN, not
            # an error) -- poisons the whole L(k), not just this offset.
            return torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device)
        indices = offset + k * torch.arange(num_steps + 1, device=values.device)
        subsequence = values[..., indices]
        length = (subsequence[..., 1:] - subsequence[..., :-1]).abs().sum(dim=-1)
        normalization = (num_points - 1) / (num_steps * k * k)
        offset_lengths.append(length * normalization)
    return torch.stack(offset_lengths, dim=0).mean(dim=0)


def higuchi_length(trajectory, k) -> torch.Tensor:
    """Higuchi's average curve length L(k), at one or more strides.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (a
            multivariate ``D > 1`` trajectory is rejected -- see module
            docstring). A tensor, numpy array, or pandas Series/DataFrame.
        k: Stride(s) to evaluate L(k) at. A positive integer, or a
            non-empty list/tensor of positive integers to sweep. Every
            value must be integer-valued -- ``k`` indexes a stride, not
            a continuous quantity like ``radius``/``box_size`` elsewhere
            in this codebase, so a non-integer value (e.g. from
            mistakenly reusing a swept float sweep) is rejected rather
            than silently truncated.

    Returns:
        L(k), shape ``[]``/``[K]`` for an unbatched trajectory or
        ``[B]``/``[K, B]`` for a batched one. ``NaN`` at any ``k`` large
        enough that some offset ``m`` in ``0, ..., k - 1`` has no valid
        subsequence step (``(T - 1 - m) // k < 1``) -- see module
        docstring; this includes an empty or otherwise too-short
        ``trajectory`` (every ``k`` degenerate), which falls into the
        same case rather than being separately rejected -- unlike
        FractalDimensions.jl's own ``higuchi_length``, which returns
        ``0.0`` (not ``NaN``) for a literally empty input, a confirmed
        deviation this module doesn't replicate (matches this
        subpackage's general preference for ``NaN`` over a
        finite-looking placeholder on a degenerate input). ``NaN`` for any row containing NaN/inf.

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1,
            the trajectory is complex, ``k`` is
            empty or complex, or any ``k`` value is ``< 1`` or non-integer.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor)
    if tensor.is_complex():
        raise ValueError("higuchi_length does not support complex input")
    dtype = _output_dtype(tensor)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(_work_dtype(dtype)))  # [B, T]

    k_list, swept = _as_positive_int_list(k, "k")

    lengths = torch.stack([_higuchi_length_single_k(values, value) for value in k_list], dim=0)  # [K, B]
    lengths = torch.where(nonfinite, float("nan"), lengths)
    if not swept:
        lengths = lengths.squeeze(0)
    if ndim != 3:
        lengths = lengths.squeeze(-1)
    return lengths.to(dtype)


def higuchi_dimension(k, length: torch.Tensor) -> torch.Tensor:
    """Higuchi fractal dimension D_H: OLS slope of ``log(L(k))`` vs
    ``log(1/k)``.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``length`` -- inherits
    :func:`higuchi_length`'s ``piecewise``, matching
    :func:`torchnlts.dimension.correlation.correlation_dimension`'s own
    framing for its analogous OLS-slope step).

    This is a plain regression over every ``(k, length)`` pair given;
    see :func:`torchnlts.scaling.scaling_slope` for automatic
    scaling-region detection (pass ``-log(k)``, ``log(length)``).

    Reference: Higuchi (1988) (D_H as the log-log slope of L(k) against
    1/k).

    Args:
        k: Shape ``[K]``, the sweep ``length`` was evaluated at.
        length: Shape ``[K]`` (unbatched trajectory) or ``[K, B]``
            (batched trajectory), as returned by :func:`higuchi_length`
            for the same sweep.

    Returns:
        D_H estimate(s), shape ``[]`` for an unbatched ``length`` or
        ``[B]`` for a batched one. Not guaranteed to lie in ``[1, 2]``
        for an arbitrary finite ``k`` sweep, even though the true,
        limiting Higuchi dimension is theoretically bounded there --
        confirmed directly: this module's own hand-derived test example
        (an 8-point series at a 2-stride sweep) gives ``3.263``. ``NaN``
        if ``k`` has fewer than two distinct values.
        ``NaN`` also if any ``length`` entry is non-positive or itself
        ``NaN`` (e.g. a degenerate ``k``, see :func:`higuchi_length`'s
        own Returns section).

    Raises:
        ValueError: If ``k``/``length`` do not have the documented
            shapes with matching sweep lengths.
    """
    neg_log_k = -_safe_log(_as_sweep(k, length.device))
    return _ols_slope(neg_log_k, _safe_log(length)).to(_output_dtype(length))
