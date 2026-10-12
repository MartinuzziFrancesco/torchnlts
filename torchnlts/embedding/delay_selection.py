"""Embedding-delay selection for a scalar time series.

Reference: Fraser, A. M. & Swinney, H. L. (1986), "Independent coordinates for strange attractors from mutual
information", Phys. Rev. A 33, 1134-1140 (first minimum of the self mutual information); Rosenstein, M. T., Collins,
J. J. & De Luca, C. J. (1994), Physica D 73, 82-98 (autocorrelation criteria).

Conventions:

- ``auto_mutual_information`` uses an equal-width ``num_bins`` x ``num_bins`` histogram over the range of the first
  ``T - max_delay`` samples, the same pairs for every delay (the fixed-width variant of DelayEmbeddings.jl's
  ``selfmutualinfo``), instead of Fraser and Swinney's adaptive partition, whose data-dependent shapes cannot be
  vectorized. Unlike ``selfmutualinfo(...; nbins=num_bins)``, the delayed series is binned with the edges of the
  original (Julia re-edges it by the largest sample of each bin) and the delayed marginal is counted from the delayed
  samples (Julia reuses the original's counts). The two agree exactly at delay 0 and for a series periodic over the
  pair window; otherwise they differ by about 0.01 nats at 200 points and 0.001 at 500. Julia reports bits; this
  module reports nats.
- ``estimate_delay`` follows DelayEmbeddings.jl's ``estimate_delay`` search rules for ``mi_min``, ``ac_zero`` and
  ``ac_min`` (the ``*_min`` methods never return delay 0). ``ac_1/e`` is this library's own threshold criterion.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _power_of_two_scale,
    _require_univariate,
    _restore_batch,
    _restore_sweep,
    _work_dtype,
)

_METHODS = ("mi_min", "ac_zero", "ac_min", "ac_1/e")


def _univariate_rows(trajectory, max_delay, name: str) -> tuple[torch.Tensor, int, int, torch.dtype]:
    """Work-dtype ``[B, T]`` values, original ndim, validated ``max_delay``, output dtype."""
    max_delay = _as_int(max_delay, "max_delay")
    batched, ndim = _ensure_batched(trajectory)
    _require_univariate(batched, name)
    out_dtype = _output_dtype(batched)
    if max_delay < 1 or max_delay >= batched.shape[1]:
        raise ValueError(f"max_delay must satisfy 1 <= max_delay < T = {batched.shape[1]}, got {max_delay}")
    return batched[..., 0].to(_work_dtype(batched.dtype)), ndim, max_delay, out_dtype


def _autocorrelation_rows(values: torch.Tensor, max_delay: int) -> torch.Tensor:
    """``[max_delay + 1, B]`` autocorrelation in the work dtype; ``NaN`` columns if constant or non-finite."""
    values, nonfinite = _mask_nonfinite_rows(values)
    invalid = nonfinite | (values.detach().amax(dim=1) == values.detach().amin(dim=1))
    # Translate by the first sample (keeps small variations of a large offset) and divide by a detached
    # power of two (the ratio is scale invariant; squares cannot overflow or underflow).
    values = values - values[:, :1]
    values = values / _power_of_two_scale(values[..., None])[:, :, 0]
    centered = values - values.mean(dim=1, keepdim=True)
    denominator = (centered * centered).sum(dim=1)
    safe = torch.where(invalid, torch.ones_like(denominator), denominator)
    length = values.shape[1]
    lags = [(centered[:, : length - lag] * centered[:, lag:]).sum(dim=1) for lag in range(max_delay + 1)]
    result = torch.stack(lags, dim=0) / safe
    return torch.where(invalid, torch.full_like(result, float("nan")), result)


def _ami_rows(values: torch.Tensor, max_delay: int, num_bins: int) -> torch.Tensor:
    """``[max_delay + 1, B]`` self mutual information in nats, in the work dtype; ``NaN`` for non-finite rows."""
    num_bins = _as_int(num_bins, "num_bins")
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    values, nonfinite = _mask_nonfinite_rows(values.detach())
    num_pairs = values.shape[1] - max_delay
    window = values[:, :num_pairs]
    low = window.amin(dim=1, keepdim=True)
    high = window.amax(dim=1, keepdim=True)
    fractions = torch.arange(1, num_bins, dtype=values.dtype, device=values.device) / num_bins
    edges = low + (high - low) * fractions
    # Samples above an edge move up a bin; out-of-window samples clamp to the end bins.
    bins = (values[:, :, None] > edges[:, None, :]).sum(dim=-1)
    one_hot = torch.nn.functional.one_hot(bins, num_bins).to(values.dtype)
    marginal_x = one_hot[:, :num_pairs].sum(dim=1)
    results = []
    for delay in range(max_delay + 1):
        shifted = one_hot[:, delay : delay + num_pairs]
        joint = torch.einsum("bnk,bnl->bkl", one_hot[:, :num_pairs], shifted)
        marginal_y = shifted.sum(dim=1)
        outer = marginal_x[:, :, None] * marginal_y[:, None, :]
        occupied = joint > 0
        safe_outer = torch.where(occupied, outer, torch.ones_like(outer))
        ratio = torch.where(occupied, joint * num_pairs / safe_outer, torch.ones_like(joint))
        results.append((joint / num_pairs * torch.log(ratio)).sum(dim=(1, 2)))
    result = torch.stack(results, dim=0)
    return torch.where(nonfinite, torch.full_like(result, float("nan")), result)


def autocorrelation(trajectory, max_delay: int) -> torch.Tensor:
    """Sample autocorrelation at delays ``0..max_delay``.

    ``c(k) = sum_t (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2`` with ``m`` the mean of the whole series (StatsBase
    ``autocor``, ``demean=true``).

    Gradient class: exact.

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]``, or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        max_delay: Largest delay, ``1 <= max_delay < T``.

    Returns:
        Shape ``[max_delay + 1]`` (or ``[max_delay + 1, B]``), in the input's floating dtype (the default dtype for
        integer input). ``NaN`` for a constant or non-finite row.

    Raises:
        ValueError: If ``trajectory`` is not univariate or is complex, or ``max_delay`` is not an integer in
            ``[1, T)``.
    """
    values, ndim, max_delay, out_dtype = _univariate_rows(trajectory, max_delay, "autocorrelation")
    return _restore_sweep(_autocorrelation_rows(values, max_delay).to(out_dtype), True, ndim)


def auto_mutual_information(trajectory, max_delay: int, *, num_bins: int = 16) -> torch.Tensor:
    """Histogram self mutual information I(x_t; x_{t+k}) in nats, ``k = 0..max_delay``.

    Reference: Fraser & Swinney (1986), Phys. Rev. A 33, 1134, with an equal-width ``num_bins`` x ``num_bins``
    histogram over the range of the first ``T - max_delay`` samples instead of their adaptive partition. It agrees
    with DelayEmbeddings.jl's ``selfmutualinfo(...; nbins=num_bins)`` at delay 0 and otherwise to about 0.01 nats at
    200 points (the delayed series is binned with the original's edges and its marginal counted from the delayed
    samples).

    Gradient class: none (histogram bin counts are piecewise constant).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]``, or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        max_delay: Largest delay, ``1 <= max_delay < T``.
        num_bins: Histogram bins per axis, ``>= 2``.

    Returns:
        Shape ``[max_delay + 1]`` (or ``[max_delay + 1, B]``), in the input's floating dtype (the default dtype for
        integer input). ``0`` for a constant row, ``NaN`` for a non-finite row.

    Raises:
        ValueError: If ``trajectory`` is not univariate or is complex, ``max_delay`` is not an integer in ``[1, T)``,
            or ``num_bins`` is not an integer ``>= 2``.
    """
    values, ndim, max_delay, out_dtype = _univariate_rows(trajectory, max_delay, "auto_mutual_information")
    return _restore_sweep(_ami_rows(values, max_delay, num_bins).to(out_dtype), True, ndim)


def estimate_delay(trajectory, max_delay: int, *, method: str = "mi_min", num_bins: int = 16) -> torch.Tensor:
    """Embedding delay in ``1..max_delay`` from a criterion on the self mutual information or the autocorrelation.

    - ``"mi_min"``: first local minimum of :func:`~torchnlts.embedding.auto_mutual_information` (Fraser & Swinney).
    - ``"ac_min"``: first local minimum of :func:`~torchnlts.embedding.autocorrelation`.
    - ``"ac_zero"``: first delay with autocorrelation ``<= 0``.
    - ``"ac_1/e"``: first delay with autocorrelation ``< 1/e``.

    A minimum is the first delay whose successor is not lower, so one at ``max_delay - 1`` or ``max_delay`` is not
    detected (except that ``max_delay == 2`` returns 1 if the curve does not decrease from delay 1 to 2). When no delay
    qualifies, ``max_delay`` is returned. DelayEmbeddings.jl does the same except that it raises ``BoundsError`` for
    ``max_delay == 1`` (minimum methods and ``ac_zero``) and for a decreasing curve at ``max_delay == 2``.

    Gradient class: none (integer output).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]``, or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        max_delay: Largest delay searched, ``1 <= max_delay < T``.
        method: One of the methods above.
        num_bins: Histogram bins per axis for ``"mi_min"``, ``>= 2``.

    Returns:
        Integer tensor of shape ``[]`` (or ``[B]``). ``-1`` for a non-finite or constant row.

    Raises:
        ValueError: If ``method`` is unknown, ``trajectory`` is not univariate or is complex, ``max_delay`` is not
            an integer in ``[1, T)``, or ``num_bins`` is not an integer ``>= 2``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    values, ndim, max_delay, _ = _univariate_rows(trajectory, max_delay, "estimate_delay")
    if method == "mi_min":
        curve = _ami_rows(values, max_delay, num_bins)  # [max_delay + 1, B]
    else:
        curve = _autocorrelation_rows(values, max_delay)
    series = curve[1:]
    constant = values.amax(dim=1) == values.amin(dim=1)
    invalid = ~torch.isfinite(curve).all(dim=0) | constant
    count = series.shape[0]
    if method in ("mi_min", "ac_min"):
        # Julia's rule: compare successive values for the first max(count - 2, 1) delays.
        tests = max(count - 2, 1) if count >= 2 else 0
        hit = series[1 : 1 + tests] >= series[:tests]
    elif method == "ac_zero":
        hit = series[:-1] <= 0
    else:
        hit = series < math.exp(-1.0)
    # argmax of the flags is 0 when none is set; the extra True row makes "none" resolve to max_delay.
    sentinel = torch.ones((1, curve.shape[1]), dtype=torch.bool, device=curve.device)
    first = torch.cat([hit, sentinel], dim=0).to(torch.int8).argmax(dim=0)
    delay = torch.where(first < hit.shape[0], first + 1, torch.full_like(first, count))
    delay = torch.where(invalid, torch.full_like(delay, -1), delay)
    return _restore_batch(delay, ndim)
