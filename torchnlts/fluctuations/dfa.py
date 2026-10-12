"""Detrended fluctuation analysis (DFA): the scaling exponent alpha of a scalar time series' integrated profile.

Reference: Peng et al. (1994), Phys. Rev. E 49(2), 1685; polynomial detrending of order ``order`` (DFA-n) from
Kantelhardt et al. (2001), Physica A 295, 441. Windows are non-overlapping and start at the first sample, and the
trailing ``T % n`` samples are dropped, as in nolds (``overlap=False``) and antropy. FluctuationAnalysis.jl agrees with
``dfa(...; bidirectional=false)``. Computed in float64 (float32 on Apple MPS, which has no float64).
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _as_int_list,
    _as_sweep,
    _ensure_batched,
    _high_precision_dtype,
    _mask_nonfinite_rows,
    _ols_slope,
    _output_dtype,
    _require_univariate,
    _restore_sweep,
    _safe_log,
)


def _detrending_basis(window_size: int, order: int, device: torch.device) -> torch.Tensor:
    """Orthonormal basis ``[window_size, order + 1]`` of the degree-``order`` polynomials on one window."""
    # Legendre basis on [-1, 1] orthogonalized in the high-precision dtype; well conditioned at high order.
    abscissa = torch.linspace(-1.0, 1.0, window_size, dtype=_high_precision_dtype(device), device=device)
    legendre = torch.special.legendre_polynomial_p(abscissa.unsqueeze(-1), torch.arange(order + 1, device=device))
    return torch.linalg.qr(legendre).Q


def dfa_fluctuation(trajectory, window_size, *, order: int = 1) -> torch.Tensor:
    """DFA fluctuation function F(n) at one or more window sizes n.

    Reference: Peng et al. (1994), Phys. Rev. E 49, 1685. The profile ``y = cumsum(x - mean(x))`` is split into
    ``T // n`` windows, a least-squares polynomial of degree ``order`` is removed per window, and
    ``F(n) = sqrt(mean(residual**2))`` over all kept samples.

    Gradient class: exact (detrending is linear in the data), not differentiable where F(n) = 0.

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.
        window_size: Window size n, or a non-empty 1-D sweep of them; integer-valued numbers ``> order + 1`` (``16.0``
            is accepted).
        order: Degree of the per-window detrending polynomial, ``>= 0`` (DFA-1 is 1).

    Returns:
        F(n), shape ``[]``/``[K]`` unbatched or ``[B]``/``[K, B]`` batched, in the trajectory's float dtype (the default
        dtype for integer input). ``NaN`` at any ``window_size > T`` and for any row containing NaN/inf. Exactly ``0``
        for a constant series, with a zero (not ``NaN``) gradient wherever F(n) is exactly ``0``.

    Raises:
        ValueError: If the trajectory is complex or its feature dim is not 1, ``order`` is not a non-negative
            integer, or ``window_size`` is empty, non-integer or ``<= order + 1`` (the fit would be exact).
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor, "dfa_fluctuation")
    order = _as_int(order, "order")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    window_sizes, swept = _as_int_list(window_size, "window_size")
    if any(size <= order + 1 for size in window_sizes):
        raise ValueError(f"every window_size must be > order + 1 = {order + 1}, got {window_sizes}")

    dtype = _output_dtype(tensor)
    # Detrending cancels the profile's polynomial part, so rounding error scales with the largest sample.
    work_dtype = _high_precision_dtype(tensor.device)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(work_dtype))  # [B, T]
    num_samples = values.shape[-1]
    # A constant series has an exactly zero profile; report F = 0 rather than rounding error.
    constant = (values == values[..., :1]).all(dim=-1) if num_samples > 0 else None

    fluctuations = []
    for size in window_sizes:
        num_windows = num_samples // size
        if num_windows == 0:
            fluctuations.append(torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device))
            continue
        # Subtract offsets before the cumsum for exactness; order 0 keeps the linear term, so use the global mean.
        kept = values[..., : num_windows * size].reshape(values.shape[0], num_windows, size)
        if order == 0:
            first = values[:, :1, None]
            kept = kept - first - (values - values[:, :1]).mean(dim=-1)[:, None, None]
        else:
            kept = kept - kept.mean(dim=-1, keepdim=True)
        windows = torch.cumsum(kept, dim=-1)
        basis = _detrending_basis(size, order, values.device)
        residual = windows - (windows @ basis) @ basis.T
        # Rescale residuals by their own maximum: kept windows can be far smaller than the row maximum and underflow.
        residual_scale = residual.detach().abs().amax(dim=(-2, -1))
        # Masking before the sqrt keeps its infinite derivative at 0 out of backward.
        zero = constant | (residual_scale == 0)
        residual_scale = torch.where(zero, torch.ones_like(residual_scale), residual_scale)
        mean_square = (residual / residual_scale[:, None, None]).pow(2).mean(dim=(-2, -1))
        safe_mean_square = torch.where(zero, torch.ones_like(mean_square), mean_square)
        normalized = torch.where(zero, torch.zeros_like(mean_square), safe_mean_square.sqrt() * residual_scale)
        fluctuations.append(normalized)

    result = torch.where(nonfinite, float("nan"), torch.stack(fluctuations, dim=0))  # [K, B]
    return _restore_sweep(result, swept, ndim).to(dtype)


def dfa_exponent(window_size, fluctuation: torch.Tensor) -> torch.Tensor:
    """DFA scaling exponent alpha: OLS slope of ``log(F(n))`` vs ``log(n)``.

    Reference: Peng et al. (1994), Phys. Rev. E 49, 1685: F(n) ~ n**alpha, with alpha = 0.5 for uncorrelated (white)
    noise and 1.5 for Brownian motion. A plain regression over every ``(window_size, fluctuation)`` pair; for automatic
    scaling-region detection use :func:`torchnlts.scaling.scaling_slope` (pass ``log(window_size)``,
    ``log(fluctuation)``).

    Gradient class: exact (closed-form OLS slope), with the gradient reaching ``fluctuation`` that of
    :func:`~torchnlts.fluctuations.dfa_fluctuation`.

    Args:
        window_size: Shape ``[K]``, the sweep ``fluctuation`` was evaluated at.
        fluctuation: Shape ``[K]`` (unbatched) or ``[K, B]`` (batched), as returned by
            :func:`~torchnlts.fluctuations.dfa_fluctuation` for the same sweep.

    Returns:
        alpha, shape ``[]`` for an unbatched ``fluctuation`` or ``[B]`` for a batched one, in its float dtype. ``NaN``
        if ``window_size`` has fewer than two distinct values or any ``fluctuation`` entry is non-positive or ``NaN``.

    Raises:
        ValueError: If ``window_size``/``fluctuation`` do not have the documented shapes with matching sweep lengths.
    """
    log_window_size = _safe_log(_as_sweep(window_size, fluctuation.device))
    return _ols_slope(log_window_size, _safe_log(fluctuation)).to(_output_dtype(fluctuation))
