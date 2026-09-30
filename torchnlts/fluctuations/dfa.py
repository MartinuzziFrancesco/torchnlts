"""Detrended fluctuation analysis (DFA): the scaling exponent alpha of a
scalar time series' integrated profile.

Reference: Peng et al. (1994), "Mosaic organization of DNA nucleotides",
Phys. Rev. E 49(2), 1685; polynomial detrending of order ``order``
(DFA-n) from Kantelhardt et al. (2001), Physica A 295, 441.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_positive_int_list, _as_sweep, _ensure_batched, _ols_slope, _output_dtype, _power_of_two_scale, _safe_log


def _residual_projector(window_size: int, order: int, like: torch.Tensor) -> torch.Tensor:
    """``I - Q Q^T`` for the degree-``order`` polynomial basis on one window."""
    # Legendre basis on [-1, 1], orthogonalized in float64: spans the same
    # polynomials as a monomial basis but stays well conditioned at high order.
    abscissa = torch.linspace(-1.0, 1.0, window_size, dtype=torch.float64, device=like.device)
    legendre = torch.special.legendre_polynomial_p(abscissa.unsqueeze(-1), torch.arange(order + 1, device=like.device))
    basis, _ = torch.linalg.qr(legendre)
    identity = torch.eye(window_size, dtype=torch.float64, device=like.device)
    return (identity - basis @ basis.T).to(like.dtype)


def dfa_fluctuation(trajectory, window_size, order: int = 1) -> torch.Tensor:
    """DFA fluctuation function F(n) at one or more window sizes n.

    Peng et al. (1994): profile ``y = cumsum(x - mean(x))``, split into
    ``T // n`` non-overlapping windows from the start (trailing ``T % n``
    samples are dropped), least-squares polynomial of degree ``order``
    removed per window, ``F(n) = sqrt(mean(residual**2))`` over all kept samples.

    Gradient class: exact (detrending is linear in the data; not
    differentiable where F(n) = 0).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        window_size: Integer window size n, or a non-empty sweep of them.
        order: Degree of the per-window detrending polynomial (DFA-1 is 1).

    Returns:
        F(n), shape ``[]``/``[K]`` unbatched or ``[B]``/``[K, B]`` batched.
        ``NaN`` at any ``window_size > T``. Exactly ``0`` for a constant series,
        with a zero (not ``NaN``) gradient wherever F(n) is exactly ``0``.

    Raises:
        ValueError: If the trajectory is complex or non-finite, the
            feature dim is not 1, ``order`` is not a
            non-negative integer, ``window_size`` is empty, or any
            ``window_size`` is non-integer or ``<= order + 1`` (the fit
            would be exact and F(n) identically 0).
    """
    tensor, ndim = _ensure_batched(trajectory)
    if tensor.shape[-1] != 1:
        raise ValueError(f"dfa_fluctuation only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    window_sizes, swept = _as_positive_int_list(window_size, "window_size")
    if any(size <= order + 1 for size in window_sizes):
        raise ValueError(f"every window_size must be > order + 1 = {order + 1}, got {window_sizes}")

    if tensor.is_complex():
        raise ValueError("dfa_fluctuation does not support complex input")
    if not torch.isfinite(tensor).all():
        raise ValueError("dfa_fluctuation requires a finite trajectory (no NaN/inf)")
    dtype = _output_dtype(tensor)
    # Always float64: detrending cancels the profile's polynomial part, and
    # its rounding error scales with the largest sample, not the residual.
    work_dtype = torch.float64
    values = tensor.squeeze(-1).to(work_dtype)  # [B, T]
    num_samples = values.shape[-1]
    # F is homogeneous of degree 1 in the amplitude: compute it on values
    # scaled to (-2, 2) and scale back, so the profile cannot overflow. The
    # scale is a detached power of two: derivatives stay exact and the
    # scaling is exact, so offsets cancel exactly below.
    scale = _power_of_two_scale(values[..., None]).reshape(-1)
    values = values / scale[:, None]
    # A constant series has an identically zero profile; mean removal leaves
    # rounding error, so report F = 0 exactly instead of a roundoff value.
    constant = (values == values[..., :1]).all(dim=-1) if num_samples > 0 else None

    fluctuations = []
    for size in window_sizes:
        num_windows = num_samples // size
        if num_windows == 0:
            fluctuations.append(torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device))
            continue
        # Each window's profile is the global profile minus a constant (the
        # sum before the window) and a linear term (the mean times the
        # position). Detrending of order >= 1 removes both, so integrating
        # each window from its own start, after subtracting any per-window
        # offset, gives the same residuals; the offset is subtracted in the
        # input's units, where x - c is exact for nearby values, so a large
        # common offset or dominant dropped samples cause no cancellation.
        # Order 0 keeps the linear term: subtract the global mean, after the
        # first sample so a large common offset cancels exactly.
        kept = values[..., : num_windows * size].reshape(values.shape[0], num_windows, size)
        if order == 0:
            first = values[:, :1, None]
            kept = kept - first - (values - values[:, :1]).mean(dim=-1)[:, None, None]
        else:
            kept = kept - kept.mean(dim=-1, keepdim=True)
        windows = torch.cumsum(kept, dim=-1)
        residual = windows @ _residual_projector(size, order, values)
        # Residuals are also rescaled by their own maximum: the kept windows
        # can be far smaller than the row maximum, and their squares underflow.
        residual_scale = residual.detach().abs().amax(dim=(-2, -1))
        # Masking before the sqrt keeps its infinite derivative at 0 out of backward.
        zero = constant | (residual_scale == 0)
        residual_scale = torch.where(zero, torch.ones_like(residual_scale), residual_scale)
        mean_square = (residual / residual_scale[:, None, None]).pow(2).mean(dim=(-2, -1))
        safe_mean_square = torch.where(zero, torch.ones_like(mean_square), mean_square)
        normalized = torch.where(zero, torch.zeros_like(mean_square), safe_mean_square.sqrt() * residual_scale)
        fluctuations.append(normalized * scale)

    result = torch.stack(fluctuations, dim=0).to(dtype)  # [K, B]
    if not swept:
        result = result.squeeze(0)
    if ndim != 3:
        result = result.squeeze(-1)
    return result


def dfa_exponent(window_size, fluctuation: torch.Tensor) -> torch.Tensor:
    """DFA scaling exponent alpha: OLS slope of ``log(F(n))`` vs ``log(n)``.

    Peng et al. (1994): F(n) ~ n**alpha; alpha=0.5 is uncorrelated
    (white) noise, alpha=1.5 is Brownian motion.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``fluctuation`` inherits
    :func:`dfa_fluctuation`'s own exact gradient class).

    No automatic scaling-region detection: fits every ``(window_size,
    fluctuation)`` pair given, same convention as ``higuchi_dimension``.

    Args:
        window_size: Shape ``[K]``, the sweep ``fluctuation`` was
            evaluated at.
        fluctuation: Shape ``[K]`` (unbatched) or ``[K, B]`` (batched),
            as returned by :func:`dfa_fluctuation` for the same sweep.

    Returns:
        alpha estimate(s), shape ``[]`` for unbatched ``fluctuation`` or
        ``[B]`` for batched. ``NaN`` if ``window_size`` has fewer than
        two distinct values, or any ``fluctuation`` entry is
        non-positive or ``NaN``.

    Raises:
        ValueError: If ``window_size``/``fluctuation`` do not have the
            documented shapes with matching sweep lengths.
    """
    log_window_size = _safe_log(_as_sweep(window_size, fluctuation.device))
    return _ols_slope(log_window_size, _safe_log(fluctuation)).to(_output_dtype(fluctuation))
