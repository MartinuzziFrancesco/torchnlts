"""Detrended fluctuation analysis (DFA): the scaling exponent alpha of a
scalar time series' integrated profile.

Reference: Peng et al. (1994), "Mosaic organization of DNA nucleotides",
Phys. Rev. E 49(2), 1685; polynomial detrending of order ``order``
(DFA-n) from Kantelhardt et al. (2001), Physica A 295, 441.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _ols_slope


def _as_int_list(value, name: str) -> tuple[list[int], bool]:
    """Validate an integer-valued scalar or sweep; returns (values, swept)."""
    # float64 so a Python float like 1.00000001 is not rounded to an
    # integer by float32 before the integrality check (same as higuchi).
    if torch.is_tensor(value) and value.is_complex():
        raise ValueError(f"{name} must be integer-valued, got complex {value.tolist()}")
    value_tensor = torch.as_tensor(value, dtype=torch.float64)
    flat = value_tensor.reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(flat).all() or not torch.equal(flat, flat.round()):
        raise ValueError(f"{name} must be integer-valued, got {flat.tolist()}")
    return [int(item) for item in flat.tolist()], value_tensor.dim() > 0


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
    window_sizes, swept = _as_int_list(window_size, "window_size")
    if any(size <= order + 1 for size in window_sizes):
        raise ValueError(f"every window_size must be > order + 1 = {order + 1}, got {window_sizes}")

    if tensor.is_complex():
        raise ValueError("dfa_fluctuation does not support complex input")
    if not torch.isfinite(tensor).all():
        raise ValueError("dfa_fluctuation requires a finite trajectory (no NaN/inf)")
    dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
    # Half precision has no CPU QR/matmul support and too little range; work in float32.
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    values = tensor.squeeze(-1).to(work_dtype)  # [B, T]
    num_samples = values.shape[-1]
    profile = torch.cumsum(values - values.mean(dim=-1, keepdim=True), dim=-1)
    # A constant series has an identically zero profile; mean removal leaves
    # rounding error, so report F = 0 exactly instead of a roundoff value.
    constant = (values == values[..., :1]).all(dim=-1) if num_samples > 0 else None

    fluctuations = []
    for size in window_sizes:
        num_windows = num_samples // size
        if num_windows == 0:
            fluctuations.append(torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device))
            continue
        windows = profile[..., : num_windows * size].reshape(values.shape[0], num_windows, size)
        residual = windows @ _residual_projector(size, order, values)
        # RMS of residual / scale, times scale: squaring cannot overflow or
        # underflow. F is homogeneous in scale, so detaching it is exact.
        scale = residual.detach().abs().amax(dim=(-2, -1))
        # Masking before the sqrt keeps its infinite derivative at 0 out of backward.
        zero = constant | (scale == 0)
        safe_scale = torch.where(zero, torch.ones_like(scale), scale)
        mean_square = (residual / safe_scale[:, None, None]).pow(2).mean(dim=(-2, -1))
        safe_mean_square = torch.where(zero, torch.ones_like(mean_square), mean_square)
        fluctuations.append(torch.where(zero, torch.zeros_like(scale), safe_mean_square.sqrt() * safe_scale))

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
    log_window_size = torch.log(torch.as_tensor(window_size, dtype=fluctuation.dtype, device=fluctuation.device))
    return _ols_slope(log_window_size, torch.log(fluctuation))
