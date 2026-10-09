"""Power-variation (variogram / madogram) fractal dimension of a scalar
time series, from lagged absolute increments; no delay embedding.

Reference: Gneiting, Sevcikova & Percival (2012), "Estimators of fractal
dimension: assessing the roughness of time series and spatial data",
Statistical Science 27(2), 247-277, eqs. (16)-(18):
``V_p(l) = 1 / (2 * (n - l)) * sum_i |x_{i+l} - x_i|**p`` and
``D = 2 - slope / p``, ``slope`` the OLS slope of ``log V_p(l)`` against
``log l`` (the paper's ``log(l/n)`` differs by a constant, same slope).
``p = 2`` is the variogram, ``p = 1`` the madogram. The paper fits lags
1 and 2 only; as elsewhere in this subpackage the caller picks the lags.
Matches the R package ``fractaldim`` (``fd.Variation``,
``fd.estim.variogram``/``fd.estim.madogram``), checked by reading its source.

Gradient class: exact for ``p = 2``, piecewise for ``p <= 1`` (a zero
increment contributes a zero gradient; for ``p < 1`` the true derivative
there is unbounded).
"""

from __future__ import annotations

import math

import numpy
import torch

from torchnlts._shape import _as_positive_int_list, _as_sweep, _ensure_batched, _is_complex, _mask_nonfinite_rows, _ols_slope, _output_dtype, _safe_log, _safe_scale


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(f"power_variogram only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")


def _variogram_single_lag(values: torch.Tensor, lag: int, power: float) -> torch.Tensor:
    """V_p(l) for a single lag ``l``, ``values`` shape ``[B, T]``."""
    num_points = values.shape[-1]
    num_terms = num_points - lag
    if num_terms < 1:
        return torch.full(values.shape[:-1], float("nan"), dtype=values.dtype, device=values.device)
    magnitude = (values[..., lag:] - values[..., :-lag]).abs()
    # For power < 1, d|x|**p/dx is unbounded at x = 0 and autograd returns
    # NaN there; mask zero increments so they contribute a zero gradient.
    zero = magnitude == 0
    # |dx|**p is summed as (|dx| / s)**p with s the (detached) largest
    # increment, so the powers cannot overflow before the average is taken.
    scale = _safe_scale(magnitude.detach().amax(dim=-1, keepdim=True))
    term = (torch.where(zero, torch.ones_like(magnitude), magnitude) / scale).pow(power)
    term_sum = torch.where(zero, torch.zeros_like(term), term).sum(dim=-1)
    # V = term_sum * s**p / (2 (n - l)), assembled in the log domain so that
    # neither s**p nor any power of the normalization can overflow when V
    # itself is representable. term_sum is 0 only when every increment is 0.
    empty = term_sum == 0
    safe_sum = torch.where(empty, torch.ones_like(term_sum), term_sum)
    log_variogram = torch.log(safe_sum) + power * torch.log(scale.squeeze(-1)) - math.log(2 * num_terms)
    return torch.where(empty, torch.zeros_like(term_sum), torch.exp(log_variogram))


def _validate_power(power):
    """``power`` as a Python float, or a 0-d tensor (keeps its gradient)."""
    if _is_complex(power) or (power.numel() if torch.is_tensor(power) else numpy.size(power)) != 1:
        raise ValueError(f"power must be a finite real number > 0, got {power!r}")
    try:
        scalar = power.reshape(()) if torch.is_tensor(power) else float(power)
    except OverflowError:  # a huge Python int
        scalar = float("inf")
    except (TypeError, ValueError) as error:
        raise ValueError(f"power must be a finite real number > 0, got {power!r}") from error
    if not 0 < scalar < float("inf"):
        raise ValueError(f"power must be a finite real number > 0, got {power!r}")
    return scalar


def power_variogram(trajectory, lag, power: float = 2.0) -> torch.Tensor:
    """Power variogram ``V_p(l) = 1 / (2 * (n - l)) * sum_i |x_{i+l} - x_i|**p``.

    Reference: Gneiting, Sevcikova & Percival (2012), Statist. Sci.
    27(2), eq. (16) (see module docstring for the ``l`` vs ``l/n``
    notational difference). The paper recommends fitting only
    ``l = 1, 2`` (its eq. 18, ``L=2``); this function evaluates whatever
    lags are given, same "caller picks the sweep" convention as
    :func:`torchnlts.dimension.higuchi_length`.

    Gradient class: depends on ``power`` -- see module docstring.

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (univariate
            only). A tensor, numpy array, or pandas Series/DataFrame.
        lag: Lag(s) ``l`` to evaluate ``V_p(l)`` at. A positive integer,
            or a non-empty list/tensor of positive integers to sweep.
            Every value must be integer-valued.
        power: ``p`` in ``V_p(l)``, must be ``> 0``. Default ``2.0``
            (variogram); ``1.0`` is the madogram.

    Returns:
        ``V_p(l)``, shape ``[]``/``[K]`` for an unbatched trajectory or
        ``[B]``/``[K, B]`` for a batched one. ``NaN`` at any lag with
        fewer than one valid term (``l >= n``). ``NaN`` for any row containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1, the trajectory is
            complex, ``power`` is not a finite positive number, ``lag`` is
            empty or complex, or any ``lag`` value is ``< 1`` or non-integer.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor)
    if tensor.is_complex():
        raise ValueError("power_variogram does not support complex input")
    power = _validate_power(power)
    dtype = _output_dtype(tensor)
    # Always float64: increments are raised to power after scaling by the
    # largest one, and in float32 the small ones (and their gradients)
    # underflow long before the result does.
    work_dtype = torch.float64
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(work_dtype))  # [B, T]

    lag_list, swept = _as_positive_int_list(lag, "lag")

    results = torch.stack([_variogram_single_lag(values, value, power) for value in lag_list], dim=0)  # [K, B]
    results = torch.where(nonfinite, float("nan"), results)
    if not swept:
        results = results.squeeze(0)
    if ndim != 3:
        results = results.squeeze(-1)
    return results.to(dtype)


def variogram_dimension(lag, variogram: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    """Power-variation fractal dimension ``D = 2 - slope / p``, ``slope``
    the OLS slope of ``log(V_p(l))`` against ``log(l)``.

    Reference: Gneiting, Sevcikova & Percival (2012), Statist. Sci.
    27(2), eq. (17)/(18). Matches the R package ``fractaldim``'s own
    ``fd.estim.variogram``/``fd.estim.madogram`` (confirmed by reading
    its source, see module docstring).

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``variogram`` -- inherits
    :func:`~torchnlts.dimension.power_variogram`'s own gradient class).

    This is a plain regression over every ``(lag, variogram)`` pair
    given (the paper itself recommends fitting only lags 1 and 2); see
    :func:`torchnlts.scaling.scaling_slope` for automatic scaling-region
    detection (pass ``log(lag)``, ``log(variogram)`` and apply
    ``2 - slope / power``).

    Args:
        lag: Shape ``[K]``, the sweep ``variogram`` was evaluated at.
        variogram: Shape ``[K]`` (unbatched) or ``[K, B]`` (batched), as
            returned by :func:`~torchnlts.dimension.power_variogram` for the same sweep.
        power: ``p`` used to compute ``variogram``; must be ``> 0`` and
            match the ``power`` passed to :func:`~torchnlts.dimension.power_variogram`.

    Returns:
        ``D`` estimate(s), shape ``[]`` for an unbatched ``variogram``
        or ``[B]`` for a batched one. ``NaN`` if ``lag`` has fewer than
        two distinct values, or any ``variogram`` entry is non-positive
        or itself ``NaN`` (e.g. a degenerate lag, see
        :func:`~torchnlts.dimension.power_variogram`'s Returns section, or a constant series,
        whose zero variogram gives ``log(0) = -inf``).

    Raises:
        ValueError: If ``power`` is not a finite positive number, or ``lag``/``variogram`` do not
            have the documented shapes with matching sweep lengths.
    """
    power = _validate_power(power)
    log_lag = _safe_log(_as_sweep(lag, variogram.device))
    slope = _ols_slope(log_lag, _safe_log(variogram))
    # A NaN slope divided by a differentiable power would put NaN into its gradient.
    valid = torch.isfinite(slope)
    dimension = 2.0 - torch.where(valid, slope, torch.zeros_like(slope)) / power
    return torch.where(valid, dimension, torch.full_like(dimension, float("nan"))).to(_output_dtype(variogram))
