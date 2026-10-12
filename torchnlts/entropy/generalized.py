"""Rényi and Tsallis entropy of a given discrete probability distribution.

These are the back end for the trajectory-taking entropies: pass any distribution, e.g. the one from
:func:`~torchnlts.entropy.ordinal_pattern_probabilities`, and get an entropy of tunable order instead of Shannon's.

References:
  - Rényi, A. (1961), Proc. Fourth Berkeley Symp. Math. Stat. Probab., Vol. 1, 547-561.
  - Tsallis, C. (1988), J. Stat. Phys. 52, 479-487.

Definitions: ``H_alpha = log(sum p ** alpha) / (1 - alpha)`` and ``S_q = k * (1 - sum p ** q) / (q - 1)``. Both reduce
to Shannon entropy at order 1, where the formulas have a removable 0/0 singularity. Rényi's order 0 (Hartley entropy,
the log of the number of nonzero probabilities) and infinity (min-entropy, ``-log max p``) are further special cases.

Conventions matching ComplexityMeasures.jl: order 1 is detected within a relative tolerance of ``sqrt(eps(Float64))``
(about 1.5e-8), independent of the dtype of ``probability``; Rényi's order 0 only by exact equality (Julia's
``isapprox`` has ``atol=0``); Tsallis accepts any real ``q``. Deviation: Julia's ``Tsallis`` drops ``k`` inside its
``q ≈ 1`` window, a discontinuity when ``k != 1``; here the window returns the limit, ``k`` times Shannon entropy.

Numerics: near order 1 the power sum is evaluated through ``expm1``/``log1p``, which avoids the cancellation in
``sum(p ** order) - 1``. Where that form saturates (order far from 1 on a peaked distribution) a ``logsumexp`` form
is used instead, selected per row with ``torch.where`` (no data-dependent Python branches, so ``torch.vmap`` works);
Tsallis folds ``k`` and ``1 / (q - 1)`` into the exponent to extend its range. Zero probabilities are excluded by
substituting a safe base (1) before the log or power and masking the result, so ``0 ** negative`` never occurs. A row
containing ``NaN`` returns ``NaN`` (otherwise ``NaN > 0`` is False and the entry would read as zero probability).

Limitation: a single ``p ** order`` that already exceeds the dtype's range (e.g. ``0.5 ** -4096`` in float64), or a
Tsallis exponent within rounding of the overflow boundary (e.g. ``q=-10, k=5e305`` in float64), overflows; there is no
extended-precision fallback. For probabilities near the subnormal floor (e.g. ``1e-310`` in float64) at an extreme
order the gradient, not the forward value, can be wrong.

Gradient class: smooth for Tsallis and for Rényi at every ``alpha`` except 0, but only on a fixed support. A
probability that is exactly zero gets gradient 0, a convention rather than the one-sided limit (unbounded for order
below 1; a softmax can underflow entries to zero too). Rényi at ``alpha = 0`` depends only on which probabilities are
nonzero: its output has no gradient path (``requires_grad=False``), which matters only if it is used alone as a loss.
``alpha = inf`` is a max, with the usual subgradient at ties.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _shannon_entropy


def _isclose_to_order(value: float, target: float) -> bool:
    """Whether ``value`` is within ``sqrt(eps(Float64))`` (relative) of ``target``, as Julia's default ``isapprox``."""
    tolerance = math.sqrt(2.220446049250313e-16)
    return math.isclose(value, target, rel_tol=tolerance, abs_tol=0.0)


def _safe_log(probability: torch.Tensor, nonzero: torch.Tensor) -> torch.Tensor:
    safe_base = torch.where(nonzero, probability, torch.ones_like(probability))
    return torch.log(safe_base)


_MAX_SAFE_EXPONENT = 50.0


def _deviation_from_one(
    probability: torch.Tensor, nonzero: torch.Tensor, log_p: torch.Tensor, order: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``u = sum(probability ** order) - 1`` via ``expm1``, accurate when ``order`` is near 1.

    Also returns whether any exponent exceeded ``_MAX_SAFE_EXPONENT`` before clamping. The exponent is clamped before
    ``expm1`` because an unclamped one can overflow, and a discarded result still sends ``expm1``'s infinite local
    gradient into backward (``0 * inf = NaN``). The flag is taken from the unclamped exponent: once clamped, ``u`` no
    longer shows that the near-1 form is inaccurate.
    """
    exponent = (order - 1.0) * log_p
    exponent = torch.where(nonzero, exponent, torch.zeros_like(exponent))
    exponent_out_of_range = (exponent.abs() > _MAX_SAFE_EXPONENT).any(dim=-1)
    safe_exponent = torch.clamp(exponent, min=-_MAX_SAFE_EXPONENT, max=_MAX_SAFE_EXPONENT)
    terms = probability * torch.expm1(safe_exponent)
    u = torch.where(nonzero, terms, torch.zeros_like(terms)).sum(dim=-1)
    return u, exponent_out_of_range


def _log_power_sum(nonzero: torch.Tensor, log_p: torch.Tensor, order: float) -> torch.Tensor:
    """``log(sum(probability ** order))`` over the nonzero entries via ``logsumexp``, accurate for extreme ``order``."""
    log_terms = order * log_p
    log_terms = torch.where(nonzero, log_terms, torch.full_like(log_terms, float("-inf")))
    return torch.logsumexp(log_terms, dim=-1)


def _deviation_saturated(u: torch.Tensor, exponent_out_of_range: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Whether the ``u``-based near-1 form is unusable and the ``logsumexp`` form should be used: an exponent out of
    range, ``u`` not finite (order far below 1 on a peaked distribution), or ``u`` within ``eps ** 0.4`` of its floor
    of -1. The exponent 0.4 was found empirically: with 0.5, gradcheck still failed for some orders between 5 and 100.
    """
    tolerance = torch.finfo(dtype).eps ** 0.4
    return exponent_out_of_range | (u <= (-1.0 + tolerance)) | ~torch.isfinite(u)


def renyi_entropy(probability: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rényi generalized order-``alpha`` entropy.

    Reference: Rényi (1961). Matches ComplexityMeasures.jl's ``Renyi`` over the whole range of ``alpha``, including
    values near its ``alpha ≈ 1`` boundary.

    Gradient class: smooth, but only on a fixed support: an exactly zero probability gets gradient 0 by convention, not
    the one-sided limit (unbounded for ``alpha < 1``). ``alpha = 0`` has no gradient path (the output does not require
    grad); ``alpha = inf`` is a max, with the usual subgradient at ties.

    Args:
        probability: Discrete distribution of shape ``[..., num_outcomes]``, summing to 1 along the last dimension (not
            validated).
        alpha: Order, ``>= 0``. Special cases: 0 (Hartley entropy, ``log`` of the number of nonzero outcomes), 1
            (Shannon), 2 (collision entropy, ``-log(sum(probability ** 2))``), ``inf`` (min-entropy,
            ``-log(max(probability))``).

    Returns:
        Entropy in nats, shape ``[...]``. ``NaN`` for a row of ``probability`` containing ``NaN``.

    Raises:
        ValueError: If ``alpha < 0``.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}")
    has_nan = torch.isnan(probability).any(dim=-1)
    nonzero = probability > 0
    dtype = probability.dtype
    if alpha == 0:
        result = torch.log(nonzero.sum(dim=-1).to(dtype))
    elif _isclose_to_order(alpha, 1.0):
        result = _shannon_entropy(probability)
    elif math.isinf(alpha):
        masked = torch.where(nonzero, probability, torch.zeros_like(probability))
        result = -torch.log(masked.amax(dim=-1))
    else:
        log_p = _safe_log(probability, nonzero)
        u, exponent_out_of_range = _deviation_from_one(probability, nonzero, log_p, alpha)
        saturated = _deviation_saturated(u, exponent_out_of_range, dtype)
        # torch.where backprops through both branches; a safe dummy u keeps log1p's gradient finite where saturated.
        safe_u = torch.where(saturated, torch.zeros_like(u), u)
        near_one = torch.log1p(safe_u) / (1.0 - alpha)
        extreme = _log_power_sum(nonzero, log_p, alpha) / (1.0 - alpha)
        result = torch.where(saturated, extreme, near_one)
    return torch.where(has_nan, torch.full_like(result, float("nan")), result)


def tsallis_entropy(probability: torch.Tensor, q: float, *, k: float = 1.0) -> torch.Tensor:
    """Tsallis generalized order-``q`` entropy.

    Reference: Tsallis (1988). Matches ComplexityMeasures.jl's ``Tsallis`` outside its ``q ≈ 1`` window and inside it
    for ``k = 1``; inside it for ``k != 1`` this function returns the true limit, ``k`` times Shannon entropy, where
    Julia drops ``k``.

    Gradient class: smooth, but only on a fixed support: an exactly zero probability gets gradient 0 by convention, not
    the one-sided limit (unbounded for ``q < 1``).

    Args:
        probability: Discrete distribution of shape ``[..., num_outcomes]``, summing to 1 along the last dimension (not
            validated).
        q: Order; any real number (unlike Rényi's ``alpha``, as in ComplexityMeasures.jl). A very large ``|q|`` can
            overflow the dtype's range.
        k: Boltzmann-constant-like scale factor.

    Returns:
        Entropy of shape ``[...]``. ``NaN`` for a row of ``probability`` containing ``NaN``.
    """
    has_nan = torch.isnan(probability).any(dim=-1)
    nonzero = probability > 0
    dtype = probability.dtype
    if _isclose_to_order(q, 1.0):
        result = k * _shannon_entropy(probability)
    else:
        log_p = _safe_log(probability, nonzero)
        u, exponent_out_of_range = _deviation_from_one(probability, nonzero, log_p, q)
        saturated = _deviation_saturated(u, exponent_out_of_range, dtype)
        # The extreme branch's exp can overflow where it is discarded; torch.where backprops through both branches.
        log_sum = _log_power_sum(nonzero, log_p, q)
        safe_log_sum = torch.where(saturated, log_sum, torch.zeros_like(log_sum))
        # Fold k into the log-space shift; multiplying by k after exponentiating overflows for extreme k.
        sign = (1.0 if q > 1.0 else -1.0) * (1.0 if k >= 0.0 else -1.0)
        shifted = (
            torch.exp(safe_log_sum - math.log(abs(q - 1.0)) + math.log(abs(k))) * sign
            if k != 0.0
            else torch.zeros_like(safe_log_sum)
        )
        extreme = k / (q - 1.0) - shifted
        near_one = k * u / (1.0 - q)
        result = torch.where(saturated, extreme, near_one)
    return torch.where(has_nan, torch.full_like(result, float("nan")), result)
