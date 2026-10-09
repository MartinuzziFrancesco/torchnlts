"""Rényi and Tsallis entropy: one-parameter families generalizing
Shannon entropy, applied to an *already-computed* probability
distribution -- not a new probability estimator of their own, unlike
this subpackage's other trajectory-taking functions (``sample_entropy``,
``permutation_entropy``, ``dispersion_entropy``, ...), each of which
both builds a distribution from a trajectory *and* fixes it to a
particular entropy (mostly, but not universally, Shannon's -- e.g.
``sample_entropy`` computes ``-log(A/B)``, not a discrete-distribution
Shannon sum, so "every other function computes Shannon entropy" would
overclaim; several of them do). These two are the reusable "back end":
pass in any discrete probability distribution -- including the ones
this subpackage's own functions build internally -- and get a
tunable-order entropy instead of a fixed one. See
``tutorials/generalized_entropy_lorenz.py`` for this plugged directly
into ``permutation_entropy``'s own ordinal-pattern distribution.

References:
  - Rényi, A. (1961), "On measures of entropy and information",
    Proceedings of the Fourth Berkeley Symposium on Mathematical
    Statistics and Probability, Vol. 1, 547-561.
  - Tsallis, C. (1988), "Possible generalization of Boltzmann-Gibbs
    statistics", Journal of Statistical Physics 52, 479-487.

Both reduce to Shannon entropy at their order parameter equal to 1 (by
a continuity/L'Hopital argument, not by definition -- their own
defining formulas have a removable 0/0 singularity there). Order
exactly 1 -- and, for Rényi, order exactly 0 -- switch to that closed
form. ComplexityMeasures.jl instead switches within a *tolerance window*
around each target, matching Julia's own default ``isapprox`` semantics
for the order value's type (double precision for a plain Python
``float``, confirmed live): around 1, a real window of about
``sqrt(eps(Float64)) ≈ 1.5e-8``; around 0, Julia's own default
``atol=0`` makes that comparison degenerate to effectively exact
equality only (confirmed live: ``isapprox(1e-10, 0.0)`` is ``false``
under Julia's own defaults), which is why this module's own ``alpha=0``
trigger is exact equality too, not a tolerance window -- an earlier
version of this module mistakenly used a *dtype-scaled* tolerance for
both (``sqrt(finfo(probability.dtype).eps)``), which is wider than
Julia's for float32 ``probability`` (confirmed to give a silently wrong
result: ``renyi_entropy(p, 0.0003)`` on float32 data returned the
alpha=0 Hartley value, ignoring the general formula's actual, different
answer at that order, and lost its gradient path along with it) --
fixed to use ``probability``-dtype-independent tolerances matching
Julia's own, since the order parameter here is always a plain Python
``float`` (i.e., always double precision), never tied to the
probability tensor's own dtype.

That tolerance-window fix alone isn't sufficient near order 1, though:
the general formula's remaining term, ``1/(1-order) * log(sum(p**order))``,
suffers real, confirmed cancellation for order *near* (not just at) 1,
independent of which special-case trigger convention is used --
confirmed directly with a skewed float32 distribution
(``[1-2**-20, 2**-20, 0]``) at order 0.9996/1.0004, both comfortably
outside any of the tolerance windows above: a naive direct-power
version of this module was off by ~93%. Fixed via a near-1-optimized
reformulation using ``expm1``/``log1p``: writing ``S = sum(p**order)``
and ``u = S - 1``, normalization (``sum(p) = 1``) gives
``u = sum(p * (p**(order-1) - 1)) = sum(p * expm1((order-1)*log(p)))``,
computed without ever forming the near-cancelling ``S`` directly, then
``log(S) = log1p(u)``, also cancellation-free for ``u`` near 0 (which is
exactly when order is near 1). Confirmed directly: reproduces the
skewed-distribution reference values to 6+ significant figures where
the naive version was off by 93%, and matches independently computed
reference values across a full order sweep (0.01 to 100) with no
accuracy gap anywhere, including right through the near-1 region.

This ``u``-based reformulation has its own failure mode, though, at the
*opposite* extreme: for order far from 1 with peaked probabilities,
``u`` saturates at its natural floor of exactly -1 (``S`` underflowing
to 0), and ``log1p(-1) = -inf`` -- confirmed directly: this
reformulation alone regressed the alpha=200/alpha=2000 extreme-order
cases an earlier, ``logsumexp``-only version of this module had already
fixed (see below), turning a correct ``log(2)`` back into ``inf``.
Both functions therefore compute *both* reformulations and select
between them via ``torch.where`` on whether ``u`` has saturated near
its -1 floor (fully vectorized, batch-row-independent, no data-dependent
Python ``if`` -- vmap-safe) -- the near-1-optimized one when ``u`` is
well away from -1, and a direct ``logsumexp``-based one (numerically
sound across a far wider order range than forming ``probability **
order`` directly ever is, confirmed directly: Rényi at alpha=2000 on
float64 or alpha=200 on float32 both gave ``inf`` under direct
computation, ``log(2)`` correctly under ``logsumexp``) otherwise.
Tsallis's extreme-order branch additionally shifts *both* the
``1/(order-1)`` scaling and ``k`` into log-space *before* exponentiating
(``exp(L - log(|order-1|) + log(|k|))`` rather than
``k * exp(L) / (order-1)``), extending its own overflow-free range
further still -- confirmed directly: recovers the correct, representable
``S_q`` for Tsallis at ``q=-128`` (float32), ``q=-1024`` (float64), and
(with an extreme ``k`` too) ``q=-1040, k=1e-300`` (float64), cases
earlier versions of this module gave ``inf`` for even though the
entropy value itself fits the dtype (an initial attempt shifted only
``1/(order-1)`` into log-space, leaving ``k`` multiplied in afterward,
still overflowing whenever ``k`` alone was extreme enough -- confirmed
directly before fixing that too). This still isn't unconditional,
though: if a *single* probability raised to a sufficiently extreme
order already exceeds the dtype's own representable range before any
summation or shifting happens (e.g. ``0.5 ** -4096`` in float64), or if
the combination of order and ``k`` pushes the *final*, fully-shifted
exponent right up against the dtype's own overflow boundary (confirmed
directly: ``q=-10, k=5e305`` in float64 lands close enough to that
boundary -- exponent ~709.5 against float64's own ~709.78 overflow
point -- that ordinary rounding in the intermediate ``logsumexp`` sum
can tip it over), no reformulation of the sum can rescue that -- an
inherent numerical-representability limit at that point, not a bug
this module tries to work around with extended-precision tricks.
Rényi's alpha=inf (min-entropy: -log of the largest probability) is a
third named special case, handled exactly (no "nearby" floating-point
neighborhood to worry about for an infinite target).

Every zero-probability entry is excluded from every sum above by
substituting a safe placeholder *base* (1, not 0) before taking a log
or exponentiating, and masking the *result* to zero afterward -- not by
substituting 0 as the base and exponentiating that, which is only safe
for a non-negative exponent (0 ** negative is infinite in IEEE 754,
would corrupt the Tsallis sum for order < 0, a case Rényi's own
alpha >= 0 requirement in this module doesn't need to worry about but
Tsallis's q has no such restriction here, matching ComplexityMeasures.jl).
A ``probability`` row containing ``NaN`` (e.g.
:func:`torchnlts.entropy.ordinal_pattern_probabilities`'s
own convention for a degenerate delay-embedded window, see its
docstring) is detected and propagated to a ``NaN`` output explicitly --
``NaN`` values would otherwise silently fail the ``> 0`` nonzero check
(``NaN > 0`` is ``False`` in IEEE 754) and be treated as zero-probability
outcomes, giving a finite-looking but meaningless result instead.

Gradient class: smooth for Tsallis and for Rényi at every ``alpha``
except exactly 0, **but only on a fixed support** -- differentiating
through a perturbation that keeps every already-zero entry at exactly
zero (a softmax's output can still be *extremely* close to zero for a
large enough logit gap, e.g. ``float32`` underflows some entries of
``softmax([0, -200])`` to exactly 0, so this caveat can arise through
that parametrization too, not only through an explicit zero literal).
The masked zero-probability entries have a well-defined gradient of 0
through the substituted-base construction above, but that 0 is a
*convention* for "this outcome's probability is fixed at zero", not
the general limiting derivative as a new outcome's probability grows
from zero -- confirmed directly (not assumed): for order < 1, that
one-sided limit is actually unbounded (e.g. at order 0.5, numerically
estimating (H(eps) - H(0)) / eps for the probability vector
``[0.5-eps/2, 0.5-eps/2, eps]`` as ``eps -> 0`` diverges rather than
converging to the masked gradient's value of 0). This module's own
gradchecks only exercise strictly positive probability vectors for
exactly this reason -- they would (correctly) fail against the true
limiting derivative on a vector with an exact zero at order < 1, since
``torch.autograd.gradcheck``'s finite-difference probe does perturb
zero entries into positive territory. Rényi's ``alpha = 0`` (Hartley
entropy) is different in kind, not just degree: it's a purely
combinatorial function of *which* probabilities are nonzero (``log``
of a count), with no continuous dependence on their values at all --
confirmed directly: its output tensor has ``requires_grad=False`` and
``grad_fn=None`` even when ``probability`` itself requires grad, so
calling ``.backward()`` on it in isolation raises, the same as it
would for any other value with no path back to a leaf requiring grad
(this is *not* the same situation as a genuinely zero-valued gradient
inside a connected graph, which backward would happily produce zeros
for). Harmless when this term is added to another differentiable loss
term (autograd only needs a live path through at least one summand),
but worth knowing before using ``alpha=0`` as a standalone loss. The
``alpha = inf`` special case (a ``max``) has the ordinary subgradient-
at-ties caveat any ``max``/``amax`` use elsewhere in this codebase has.

The near-1/extreme-order hybrid above (not just its forward value, its
*gradient* too) is verified by gradcheck across a wide sweep of orders
(0.0001 to 100, plus inf, and negative q for Tsallis) and randomly
generated probability distributions with no failures -- but that sweep
uses ordinarily-scaled probabilities, the kind any realistic outcome-
counting distribution (this subpackage's own ordinal-pattern/dispersion-
pattern distributions included) actually produces. It does **not**
cover probabilities near the dtype's own subnormal floor (e.g.
``1e-310`` in float64) combined with a simultaneously extreme order --
confirmed directly: at that joint extreme, the gradient (not the
forward value, which remains correct) can still be substantially wrong,
not just imprecise, likely from ``log(probability)``'s own local
gradient (``1/probability``) itself becoming enormous there,
independent of anything this module's branch-selection logic controls.
Not pursued further: no realistic discrete-outcome probability
distribution has mass this close to zero without simply being exactly
zero (which this module already handles correctly), so this is a
theoretical corner beyond where the effort to chase it further seems
justified, not a silently-accepted common case.
"""

from __future__ import annotations

import math

import torch


def _isclose_to_order(value: float, target: float) -> bool:
    """Matches ComplexityMeasures.jl's own ``order ≈ target`` special-case
    trigger for a plain (double-precision) order parameter: Julia's
    default ``isapprox`` relative tolerance for the order's own
    (double-precision) type, ``sqrt(eps(Float64)) ≈ 1.5e-8`` -- *not*
    scaled by ``probability``'s dtype (see module docstring for why an
    earlier, dtype-scaled version of this check was a real, confirmed
    bug). Never used for the ``target=0`` case, where Julia's own
    ``atol=0`` default makes the comparison degenerate to exact equality
    (see module docstring) -- callers check ``value == 0`` directly there.
    """
    tolerance = math.sqrt(2.220446049250313e-16)
    return math.isclose(value, target, rel_tol=tolerance, abs_tol=0.0)


def _safe_log(probability: torch.Tensor, nonzero: torch.Tensor) -> torch.Tensor:
    safe_base = torch.where(nonzero, probability, torch.ones_like(probability))
    return torch.log(safe_base)


_MAX_SAFE_EXPONENT = 50.0


def _deviation_from_one(
    probability: torch.Tensor, nonzero: torch.Tensor, log_p: torch.Tensor, order: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``u = sum(probability ** order) - 1``, via ``expm1`` -- accurate
    without cancellation when ``order`` is near 1 (see module docstring).

    Also returns whether any outcome's exponent was extreme enough to
    need clamping, checked on the *unclamped* exponent, separately from
    the clamped computation below -- two related but distinct problems,
    each needing its own fix, both found only by stress-testing this
    module's own gradcheck, not by advance reasoning:

    1) The exponent is clamped *before* ``expm1``, not just masked
       afterward: an unclamped exponent can itself overflow for an
       extreme ``order`` combined with a near-zero ``probability``
       entry, and ``expm1``'s own local gradient (``exp`` of that same
       exponent) is then also infinite there -- a masked/discarded
       *result* doesn't stop that infinite local gradient from
       poisoning the total gradient via ``0 * inf = NaN`` once backprop
       reaches this point, confirmed directly with a near-degenerate
       softmax-derived distribution that masking only the downstream
       ``log1p``/``exp`` steps (an earlier fix) didn't catch, since the
       poisoning source was further upstream than that fix reached.

    2) Clamping the exponent, on its own, breaks the *value* of ``u``
       for exactly the inputs it's meant to protect -- confirmed
       directly: without also reporting the clamp separately,
       :func:`_deviation_saturated`'s own ``u``-based check silently
       stopped detecting saturation once ``u`` itself was already
       clamped into looking well-behaved, so the (inaccurate, clamped)
       near-1 branch was wrongly selected instead of falling back,
       giving a plausible-looking but wrong forward value, not just a
       bad gradient. Saturation detection therefore needs the
       *unclamped* exponent, computed here before clamping, not the
       (by-then-already-safe) ``u`` alone.
    """
    exponent = (order - 1.0) * log_p
    exponent = torch.where(nonzero, exponent, torch.zeros_like(exponent))
    exponent_out_of_range = (exponent.abs() > _MAX_SAFE_EXPONENT).any(dim=-1)
    safe_exponent = torch.clamp(exponent, min=-_MAX_SAFE_EXPONENT, max=_MAX_SAFE_EXPONENT)
    terms = probability * torch.expm1(safe_exponent)
    u = torch.where(nonzero, terms, torch.zeros_like(terms)).sum(dim=-1)
    return u, exponent_out_of_range


def _log_power_sum(nonzero: torch.Tensor, log_p: torch.Tensor, order: float) -> torch.Tensor:
    """``log(sum(probability ** order))`` over the nonzero entries only,
    via ``torch.logsumexp`` rather than forming ``probability ** order``
    directly -- accurate across a far wider range of extreme ``order``
    than the ``u``-based reformulation above (see module docstring).
    """
    log_terms = order * log_p
    log_terms = torch.where(nonzero, log_terms, torch.full_like(log_terms, float("-inf")))
    return torch.logsumexp(log_terms, dim=-1)


def _deviation_saturated(u: torch.Tensor, exponent_out_of_range: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Whether ``u`` (see :func:`_deviation_from_one`) is no longer
    usable for the ``log1p(u)``/``u``-based near-1 formulas, and the
    ``logsumexp``-based reformulation should be used instead (see
    module docstring). Three distinct failure conditions, all checked:
    ``exponent_out_of_range`` (see :func:`_deviation_from_one` -- must
    be checked independently of ``u`` itself, since clamping there for
    safety also makes ``u`` alone stop reliably indicating this case),
    plus two more on ``u`` directly:
    ``u`` overflowing to ``inf`` (order far *below* 1 with a peaked
    distribution: the ``expm1`` exponent flips sign and grows large and
    positive -- confirmed directly: without this check, Tsallis at very
    negative ``q``, e.g. -128, silently returned ``inf`` from the
    near-1 branch itself, never reaching the intended fallback), or
    ``u`` merely *close* to its floor of -1, not only exactly at it.

    That second bound needs real margin, not just "not exactly -1":
    confirmed directly by comparing this formula's own output against
    the (verified-accurate-throughout) ``logsumexp``-based one across a
    sweep of orders -- relative error grows smoothly and substantially
    well before ``u`` reaches its literal floor. The exponent
    ``0.4`` (i.e. ``finfo(dtype).eps ** 0.4``, wider than the
    ``sqrt`` = ``** 0.5`` scale used for the order ≈ 1 special-case
    boundary) was chosen empirically, not by derivation: ``** 0.5``
    still left a real gap where this module's own gradcheck could
    fail (not from inaccuracy, but from a distinct hazard -- see below),
    swept down from there until 15 random float64 distributions x 9
    orders (5 to 100) all passed with margin.

    Getting the *value* comparison to agree is necessary but not
    sufficient for a clean gradient, though. Both branches here also
    needed a fix, confirmed only by gradcheck actually failing, not by
    reasoning about it in advance: ``torch.where`` backprops through
    *both* branches regardless of which is selected, and each branch's
    own local gradient can blow up exactly where the *other* branch is
    selected (``log1p(u)``'s local gradient ``1/(1+u)`` at ``u -> -1``
    for the near-1 branch; the extreme branch's own ``exp(...)`` for
    a large enough exponent) -- an unselected branch's ``0 *`` factor
    doesn't save you when the other factor is ``inf`` (``0*inf=NaN``).
    Same class of bug as this codebase's own ``rosenstein.py`` lesson:
    both branches of a ``torch.where`` need a finite local gradient
    everywhere the mask is defined, not just a correct forward value
    where selected. Both :func:`~torchnlts.entropy.renyi_entropy` and :func:`~torchnlts.entropy.tsallis_entropy`
    substitute a safe dummy value into whichever branch is about to be
    discarded, specifically to keep that branch's own local gradient
    finite there too.
    """
    tolerance = torch.finfo(dtype).eps ** 0.4
    return exponent_out_of_range | (u <= (-1.0 + tolerance)) | ~torch.isfinite(u)


def renyi_entropy(probability: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rényi generalized order-``alpha`` entropy.

    Gradient class: smooth, except at ``alpha=0`` (no gradient path at
    all, not a zero-valued one within a connected graph) and, for
    every ``alpha``, only on a fixed support (see module docstring for
    both caveats in full).

    Reference: Rényi (1961). Matches ComplexityMeasures.jl's ``Renyi``
    across its full parameter range (``alpha`` in ``{0, 1, 2, ..., inf}``
    and general non-integer values, including near its own ``alpha ≈ 1``
    special-case boundary, not only exactly at it -- see module
    docstring for the precise tolerance conventions, which differ
    between ``alpha ≈ 0`` and ``alpha ≈ 1``), including its own
    ``alpha >= 0`` requirement.

    Args:
        probability: A discrete probability distribution, shape
            ``[..., num_outcomes]``, summing to 1 along the last
            dimension (not validated -- the caller's responsibility,
            matching this codebase's general trust-the-caller
            convention for well-formed inputs).
        alpha: Order. Must be ``>= 0``. Named special cases:
            ``alpha=0`` (Hartley/max entropy, ``log`` of the number of
            nonzero-probability outcomes), ``alpha=1`` (Shannon
            entropy), ``alpha=2`` (collision entropy,
            ``-log(sum(probability**2))``), ``alpha=inf`` (min-entropy,
            ``-log(max(probability))``).

    Returns:
        Entropy in nats, shape ``[...]`` (``probability`` with its last
        dimension reduced away). ``NaN`` if the corresponding
        ``probability`` row contains any ``NaN`` (see module docstring).

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
        safe_probability = torch.where(nonzero, probability, torch.ones_like(probability))
        result = -(probability * torch.log(safe_probability)).sum(dim=-1)
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


def tsallis_entropy(probability: torch.Tensor, q: float, k: float = 1.0) -> torch.Tensor:
    """Tsallis generalized order-``q`` entropy.

    Gradient class: smooth, only on a fixed support (see module
    docstring).

    Reference: Tsallis (1988). Matches ComplexityMeasures.jl's
    ``Tsallis`` for ``q`` outside its own ``q ≈ 1`` tolerance window
    and for ``q ≈ 1`` with the default ``k=1``. Within that window with
    ``k != 1``, this function returns the true limit ``k`` times
    Shannon entropy (derivable via L'Hopital's rule on the general
    formula as ``q -> 1``, confirmed directly: evaluating the general
    formula at ``q`` values approaching 1 from outside Julia's own
    tolerance window converges to this) -- ComplexityMeasures.jl's own
    ``q ≈ 1`` special case silently drops ``k`` throughout that window
    (confirmed live: its result is identical for ``k=1`` and ``k=2`` at
    both ``q=1`` exactly and ``q=1.00000001`` -- inside its default
    tolerance -- while its own general-formula branch at
    ``q=1.0000001`` -- just outside it -- scales correctly with ``k``;
    a genuine discontinuity in that package across its whole ``q ≈ 1``
    window when ``k != 1``, not a single isolated point, and not a
    difference in convention this module chose to diverge from).

    Args:
        probability: A discrete probability distribution, shape
            ``[..., num_outcomes]``, summing to 1 along the last
            dimension (not validated, see :func:`~torchnlts.entropy.renyi_entropy`).
        q: Order. Unlike :func:`~torchnlts.entropy.renyi_entropy`'s ``alpha``, no sign
            restriction (matching ComplexityMeasures.jl) -- but see the
            module docstring's numerical-range caveat for very large
            ``|q|``.
        k: Boltzmann-constant-like scale factor.

    Returns:
        Entropy, shape ``[...]``. ``NaN`` if the corresponding
        ``probability`` row contains any ``NaN`` (see module docstring).
    """
    has_nan = torch.isnan(probability).any(dim=-1)
    nonzero = probability > 0
    dtype = probability.dtype
    if _isclose_to_order(q, 1.0):
        safe_probability = torch.where(nonzero, probability, torch.ones_like(probability))
        result = -k * (probability * torch.log(safe_probability)).sum(dim=-1)
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
