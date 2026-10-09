"""Fuzzy entropy: matched-length template matching with a continuous
(fuzzy-set) similarity function instead of a hard threshold.

Reference: Chen, Wang, Xie & Yu (2007), IEEE Trans. Neural Syst.
Rehabil. Eng. 15(2), 266-272, eq. 2-10: FuzzyEn(m, n, r, N) =
ln(phi^m(n, r)) - ln(phi^{m+1}(n, r)), where phi^m averages the fuzzy
membership D_ij = exp(-(d_ij)^n / r) over template pairs, d_ij is the
Chebyshev distance between *mean-removed* m-length template vectors
(eq. 2-4), and both phi^m and phi^{m+1} are restricted to the same
N-m valid template start indices.

Unlike sample/approximate entropy, there's no separate "soft" variant
here: the fuzzy membership function is already continuous by
construction (that continuity over the Heaviside step of SampEn/ApEn
is the paper's whole point), so a single function covers both roles.
For multivariate (``[T, D]``/``[B, T, D]``) input, a template is the
flattened ``m*D`` window and mean-removal subtracts one scalar mean
over all of it, generalizing the paper's own univariate ``[T]`` case
the same way :func:`torchnlts.entropy.sample_entropy` (via the
same :func:`_matched_length_templates`) generalizes to multivariate
trajectories.

Reuses :func:`torchnlts.entropy.sample._matched_length_templates` for
the matched short/long embeddings and mirrors
:func:`torchnlts.dimension.soft_correlation_sum`'s
Theiler-masked averaging pattern for phi^m/phi^{m+1} -- a single flat
average over all valid pairs, which exactly matches the paper's own
two-level average (eq. 7) at ``theiler_window=0`` (every template has
the same number of valid neighbors there), and generalizes it the
same way :func:`torchnlts.entropy.sample_entropy` generalizes
Richman & Moorman's fixed self-exclusion.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _broadcast_over_matrix
from torchnlts.entropy.sample import _matched_length_templates
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, theiler_window_mask


def _fuzzy_membership(distance: torch.Tensor, radius: torch.Tensor, fuzzy_power: float) -> torch.Tensor:
    """Fuzzy similarity D_ij = exp(-(d_ij)^n / r), eq. 5-6."""
    return torch.exp(-(distance**fuzzy_power) / radius)


def _fuzzy_correlation(
    embedded: torch.Tensor, radius, fuzzy_power: float, theiler_window: int, norm_order: float
) -> torch.Tensor:
    distance = pairwise_distance(embedded, norm_order=norm_order)
    radius = torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    # Substitute a placeholder on the zero diagonal before the power: its gradient is infinite for fuzzy_power < 1,
    # and the Chebyshev and Manhattan cdist backwards give NaN at zero distance.
    safe_distance = torch.where(mask, distance, torch.ones_like(distance))
    weight = _fuzzy_membership(safe_distance, _broadcast_over_matrix(radius, distance), fuzzy_power)
    return torch.where(mask, weight, torch.zeros_like(weight)).sum(dim=(-2, -1)) / mask.sum()


def _degenerate_unit_template_correlation(embedded: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """phi^1(n, r) == 1 exactly, for any positive n, r, and any
    trajectory: a *scalar* (flattened length 1) template's own mean
    equals itself, so every mean-removed scalar template -- not just
    some data-dependent coincidence, but every one, for every
    trajectory -- is the zero vector, pinning every pairwise distance
    in this branch at 0 regardless of the trajectory
    (D_ij = exp(-0^n/r) = 1). Only applies at ``embedding_dim=1`` for
    a univariate trajectory (a multivariate one's flattened
    ``embedding_dim * D`` template is ``D``, not 1, and mean-removal
    across its ``D`` features generally isn't zero -- see the caller).
    Bypasses :func:`_fuzzy_correlation`'s general distance/power/exp
    pipeline, which would poison the gradient for
    ``0 < fuzzy_power < 1`` on every call in this regime for every
    ``norm_order`` except Euclidean (an ``inf * 0`` chain-rule
    product: the membership function's locally infinite slope at
    distance 0, times an identically-zero distance-w.r.t.-trajectory
    term) even though the true gradient is exactly zero -- a constant,
    disconnected from the autograd graph, sidesteps that instead of
    trying to patch the chain rule after the fact.

    Whether any pair survives the Theiler window depends only on
    ``embedded``'s (static) template count and ``theiler_window``
    (already a plain int), not on any tensor's data -- computed
    directly rather than via :func:`~torchnlts.neighbors.theiler_window_mask` plus a
    data-dependent ``.item()``, which would force a host sync and
    break ``torch.compile`` fullgraph tracing (same category of
    ceiling as ``fractional_delay_embedding``'s own documented one).
    """
    if embedded.shape[-2] - 1 > theiler_window:
        return torch.ones((), dtype=embedded.dtype, device=embedded.device)
    return torch.full((), float("nan"), dtype=embedded.dtype, device=embedded.device)


def fuzzy_entropy(
    trajectory,
    radius,
    fuzzy_power: float = 2.0,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Fuzzy entropy FuzzyEn(m, n, r): -log(phi^{m+1}/phi^m) via fuzzy-matched templates.

    Gradient class: smooth (built on the continuous fuzzy membership
    function), except:

    1. Mean-removal (eq. 2-3) collapses every centered template whose
       *flattened* length (``embedding_dim * D`` or
       ``(embedding_dim + 1) * D`` for a ``D``-dimensional trajectory)
       is exactly 2 into the form ``(v, -v)``, so the default Chebyshev
       (``norm_order=inf``) max is a *systematic* tie between both
       components at every single distance evaluation on that
       template -- not the measure-zero coincidence max-norm ties are
       elsewhere in this codebase (e.g.
       :func:`torchnlts.entropy.soft_sample_entropy`, which has
       no mean-removal step). For a univariate trajectory this is
       exactly ``embedding_dim`` in ``{1, 2}`` (1 via its long-template
       branch, 2 via its short-template branch); for a multivariate
       one it depends on ``D`` too (e.g. ``embedding_dim=1, D=2`` hits
       it the same way). Confirmed (not just suspected) by comparing
       against central differences: ``torch.cdist``'s backward sums
       the gradient contribution over every tied argmax component
       rather than selecting one, so this systematic tie gives a
       gradient that's off by a factor tied to how many components
       tie -- not a merely non-canonical but still-valid subgradient.
       Forward values are exact regardless (this only affects
       backward). ``norm_order=2.0`` avoids this specific tie (the
       Euclidean max has no equivalent argmax ambiguity); a
       flattened length other than 2 on both branches avoids it too,
       but neither guarantees ``gradcheck`` agreement in general --
       see the next caveat, which is independent of this one.
    2. For ``0 < fuzzy_power < 1``, ``distance**fuzzy_power`` has an
       infinite local gradient wherever ``distance`` is exactly ``0``.
       Under the default Chebyshev distance and under Manhattan
       (``norm_order=1.0``) this actually reaches autograd as ``NaN``
       (``torch.cdist``'s backward propagates it for both); under
       Euclidean (``norm_order=2.0``) it doesn't -- confirmed directly,
       ``torch.cdist``'s own Euclidean backward already returns a
       finite gradient at a zero distance on its own, so this specific
       poisoning is Euclidean's exception, not the general rule. At
       the (always-masked) diagonal it's substituted away safely
       regardless of ``norm_order``, and at a flattened length exactly
       1 (a univariate trajectory's ``embedding_dim=1`` short-template
       branch, whose every mean-removed template is the zero vector
       *unconditionally*, not just for some inputs) it's bypassed
       entirely via :func:`_degenerate_unit_template_correlation`. It
       is *not* corrected for at any other unmasked pair of exactly
       coincident templates under a non-Euclidean ``norm_order``,
       which a sufficiently degenerate/periodic trajectory can still
       produce (e.g. any arithmetic-progression window against its
       neighbor, for any ``embedding_dim``) -- that
       remaining case is a genuine, data-dependent singularity of the
       fuzzy membership function itself, the same category as the
       measure-zero max-norm tie caveats elsewhere in this codebase,
       except triggered by exact (not just approximate) template
       coincidence, and ``norm_order`` doesn't avoid it. ``fuzzy_power``
       is intended to be positive, matching the paper's own
       parameterization (n, an exponent); this codebase doesn't
       validate that domain, matching ``sample_entropy``'s own
       convention of not validating ``radius``'s positivity either.

    Reference: Chen, Wang, Xie & Yu (2007), eq. 2-10. Each
    ``embedding_dim``-length template has its own mean subtracted
    before distances are computed (eq. 2-3), and the fuzzy membership
    ``exp(-d^fuzzy_power / radius)`` (eq. 5-6) replaces the hard
    threshold :func:`torchnlts.entropy.sample_entropy` uses.
    ``theiler_window`` generalizes the paper's fixed self-match-only
    exclusion the same way :func:`torchnlts.entropy.sample_entropy`
    does; it must be non-negative for the same reason.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        radius: Membership divisor r. Must be positive. Scalar or
            ``[R]`` tensor to sweep.
        fuzzy_power: Membership exponent n.
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices.
            Must be ``>= 0``.
        norm_order: Minkowski norm order for template distances
            (``inf`` for the paper's Chebyshev definition).

    Returns:
        FuzzyEn, shape ``[]``/``[R]`` for an unbatched trajectory or
        ``[B]``/``[R, B]`` for a batched one.

    Raises:
        ValueError: If ``theiler_window < 0``.
    """
    _require_self_exclusion(theiler_window, "fuzzy entropy")
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim, delay)
    short_embed = short_embed - short_embed.mean(dim=-1, keepdim=True)
    long_embed = long_embed - long_embed.mean(dim=-1, keepdim=True)
    if short_embed.shape[-1] == 1:
        phi_short = _degenerate_unit_template_correlation(short_embed, theiler_window)
    else:
        phi_short = _fuzzy_correlation(short_embed, radius, fuzzy_power, theiler_window, norm_order)
    phi_long = _fuzzy_correlation(long_embed, radius, fuzzy_power, theiler_window, norm_order)
    return -torch.log(phi_long / phi_short)
