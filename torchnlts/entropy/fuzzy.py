"""Fuzzy entropy: template matching with a continuous (fuzzy-set) similarity instead of a hard threshold.

Reference: Chen, Wang, Xie & Yu (2007), IEEE Trans. Neural Syst. Rehabil. Eng. 15(2), 266-272, eq. 2-10:
FuzzyEn(m, n, r, N) = ln(phi^m(n, r)) - ln(phi^{m+1}(n, r)), where phi^m averages the fuzzy membership
D_ij = exp(-(d_ij)^n / r) over template pairs, d_ij is the Chebyshev distance between mean-removed m-length templates
(eq. 2-4), and both phi are restricted to the same N-m template start indices.

The membership is already continuous, so there is no separate soft variant. For ``[T, D]`` input a template is the
flattened ``m * D`` window and mean removal subtracts one scalar over all of it. phi is a flat average over all valid
pairs, which equals the paper's two-level average (eq. 7) at ``theiler_window=0`` (every template then has the same
number of valid neighbors) and generalizes it to larger windows as :func:`~torchnlts.entropy.sample_entropy` does.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _broadcast_over_matrix, _output_dtype, _work_dtype
from torchnlts.entropy.sample import _mask_nonfinite_input, _matched_length_templates, _negative_log_ratio
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
    """phi^1(n, r) for flattened-length-1 templates: exactly 1 for any trajectory and any positive n, r (a mean-removed
    scalar template is zero, so every distance is 0 and D_ij = 1), or ``NaN`` if the Theiler window leaves no pair.

    Used only for ``embedding_dim=1`` and ``D=1``. It bypasses ``_fuzzy_correlation``, whose gradient would be
    ``NaN`` here for ``0 < fuzzy_power < 1`` (infinite slope at distance 0 times a zero derivative); the result is a
    constant, disconnected from the graph. The pair count depends only on static shapes, so no ``.item()`` host sync
    breaks ``torch.compile``.
    """
    if embedded.shape[-2] - 1 > theiler_window:
        return torch.ones((), dtype=embedded.dtype, device=embedded.device)
    return torch.full((), float("nan"), dtype=embedded.dtype, device=embedded.device)


def fuzzy_entropy(
    trajectory,
    radius,
    *,
    fuzzy_power: float = 2.0,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Fuzzy entropy FuzzyEn(m, n, r): -log(phi^{m+1} / phi^m) via fuzzy-matched templates.

    Reference: Chen, Wang, Xie & Yu (2007), eq. 2-10. Each template has its own mean subtracted (eq. 2-3) and the
    membership ``exp(-d^fuzzy_power / radius)`` (eq. 5-6) replaces the hard threshold of
    :func:`~torchnlts.entropy.sample_entropy`. ``theiler_window`` generalizes the paper's self-match exclusion as in
    that function. Matches EntropyHub's ``FuzzEn`` (``Fx='default'``).

    Gradient class: smooth (the membership is continuous), with two exceptions:

    1. Mean removal turns every template of flattened length 2 (``embedding_dim * D`` or ``(embedding_dim + 1) * D``;
       for ``D = 1``, ``embedding_dim`` 1 or 2) into ``(v, -v)``, a systematic tie in the default Chebyshev max.
       ``torch.cdist``'s backward then sums over the tied components, so the gradient is wrong by a factor tied to their
       number (forward values are exact). ``norm_order=2.0`` or a flattened length other than 2 on both branches avoids
       it.
    2. For ``0 < fuzzy_power < 1``, ``distance ** fuzzy_power`` has an infinite slope at distance 0, which reaches
       autograd as ``NaN`` under Chebyshev and Manhattan distances (not Euclidean). The masked diagonal and the
       ``D = 1``, ``embedding_dim = 1`` short templates (all zero) are handled; any other unmasked pair of exactly
       coincident templates (e.g. a window and its neighbor in an arithmetic progression) still gives a ``NaN``
       gradient for non-Euclidean ``norm_order``.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        radius: Membership divisor r, positive. Scalar or ``[R]`` tensor to sweep.
        fuzzy_power: Membership exponent n; intended to be positive (not validated).
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices, ``>= 0``.
        norm_order: Minkowski norm order for template distances (``inf`` for the paper's Chebyshev definition).

    Returns:
        FuzzyEn, shape ``[]``/``[R]`` for an unbatched trajectory or ``[B]``/``[R, B]`` for a batched one. ``NaN`` for
        a row containing ``NaN`` or ``inf``, when the Theiler window leaves no template pair, or when the membership
        sum phi^m underflows to 0 (``radius`` small against the template distances); ``inf`` if only phi^{m+1}
        underflows. Such rows put no ``NaN`` into the gradients of other rows or of ``radius``.

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an integer
            ``>= 0``, the trajectory is complex, or the trajectory is too short to embed.
    """
    theiler_window = _require_self_exclusion(theiler_window, "fuzzy entropy")
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    dtype = _output_dtype(trajectory)
    short_embed, long_embed = _matched_length_templates(trajectory.to(_work_dtype(dtype)), embedding_dim, delay)
    short_embed = short_embed - short_embed.mean(dim=-1, keepdim=True)
    long_embed = long_embed - long_embed.mean(dim=-1, keepdim=True)
    if short_embed.shape[-1] == 1:
        phi_short = _degenerate_unit_template_correlation(short_embed, theiler_window)
    else:
        phi_short = _fuzzy_correlation(short_embed, radius, fuzzy_power, theiler_window, norm_order)
    phi_long = _fuzzy_correlation(long_embed, radius, fuzzy_power, theiler_window, norm_order)
    return torch.where(nonfinite, float("nan"), _negative_log_ratio(phi_long, phi_short)).to(dtype)
