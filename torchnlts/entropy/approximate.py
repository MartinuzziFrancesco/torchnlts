"""Approximate entropy: hard threshold and smooth kernel-weighted variants.

Reference: Pincus, S.M. (1991), "Approximate entropy as a measure of
system complexity", Proc. Natl. Acad. Sci. 88(6), 2297-2301: ApEn(m, r)
= phi(m) - phi(m+1), where phi(k) is the mean over all length-k embedded
windows i of log(C_i(r)), and C_i(r) is the fraction of length-k windows
(including window i itself) within Chebyshev distance r of window i.

Unlike sample entropy (its self-excluding successor, Richman & Moorman
2000), ApEn always includes the self-match (C_i(r) >= 1/N unconditionally,
a known source of bias) -- generalized here into an optional Theiler
window (default -1, "no exclusion", i.e. Pincus's original definition)
rather than hard-coded, matching this codebase's `theiler_window`
convention elsewhere (`theiler_window=0` recovers the self-excluding
"modified ApEn" variant some literature also uses).

phi(m) and phi(m+1) are computed from their own independent windows (of
length m and m+1 respectively), with no shared/truncated index set --
unlike sample_entropy's B(r)/A(r), whose shared Theiler normalizer
requires matching lengths, ApEn's mean-of-logs structure has no such
shared denominator to cancel, so no truncation is needed or done.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _broadcast_over_matrix
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.neighbors.theiler import theiler_window_mask


def _mask_invalid_windows(conditional_probability: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
    # A NaN anywhere in window i's own coordinates makes every distance
    # from i (including to itself) NaN, so a naive `<= radius`/kernel
    # weight comparison silently evaluates to "no match" (False / 0)
    # rather than propagating NaN -- giving a plausible-looking but wrong
    # finite C_i(r) instead. Forced here via torch.where (a plain tensor
    # op, not a data-dependent Python `if`) so it stays torch.vmap-safe.
    invalid = torch.isnan(embedded).any(dim=-1)
    return torch.where(invalid, torch.full_like(conditional_probability, float("nan")), conditional_probability)


def _log_conditional_probability(embedded, radius, theiler_window: int, norm_order: float) -> torch.Tensor:
    distance = pairwise_distance(embedded, norm_order=norm_order)
    radius = torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
    within_radius = distance <= _broadcast_over_matrix(radius, distance)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    conditional_probability = (within_radius & mask).sum(dim=-1).to(distance.dtype) / mask.sum(dim=-1)
    conditional_probability = _mask_invalid_windows(conditional_probability, embedded)
    return torch.log(conditional_probability).mean(dim=-1)


def _log_soft_conditional_probability(embedded, temperature, theiler_window: int, kind: str, norm_order: float) -> torch.Tensor:
    distance = pairwise_distance(embedded, norm_order=norm_order)
    temperature = torch.as_tensor(temperature, dtype=distance.dtype, device=distance.device)
    weight = trajectory_kernel(distance, _broadcast_over_matrix(temperature, distance), kind=kind)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    conditional_probability = torch.where(mask, weight, torch.zeros_like(weight)).sum(dim=-1) / mask.sum(dim=-1)
    conditional_probability = _mask_invalid_windows(conditional_probability, embedded)
    return torch.log(conditional_probability).mean(dim=-1)


def approximate_entropy(
    trajectory,
    radius,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = -1,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Approximate entropy ApEn(m, r): phi(m) - phi(m+1) via embedded template matching.

    Gradient class: none (Heaviside threshold, no useful gradient).

    Reference: Pincus (1991), eq. 1-4. Matches ComplexityMeasures.jl's
    ``ApproximateEntropy`` at the default ``theiler_window=-1`` (its
    definition has no Theiler-window concept at all, i.e. always
    self-inclusive) for ``embedding_dim >= 2`` -- confirmed by directly
    reading its source: at ``m=1`` that package deliberately special-cases
    ``complexity`` to return ``phi(1)`` alone rather than ``phi(1) -
    phi(2)`` (a different, non-Pincus definition cited to a different
    paper in its own source comment), so this function's ``embedding_dim=1``
    output does not match it -- see the locked disagreement test in
    ``tests/test_approximate_entropy.py``.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame. A ``NaN``
            anywhere in a given window NaNs that window's contribution
            (see Returns) rather than raising -- a raise would need a
            data-dependent Python ``if`` on the tensor's contents, which
            rejects ``torch.vmap``'s ``BatchedTensor`` inputs outright.
        radius: Tolerance r. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
            ``-1`` (default) recovers Pincus's original self-inclusive
            definition; ``0`` excludes only the self-match.
        norm_order: Minkowski norm order for template distances (``inf``
            for the classic Chebyshev definition).

    Returns:
        ApEn, shape ``[]``/``[R]`` for an unbatched trajectory or
        ``[B]``/``[R, B]`` for a batched one. ``NaN`` if ``theiler_window``
        excludes every candidate for some window (a zero denominator), or
        if some window contains ``NaN`` (see Args). If some window's
        conditional probability is exactly ``0`` at the given
        ``theiler_window``/``radius`` (``log(0) = -inf``): ``+inf`` if
        only ``phi(m+1)`` is ``-inf``, ``-inf`` if only ``phi(m)`` is, and
        ``NaN`` (``-inf - -inf``) if both are.
    """
    short_embed = delay_embedding(trajectory, embedding_dim, delay)
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay)
    phi_short = _log_conditional_probability(short_embed, radius, theiler_window, norm_order)
    phi_long = _log_conditional_probability(long_embed, radius, theiler_window, norm_order)
    return phi_short - phi_long


def soft_approximate_entropy(
    trajectory,
    temperature,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = -1,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted approximate entropy, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`approximate_entropy`, built on
    :func:`torchnlts.neighbors.kernels.trajectory_kernel`) -- except at
    ties in the default Chebyshev (``norm_order=inf``) distance, the same
    caveat as :func:`torchnlts.entropy.sample.soft_sample_entropy`.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame. A ``NaN``
            anywhere in a given window NaNs that window's contribution
            rather than raising (see :func:`approximate_entropy`'s Args).
        temperature: Kernel bandwidth, playing ``radius``'s role. Scalar
            or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window, forwarded to the point mask (see
            :func:`approximate_entropy`).
        norm_order: Minkowski norm order for template distances.

    Returns:
        Weighted ApEn, same shape convention as :func:`approximate_entropy`.
    """
    short_embed = delay_embedding(trajectory, embedding_dim, delay)
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay)
    phi_short = _log_soft_conditional_probability(short_embed, temperature, theiler_window, kind, norm_order)
    phi_long = _log_soft_conditional_probability(long_embed, temperature, theiler_window, kind, norm_order)
    return phi_short - phi_long
