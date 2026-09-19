"""Sample entropy: hard threshold and smooth kernel-weighted variants.

Reference: Richman & Moorman (2000), Am. J. Physiol. Heart Circ.
Physiol. 278(6), H2039-H2049: SampEn(m, r, N) = -ln(A(r)/B(r)), where
B(r) counts matching m-length embedded template pairs within tolerance
r and A(r) counts matching (m+1)-length template pairs, both excluding
self-matches (j != i) over the same set of template start indices.

Built entirely on shared primitives, not a new counting algorithm:
`delay_embedding` constructs the template vectors, and
`correlation_sum`/`soft_correlation_sum` (this codebase's
fraction-of-pairs-within-radius) supply B(r) and A(r) directly, since
each is already "count of index pairs within radius, excluding
temporally correlated ones, normalized by the pair count" -- the same
normalizer (the valid-pair count from `theiler_window_mask`, which is
``(N-m*delay)(N-m*delay-1)`` only in the classic ``theiler_window=0``
case) for both B(r) and A(r), since both come from the same number of
template start indices. That shared normalizer cancels in the A/B
ratio, so no new counting logic is needed here.
"""

from __future__ import annotations

import torch

from torchnlts.dimension.correlation import correlation_sum, soft_correlation_sum
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.theiler import _require_self_exclusion


def _matched_length_templates(trajectory, embedding_dim: int, delay: int):
    """Delay-embed at dims ``embedding_dim`` and ``embedding_dim + 1``,
    truncated to the same set of template start indices (dropping the
    last ``delay`` rows of the shorter embedding, which have no
    corresponding longer template).
    """
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay)
    short_embed = delay_embedding(trajectory, embedding_dim, delay)
    short_embed = short_embed[..., : long_embed.shape[-2], :]
    return short_embed, long_embed


def sample_entropy(
    trajectory,
    radius,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Sample entropy SampEn(m, r): -log(A(r)/B(r)) via embedded template matching.

    Gradient class: none (built on :func:`correlation_sum`'s Heaviside threshold).

    Reference: Richman & Moorman (2000), eq. 1-3. Loosely comparable to
    ComplexityMeasures.jl's ``SampleEntropy`` (also Chebyshev distance
    by default, i.e. ``norm_order=inf``) -- not an exact match in
    general, since that package's B(r) term has a boundary-handling
    deviation from the paper for its default ``theiler_window``-free
    case (see ``tests/test_sample_entropy.py``'s dual-validation test
    for the confirmed discrepancy and how the test neutralizes it; it
    also returns ``NaN`` rather than this function's ``inf`` when
    ``A(r) = 0``). ``theiler_window`` generalizes Richman & Moorman's
    fixed self-match-only exclusion (``theiler_window = 0`` recovers
    the classic definition, matching this codebase's ``|i - j| <=
    window`` convention); it must be non-negative, since SampEn's
    definition requires excluding self-matches unconditionally.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        radius: Tolerance r. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
            Must be ``>= 0``.
        norm_order: Minkowski norm order for template distances
            (``inf`` for the classic Chebyshev definition).

    Returns:
        SampEn, shape ``[]``/``[R]`` for an unbatched trajectory or
        ``[B]``/``[R, B]`` for a batched one. ``NaN`` if ``B(r)`` is
        zero (no ``embedding_dim``-length template pair matches within
        ``radius``) or itself ``NaN`` (only one usable template, or
        every pair excluded by ``theiler_window``, making
        :func:`correlation_sum`'s own denominator zero); ``inf`` if
        every such match fails to extend to length ``embedding_dim +
        1`` (``A(r) = 0``).

    Raises:
        ValueError: If ``theiler_window < 0``.
    """
    _require_self_exclusion(theiler_window, "sample entropy")
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim, delay)
    b = correlation_sum(short_embed, radius=radius, theiler_window=theiler_window, norm_order=norm_order)
    a = correlation_sum(long_embed, radius=radius, theiler_window=theiler_window, norm_order=norm_order)
    return -torch.log(a / b)


def soft_sample_entropy(
    trajectory,
    temperature,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted sample entropy, a smooth stand-in for the hard threshold.

    Gradient class: smooth (differentiable relaxation of
    :func:`sample_entropy`, built on :func:`soft_correlation_sum`) --
    except at ties in the default Chebyshev (``norm_order=inf``)
    distance, e.g. two template coordinates equally far apart, where
    the underlying max-norm itself is non-differentiable (a
    measure-zero case in random data, same caveat as any max-norm use
    of :func:`torchnlts.neighbors.distance.pairwise_distance`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        temperature: Kernel bandwidth, playing ``radius``'s role.
            Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window, forwarded to the point mask.
            Must be ``>= 0`` (see :func:`sample_entropy`).
        norm_order: Minkowski norm order for template distances.

    Returns:
        Weighted SampEn, same shape convention as :func:`sample_entropy`.
        ``NaN`` if ``temperature`` is small enough relative to the
        template distances that both the numerator and denominator
        kernel-weight sums underflow to exactly ``0`` (``0/0``, the
        smooth analogue of :func:`sample_entropy`'s ``B(r) = 0`` case;
        gradients are ``NaN`` too in this regime, in the usual
        PyTorch sense).

    Raises:
        ValueError: If ``theiler_window < 0``.
    """
    _require_self_exclusion(theiler_window, "sample entropy")
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim, delay)
    b = soft_correlation_sum(
        short_embed, temperature=temperature, kind=kind, theiler_window=theiler_window, norm_order=norm_order
    )
    a = soft_correlation_sum(
        long_embed, temperature=temperature, kind=kind, theiler_window=theiler_window, norm_order=norm_order
    )
    return -torch.log(a / b)
