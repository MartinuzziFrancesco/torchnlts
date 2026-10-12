"""Sample entropy: hard-threshold and smooth kernel-weighted variants.

Reference: Richman & Moorman (2000), Am. J. Physiol. Heart Circ. Physiol. 278(6), H2039-H2049, eq. 1-3:
SampEn(m, r, N) = -ln(A(r) / B(r)), where B(r) counts matching m-length template pairs within tolerance r and A(r)
counts matching (m+1)-length pairs, both excluding self-matches (j != i) over the same template start indices.

B(r) and A(r) are the pair fractions that :func:`~torchnlts.dimension.correlation_sum` returns for the m- and
(m+1)-length embeddings. Both share the valid-pair normalizer (from the Theiler window mask), which cancels in the
ratio.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _restore_shape_preserving,
    _work_dtype,
)
from torchnlts.dimension.correlation import correlation_sum, soft_correlation_sum
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.theiler import _require_self_exclusion


def _mask_nonfinite_input(trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """``trajectory`` with every row containing NaN/inf set to zero, and the mask of those rows (``[B]``, or 0-d for
    an unbatched input).

    Applied to the raw series, so a non-finite sample that enters no embedded window or coarse-grained window still
    marks its row. Callers set the result of masked rows to ``NaN``; the zeros keep NaN out of backward.
    """
    batched, ndim = _ensure_batched(trajectory)
    masked, nonfinite = _mask_nonfinite_rows(batched)
    return _restore_shape_preserving(masked, ndim), (nonfinite if ndim == 3 else nonfinite.squeeze(0))


def _matched_length_templates(trajectory, embedding_dim: int, delay: int):
    """Delay embeddings at dims ``embedding_dim`` and ``embedding_dim + 1`` over the same template start indices (the
    last ``delay`` rows of the shorter one have no longer partner and are dropped).
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay=delay)
    short_embed = delay_embedding(trajectory, embedding_dim, delay=delay)
    short_embed = short_embed[..., : long_embed.shape[-2], :]
    return short_embed, long_embed


def _negative_log_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """``-log(numerator / denominator)``: ``inf`` where the numerator is 0, ``NaN`` where the denominator is 0 or NaN.
    Safe operands go into the logs, so the discarded cases put no inf/NaN into backward (``0 * inf`` reaches a shared
    bandwidth otherwise). A difference of logs, since the quotient's gradient ``numerator / denominator**2`` overflows
    for denominators below about ``1e-19`` in float32."""
    no_match = (denominator == 0) | torch.isnan(denominator)
    no_extension = (numerator == 0) & ~no_match
    safe_numerator = torch.where(no_match | no_extension, torch.ones_like(numerator), numerator)
    safe_denominator = torch.where(no_match, torch.ones_like(denominator), denominator)
    entropy = torch.log(safe_denominator) - torch.log(safe_numerator)
    entropy = torch.where(no_extension, torch.full_like(entropy, float("inf")), entropy)
    return torch.where(no_match, torch.full_like(entropy, float("nan")), entropy)


def sample_entropy(
    trajectory,
    radius,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Sample entropy SampEn(m, r): -log(A(r) / B(r)) via embedded template matching.

    Reference: Richman & Moorman (2000), eq. 1-3. ``theiler_window`` generalizes their fixed self-match exclusion
    (``theiler_window=0`` is the classic definition, with this library's ``|i - j| <= window`` convention). Compared
    with ComplexityMeasures.jl's ``SampleEntropy`` (also Chebyshev by default), whose B(r) additionally counts matches
    with the last m-length template that has no (m+1)-length partner (``delay=1``) and which returns ``NaN`` rather than
    ``inf`` when A(r) = 0, the values agree only when no such extra match exists.

    Gradient class: none (built on the Heaviside threshold of :func:`~torchnlts.dimension.correlation_sum`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        radius: Tolerance r. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices (see
            :func:`torchnlts.neighbors.theiler_window_mask`); must be ``>= 0`` since self-matches are always excluded.
        norm_order: Minkowski norm order for template distances (``inf`` for the classic Chebyshev definition).

    Returns:
        SampEn, shape ``[]``/``[R]`` for an unbatched trajectory or ``[B]``/``[R, B]`` for a batched one. ``NaN`` if
        B(r) is zero (no m-length template pair matches within ``radius``) or itself ``NaN`` (a single usable
        template, or every pair excluded by ``theiler_window``), and for a row containing ``NaN`` or ``inf``; ``inf``
        if no match extends to length ``embedding_dim + 1`` (A(r) = 0).

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an integer
            ``>= 0``, or the trajectory is too short to embed.
    """
    theiler_window = _require_self_exclusion(theiler_window, "sample entropy")
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim, delay)
    short_matches = correlation_sum(short_embed, radius=radius, theiler_window=theiler_window, norm_order=norm_order)
    long_matches = correlation_sum(long_embed, radius=radius, theiler_window=theiler_window, norm_order=norm_order)
    return torch.where(nonfinite, float("nan"), -torch.log(long_matches / short_matches)).to(out_dtype)


def soft_sample_entropy(
    trajectory,
    temperature,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted sample entropy, a smooth stand-in for the hard threshold.

    Gradient class: smooth (a relaxation of :func:`~torchnlts.entropy.sample_entropy` built on
    :func:`~torchnlts.dimension.soft_correlation_sum`), except at ties in the default Chebyshev (``norm_order=inf``)
    distance, where the max-norm itself is not differentiable (a measure-zero case in random data).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        temperature: Kernel bandwidth, playing ``radius``'s role. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window over template start indices; must be ``>= 0``, as for
            :func:`~torchnlts.entropy.sample_entropy`.
        norm_order: Minkowski norm order for template distances.

    Returns:
        Weighted SampEn, with the shape convention of :func:`~torchnlts.entropy.sample_entropy`. ``NaN`` if
        ``temperature`` is small enough relative to the template distances that both kernel-weight sums underflow to 0
        (``0 / 0``), and for a row containing ``NaN`` or ``inf``; ``inf`` if only the numerator underflows. A ``NaN``
        entry puts no ``NaN`` into the gradient of other batch rows that share an input.

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an integer
            ``>= 0``, or the trajectory is too short to embed.
    """
    theiler_window = _require_self_exclusion(theiler_window, "sample entropy")
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    short_embed, long_embed = _matched_length_templates(trajectory, embedding_dim, delay)
    short_matches = soft_correlation_sum(
        short_embed, temperature=temperature, kind=kind, theiler_window=theiler_window, norm_order=norm_order
    )
    long_matches = soft_correlation_sum(
        long_embed, temperature=temperature, kind=kind, theiler_window=theiler_window, norm_order=norm_order
    )
    entropy = _negative_log_ratio(long_matches, short_matches)
    return torch.where(nonfinite, torch.full_like(entropy, float("nan")), entropy).to(out_dtype)
