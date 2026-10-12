"""Approximate entropy: hard-threshold and smooth kernel-weighted variants.

Reference: Pincus (1991), Proc. Natl. Acad. Sci. 88(6), 2297-2301, eq. 1-4: ApEn(m, r) = phi(m) - phi(m+1), where
phi(k) is the mean over all length-k embedded windows i of log(C_i(r)), and C_i(r) is the fraction of length-k windows
(including window i itself) within Chebyshev distance r of window i.

Unlike sample entropy, ApEn includes the self-match (C_i(r) >= 1/N), which biases it. This is the default here
(``theiler_window=-1``, Pincus's definition); ``theiler_window=0`` gives the self-excluding "modified ApEn" variant.
phi(m) and phi(m+1) use their own window sets with no truncation to common start indices, since the mean of logs has
no shared denominator to cancel.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _broadcast_over_matrix, _output_dtype
from torchnlts.embedding.delay import delay_embedding
from torchnlts.entropy.sample import _mask_nonfinite_input
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.neighbors.theiler import theiler_window_mask


def _log_conditional_probability(embedded, radius, theiler_window: int, norm_order: float) -> torch.Tensor:
    distance = pairwise_distance(embedded, norm_order=norm_order)
    radius = torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
    within_radius = distance <= _broadcast_over_matrix(radius, distance)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    conditional_probability = (within_radius & mask).sum(dim=-1).to(distance.dtype) / mask.sum(dim=-1)
    return torch.log(conditional_probability).mean(dim=-1)


def _mean_log(probability: torch.Tensor) -> torch.Tensor:
    """Mean over the last dim of ``log(probability)``, ``-inf`` if an entry is exactly 0. The log sees a safe operand
    there, so backward gets no ``0 * inf`` (an underflowed kernel sum would otherwise reach a shared bandwidth)."""
    zero = probability == 0
    safe = torch.where(zero, torch.ones_like(probability), probability)
    return torch.where(zero, torch.full_like(probability, float("-inf")), torch.log(safe)).mean(dim=-1)


def _log_soft_conditional_probability(
    embedded, temperature, theiler_window: int, kind: str, norm_order: float
) -> torch.Tensor:
    distance = pairwise_distance(embedded, norm_order=norm_order)
    temperature = torch.as_tensor(temperature, dtype=distance.dtype, device=distance.device)
    weight = trajectory_kernel(distance, _broadcast_over_matrix(temperature, distance), kind=kind)
    mask = theiler_window_mask(distance.shape[-1], theiler_window, device=distance.device)
    conditional_probability = torch.where(mask, weight, torch.zeros_like(weight)).sum(dim=-1) / mask.sum(dim=-1)
    return _mean_log(conditional_probability)


def approximate_entropy(
    trajectory,
    radius,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = -1,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Approximate entropy ApEn(m, r): phi(m) - phi(m+1) via embedded template matching.

    Reference: Pincus (1991), eq. 1-4. Matches ComplexityMeasures.jl's ``ApproximateEntropy`` for ``embedding_dim >= 2``
    at the default ``theiler_window=-1`` (that package has no Theiler window and is always self-inclusive). At
    ``embedding_dim=1`` it returns ``phi(1)`` alone, a different (non-Pincus) definition, so the two disagree there.

    Gradient class: none (Heaviside threshold).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        radius: Tolerance r. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        theiler_window: Theiler window over template start indices (see
            :func:`torchnlts.neighbors.theiler_window_mask`). ``-1`` (default) is Pincus's self-inclusive definition;
            ``0`` excludes only the self-match.
        norm_order: Minkowski norm order for template distances (``inf`` for the classic Chebyshev definition).

    Returns:
        ApEn, shape ``[]``/``[R]`` for an unbatched trajectory or ``[B]``/``[R, B]`` for a batched one. ``NaN`` if
        ``theiler_window`` excludes every candidate for some window (zero denominator) or the row contains ``NaN`` or
        ``inf``. If some window's conditional probability is exactly 0 (``log(0) = -inf``): ``+inf`` if only
        ``phi(m+1)`` is ``-inf``, ``-inf`` if only ``phi(m)`` is, and ``NaN`` if both are.

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an integer,
            ``norm_order`` is invalid, the trajectory is complex, or it is too short to embed.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    theiler_window = _as_int(theiler_window, "theiler_window")
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    short_embed = delay_embedding(trajectory, embedding_dim, delay=delay)
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay=delay)
    phi_short = _log_conditional_probability(short_embed, radius, theiler_window, norm_order)
    phi_long = _log_conditional_probability(long_embed, radius, theiler_window, norm_order)
    return torch.where(nonfinite, float("nan"), phi_short - phi_long).to(_output_dtype(trajectory))


def soft_approximate_entropy(
    trajectory,
    temperature,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = -1,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted approximate entropy, a smooth stand-in for the hard threshold.

    Gradient class: smooth (a relaxation of :func:`~torchnlts.entropy.approximate_entropy` built on
    :func:`~torchnlts.neighbors.trajectory_kernel`), except at ties in the default Chebyshev (``norm_order=inf``)
    distance, as for :func:`~torchnlts.entropy.soft_sample_entropy`.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        temperature: Kernel bandwidth, playing ``radius``'s role. Scalar or ``[R]`` tensor to sweep.
        embedding_dim: Template length m.
        delay: Embedding lag.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``trajectory_kernel``.
        theiler_window: Theiler window over template start indices, as for
            :func:`~torchnlts.entropy.approximate_entropy`.
        norm_order: Minkowski norm order for template distances.

    Returns:
        Weighted ApEn, with the shape and ``NaN`` conventions of :func:`~torchnlts.entropy.approximate_entropy`; a
        window whose kernel-weight sum underflows to 0 counts as a conditional probability of exactly 0 there, and
        puts no ``NaN`` into the gradients of other rows or of ``temperature``.

    Raises:
        ValueError: If ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is not an integer,
            ``kind`` is unknown, ``temperature`` is not a positive real number or tensor, ``norm_order`` is invalid,
            the trajectory is complex, or it is too short to embed.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    theiler_window = _as_int(theiler_window, "theiler_window")
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    short_embed = delay_embedding(trajectory, embedding_dim, delay=delay)
    long_embed = delay_embedding(trajectory, embedding_dim + 1, delay=delay)
    phi_short = _log_soft_conditional_probability(short_embed, temperature, theiler_window, kind, norm_order)
    phi_long = _log_soft_conditional_probability(long_embed, temperature, theiler_window, kind, norm_order)
    return torch.where(nonfinite, float("nan"), phi_short - phi_long).to(_output_dtype(trajectory))
