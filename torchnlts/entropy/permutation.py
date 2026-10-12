"""Permutation entropy: Shannon entropy of ordinal (Bandt-Pompe) patterns.

Reference: Bandt & Pompe (2002), Phys. Rev. Lett. 88(17), 174102, Sec. II-III. Each delay-embedded vector (order
m = ``embedding_dim``, lag tau) is mapped to the permutation that sorts its coordinates; the entropy is that of the
pattern distribution, optionally divided by ``log(m!)``. Entropies are in nats (the paper uses bits). Univariate
input only.

Weighted variants (``weighting``) replace each vector's count by a weight: ``"variance"`` (Fadlallah, Chen, Keil &
Principe (2013), Phys. Rev. E 87, 022911; population variance of the vector's coordinates) or ``"amplitude"``
(Azami & Escudero (2016), Comput. Methods Programs Biomed. 128, 40-51; ``A / m * sum|x_i| + (1 - A) / (m - 1) *
sum|x_i - x_(i-1)|``). Probabilities are the summed weights over the total weight.

Limitation: a tied window goes to one pattern by earliest index (as in plain permutation entropy and
ComplexityMeasures.jl), whereas Azami & Escudero split its weight equally among the compatible permutations;
implement the split in ``_ordinal_pattern_codes`` if the paper's tie handling is needed.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import (
    _ensure_batched,
    _output_dtype,
    _pattern_counts,
    _power_of_two_scale,
    _real_scalar,
    _require_univariate,
    _shannon_entropy,
    _work_dtype,
)
from torchnlts.convert import as_tensor
from torchnlts.embedding.delay import delay_embedding
from torchnlts.entropy.sample import _mask_nonfinite_input


def _ordinal_pattern_codes(embedded: torch.Tensor) -> torch.Tensor:
    """Lehmer-code each row's ordinal pattern into ``[0, order!)``; ties go to the earliest index (stable argsort)."""
    order = embedded.shape[-1]
    ranks = torch.argsort(embedded, dim=-1, stable=True)
    later = torch.triu(torch.ones(order, order, dtype=torch.bool, device=embedded.device), diagonal=1)
    smaller_at_j = ranks.unsqueeze(-1) > ranks.unsqueeze(-2)  # [..., i, j] = ranks[i] > ranks[j]
    digits = (smaller_at_j & later).sum(dim=-1)  # digits[..., i] = #{j > i : ranks[j] < ranks[i]}
    weights = torch.tensor([math.factorial(order - 1 - i) for i in range(order)], device=embedded.device)
    return (digits * weights).sum(dim=-1)


def _validate_weighting(weighting, amplitude_weight, embedding_dim: int) -> float:
    """Validate the weighting arguments; returns ``amplitude_weight`` as a float."""
    if weighting not in (None, "variance", "amplitude"):
        raise ValueError(f"weighting must be None, 'variance' or 'amplitude', got {weighting!r}")
    if weighting != "amplitude":
        return 0.5
    amplitude_weight = _real_scalar(amplitude_weight, "amplitude_weight")
    if not 0 <= amplitude_weight <= 1:
        raise ValueError(f"amplitude_weight must be in [0, 1], got {amplitude_weight}")
    if embedding_dim < 2:
        raise ValueError(
            "amplitude weighting needs embedding_dim >= 2 (the difference term divides by embedding_dim - 1)"
        )
    return amplitude_weight


def _pattern_weights(embedded: torch.Tensor, weighting: str, amplitude_weight: float) -> torch.Tensor:
    """Per-vector weight ``[..., N]`` of a ``[..., N, order]`` embedding, up to a per-row positive factor (which
    cancels in the probabilities). Variance weights are translation invariant, so each window is first shifted by its
    own first coordinate (exact differences, no mean rounding); amplitude weights are not, so they are only rescaled.
    The scale is a detached power of two, so squaring cannot overflow or underflow.
    """
    if weighting == "variance":
        embedded = embedded - embedded[..., :1]
    embedded = embedded / _power_of_two_scale(embedded)
    if weighting == "variance":
        return ((embedded - embedded.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    order = embedded.shape[-1]
    level = embedded.abs().sum(dim=-1) / order
    change = (embedded[..., 1:] - embedded[..., :-1]).abs().sum(dim=-1) / (order - 1)
    return amplitude_weight * level + (1 - amplitude_weight) * change


def _probabilities(trajectory, embedding_dim: int, delay: int, weighting, amplitude_weight, name: str) -> torch.Tensor:
    """:func:`~torchnlts.entropy.ordinal_pattern_probabilities` in the working dtype (float32 for half input)."""
    tensor, _ = _ensure_batched(trajectory)
    _require_univariate(tensor, name)
    amplitude_weight = _validate_weighting(weighting, amplitude_weight, embedding_dim)
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    embedded = delay_embedding(trajectory, embedding_dim, delay=delay)
    num_patterns = math.factorial(embedding_dim)
    work = embedded.to(_work_dtype(embedded.dtype))
    invalid = nonfinite[..., None]
    if weighting is not None:
        codes = _ordinal_pattern_codes(work)
        sums = _pattern_counts(codes, num_patterns, _pattern_weights(work, weighting, amplitude_weight))
        total = sums.sum(dim=-1, keepdim=True)
        invalid = invalid | (total == 0)
        probability = sums / torch.where(invalid, torch.ones_like(total), total)
    else:
        codes = _ordinal_pattern_codes(embedded)
        counts = _pattern_counts(codes, num_patterns, dtype=work.dtype)
        probability = counts / counts.sum(dim=-1, keepdim=True)
    return torch.where(invalid, torch.full_like(probability, float("nan")), probability)


def ordinal_pattern_probabilities(
    trajectory,
    *,
    embedding_dim: int = 3,
    delay: int = 1,
    weighting: str | None = None,
    amplitude_weight: float = 0.5,
) -> torch.Tensor:
    """Relative frequency of each ordinal pattern in a delay-embedded trajectory, optionally weighted.

    This is the distribution :func:`~torchnlts.entropy.permutation_entropy` applies Shannon entropy to; pass it to
    :func:`~torchnlts.entropy.renyi_entropy` or :func:`~torchnlts.entropy.tsallis_entropy` for another entropy. A tied
    window goes to one pattern by earliest index (as in ComplexityMeasures.jl); for ``"amplitude"`` Azami & Escudero
    instead split its weight among the compatible patterns.

    Reference: Bandt & Pompe (2002) for ``None``; Fadlallah et al. (2013) for ``"variance"``; Azami & Escudero (2016)
    for ``"amplitude"``.

    Gradient class: none for ``weighting=None`` (ranking via ``argsort``); piecewise smooth otherwise (the pattern
    assignment is piecewise constant and the weights are smooth; ``"amplitude"`` has kinks where a coordinate or an
    adjacent difference is 0, and both weightings jump where two coordinates of a vector tie).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]`` (``D > 1`` is rejected). A tensor, numpy array, or
            pandas Series/DataFrame.
        embedding_dim: Pattern order m.
        delay: Embedding lag.
        weighting: ``None`` (plain counts), ``"variance"`` or ``"amplitude"``.
        amplitude_weight: Parameter A in ``[0, 1]`` of ``"amplitude"`` (``A = 1``: mean absolute level only,
            ``A = 0``: adjacent differences only); ignored otherwise.

    Returns:
        Probabilities over the ``embedding_dim!`` ordinal patterns, shape ``[embedding_dim!]`` or
        ``[B, embedding_dim!]``, summing to 1 along the last dimension. A row is entirely ``NaN`` if it contains
        ``NaN`` or ``inf`` or, with a weighting, if its total weight is 0 (e.g. a constant trajectory under
        ``"variance"``); such rows do not affect the gradient of other rows.

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1, ``embedding_dim`` or ``delay`` is not a positive
            integer, or ``trajectory`` is too short for them (see :func:`torchnlts.embedding.delay_embedding`), if
            ``weighting`` is unknown, or for ``"amplitude"`` if ``amplitude_weight`` is not a real number in ``[0, 1]``
            or ``embedding_dim < 2``.
    """
    probability = _probabilities(
        trajectory, embedding_dim, delay, weighting, amplitude_weight, "ordinal_pattern_probabilities"
    )
    return probability.to(_output_dtype(as_tensor(trajectory)))


def permutation_entropy(
    trajectory,
    *,
    embedding_dim: int = 3,
    delay: int = 1,
    normalize: bool = False,
    weighting: str | None = None,
    amplitude_weight: float = 0.5,
) -> torch.Tensor:
    """Permutation entropy: Shannon entropy of ordinal pattern frequencies.

    ``weighting="variance"`` gives weighted permutation entropy and ``weighting="amplitude"`` amplitude-aware
    permutation entropy.

    Reference: Bandt & Pompe (2002), Sec. II-III; Fadlallah et al. (2013); Azami & Escudero (2016). Matches
    ComplexityMeasures.jl's ``entropy(Shannon(base=ℯ), OrdinalPatterns(m, τ), x)`` and its weighted and
    amplitude-aware counterparts (that package's default base is 2).

    Gradient class: none for ``weighting=None``; piecewise smooth otherwise (see
    :func:`~torchnlts.entropy.ordinal_pattern_probabilities`).

    Args:
        trajectory: Forwarded to :func:`~torchnlts.entropy.ordinal_pattern_probabilities`.
        embedding_dim: Pattern order m.
        delay: Embedding lag.
        normalize: Divide by ``log(embedding_dim!)`` so the result lies in ``[0, 1]``; ``NaN`` (``0 / 0``) at
            ``embedding_dim = 1``, which has one pattern and no normalizing constant.
        weighting: ``None``, ``"variance"`` or ``"amplitude"``.
        amplitude_weight: Parameter A in ``[0, 1]`` of ``"amplitude"``.

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` entropy in nats (dimensionless if ``normalize``). ``NaN`` for the
        rows that are ``NaN`` in :func:`~torchnlts.entropy.ordinal_pattern_probabilities`.

    Raises:
        ValueError: As :func:`~torchnlts.entropy.ordinal_pattern_probabilities`.
    """
    probability = _probabilities(trajectory, embedding_dim, delay, weighting, amplitude_weight, "permutation_entropy")
    entropy = _shannon_entropy(probability)
    if normalize:
        entropy = entropy / math.log(math.factorial(embedding_dim))
    return entropy.to(_output_dtype(as_tensor(trajectory)))
