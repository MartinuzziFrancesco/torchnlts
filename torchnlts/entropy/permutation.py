"""Permutation entropy: Shannon entropy of ordinal (Bandt-Pompe) patterns.

Reference: Bandt, C. & Pompe, B. (2002), "Permutation Entropy: A Natural
Complexity Measure for Time Series", Phys. Rev. Lett. 88(17), 174102,
Sec. II (ordinal patterns and entropy): delay-embed a scalar trajectory at order
(``embedding_dim``) m and delay tau, take each embedded vector's ordinal
pattern -- the permutation that sorts its m coordinates -- and compute
the Shannon entropy of the resulting pattern distribution, optionally
normalized by ``log(m!)`` to lie in ``[0, 1]`` (Sec. III). The paper
reports entropy in bits (log base 2); this module uses nats
(natural log) throughout, matching the rest of this codebase.

Univariate only: ordinal ranking compares scalar values, so a ``D > 1``
trajectory has no single well-defined pattern without an extra convention
the paper doesn't specify for combining dimensions -- rejected rather
than guessed at.

Weighted variants (``weighting``): each embedded vector's pattern count is
replaced by a weight, ``"variance"`` (Fadlallah, Chen, Keil & Principe
(2013), Phys. Rev. E 87, 022911: population variance of the
vector's coordinates around its own mean) or ``"amplitude"`` (Azami &
Escudero (2016), Comput. Methods Programs Biomed. 128, 40-51: ``A / m *
sum|x_i| + (1 - A) / (m - 1) * sum|x_i - x_(i-1)|``). Probabilities are
summed weights over total weight.

Limitation: a tied window goes to one pattern by earliest index (same as plain
permutation entropy and ComplexityMeasures.jl), whereas Azami & Escudero
(2016) split its weight equally among the compatible permutations; implement
the split in ``_ordinal_pattern_codes`` if the paper's tie handling is needed.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _output_dtype, _power_of_two_scale, _work_dtype
from torchnlts.convert import as_tensor
from torchnlts.embedding.delay import delay_embedding


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(
            f"permutation_entropy only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}"
        )


def _ordinal_pattern_codes(embedded: torch.Tensor) -> torch.Tensor:
    """Lehmer-code each row's ordinal pattern into ``[0, order!)``.

    Ties are broken by earliest index (stable ascending ``argsort``),
    matching Bandt & Pompe's tie-free "generic" assumption and
    ComplexityMeasures.jl's ``OrdinalPatterns`` convention for real-world
    repeated values.
    """
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
    try:
        if isinstance(amplitude_weight, str):
            raise TypeError
        amplitude_weight = float(amplitude_weight)
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(f"amplitude_weight must be a real scalar in [0, 1], got {amplitude_weight!r}") from None
    if not 0 <= amplitude_weight <= 1:
        raise ValueError(f"amplitude_weight must be in [0, 1], got {amplitude_weight}")
    if embedding_dim < 2:
        raise ValueError("amplitude weighting needs embedding_dim >= 2 (the difference term divides by embedding_dim - 1)")
    return amplitude_weight


def _pattern_weights(embedded: torch.Tensor, weighting: str, amplitude_weight: float) -> torch.Tensor:
    """Per-vector weight ``[..., N]`` of a ``[..., N, order]`` embedding,
    up to a per-row positive factor (which cancels in the probabilities).
    Variance weights are translation invariant, so each window is first
    shifted by its own first coordinate (exact differences, no mean
    rounding); amplitude weights are not, so they are only rescaled. The
    scale is a detached power of two, so squaring cannot overflow/underflow.
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


def _probabilities(trajectory, embedding_dim: int, delay: int, weighting, amplitude_weight) -> torch.Tensor:
    """:func:`ordinal_pattern_probabilities` in the working dtype (float32 for half input)."""
    tensor, _ = _ensure_batched(trajectory)
    _require_univariate(tensor)
    amplitude_weight = _validate_weighting(weighting, amplitude_weight, embedding_dim)
    embedded = delay_embedding(trajectory, embedding_dim, delay)
    num_patterns = math.factorial(embedding_dim)
    work = embedded.to(_work_dtype(embedded.dtype))
    if weighting is not None:
        finite = torch.isfinite(work).all(dim=(-2, -1), keepdim=True)
        work = torch.where(finite, work, torch.zeros_like(work))  # keeps NaN out of backward
        codes = _ordinal_pattern_codes(work)
        weights = _pattern_weights(work, weighting, amplitude_weight)
        sums = (torch.nn.functional.one_hot(codes, num_classes=num_patterns).to(weights.dtype) * weights.unsqueeze(-1)).sum(dim=-2)
        total = sums.sum(dim=-1, keepdim=True)
        invalid = (total == 0) | ~finite.squeeze(-1)
        probability = sums / torch.where(invalid, torch.ones_like(total), total)
        return torch.where(invalid, torch.full_like(probability, float("nan")), probability)
    has_nan = torch.isnan(embedded).any(dim=(-2, -1))
    codes = _ordinal_pattern_codes(embedded)  # exact ranks for integer input
    counts = torch.nn.functional.one_hot(codes, num_classes=num_patterns).sum(dim=-2).to(work.dtype)
    probability = counts / counts.sum(dim=-1, keepdim=True)
    return torch.where(has_nan.unsqueeze(-1), torch.full_like(probability, float("nan")), probability)


def ordinal_pattern_probabilities(
    trajectory,
    embedding_dim: int = 3,
    delay: int = 1,
    weighting: str | None = None,
    amplitude_weight: float = 0.5,
) -> torch.Tensor:
    """Relative frequency of each ordinal pattern in a delay-embedded
    trajectory (optionally weighted) -- the probability
    distribution :func:`permutation_entropy` applies Shannon entropy to,
    exposed so a different entropy (e.g.
    :func:`torchnlts.entropy.generalized.renyi_entropy`) can be applied to it
    (see ``tutorials/generalized_entropy_lorenz.py``).

    Gradient class: none for ``weighting=None`` (ranking via ``argsort``).
    For ``"variance"``/``"amplitude"``: piecewise smooth in the trajectory
    (the pattern assignment is piecewise constant, the weights are smooth;
    ``"amplitude"`` has kinks where a coordinate or an adjacent difference
    is 0, and both jump where two coordinates of a vector tie).

    Reference: Bandt & Pompe (2002) for ``None``; Fadlallah et al. (2013)
    for ``"variance"``; Azami & Escudero (2016) for ``"amplitude"``.

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (a
            multivariate ``D > 1`` trajectory is rejected -- see module
            docstring). A tensor, numpy array, or pandas Series/DataFrame.
        embedding_dim: Pattern order m (number of points per pattern).
        delay: Embedding lag.
        weighting: ``None`` (plain counts), ``"variance"`` or ``"amplitude"``.
        amplitude_weight: The parameter A in ``[0, 1]`` of ``"amplitude"``
            weighting (``A = 1``: mean absolute level only, ``A = 0``:
            adjacent differences only); ignored otherwise.

    Returns:
        Probability distribution over the ``embedding_dim!`` possible
        ordinal patterns, shape ``[embedding_dim!]`` (unbatched) or
        ``[B, embedding_dim!]`` (batched), summing to 1 along the last
        dimension. A row is entirely ``NaN`` if the corresponding
        delay-embedded window contains ``NaN`` (an ordinal pattern is
        undefined when a coordinate can't be compared to the others) --
        masked via a plain tensor op rather than raised, so a NaN
        trajectory degrades gracefully under ``torch.vmap`` too. With a
        weighting, a row is also ``NaN`` if it contains ``inf`` or its total
        weight is 0 (e.g. a constant trajectory under ``"variance"``); such
        rows do not affect the gradient of other batch rows.

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1, or
            is too short for ``embedding_dim``/``delay`` (see
            :func:`torchnlts.embedding.delay.delay_embedding`), if
            ``weighting`` is unknown, or for ``"amplitude"`` if
            ``amplitude_weight`` is outside ``[0, 1]`` or
            ``embedding_dim < 2``.
    """
    probability = _probabilities(trajectory, embedding_dim, delay, weighting, amplitude_weight)
    return probability.to(_output_dtype(delay_embedding(trajectory, embedding_dim, delay)))


def permutation_entropy(
    trajectory,
    embedding_dim: int = 3,
    delay: int = 1,
    normalize: bool = False,
    weighting: str | None = None,
    amplitude_weight: float = 0.5,
) -> torch.Tensor:
    """Permutation entropy: Shannon entropy of ordinal pattern frequencies.
    ``weighting="variance"`` gives weighted permutation entropy and
    ``weighting="amplitude"`` amplitude-aware permutation entropy.

    Gradient class: none for ``weighting=None``; piecewise smooth otherwise
    (see :func:`ordinal_pattern_probabilities`).

    Reference: Bandt & Pompe (2002), Sec. II-III; Fadlallah et al. (2013);
    Azami & Escudero (2016). Matches
    ComplexityMeasures.jl's ``entropy(Shannon(base=MathConstants.e),
    OrdinalPatterns(m=embedding_dim, τ=delay), x)`` (natural log, forced
    explicitly since that package's own default base is 2); likewise
    ``WeightedOrdinalPatterns``/``AmplitudeAwareOrdinalPatterns``.

    Args:
        trajectory: Forwarded to :func:`ordinal_pattern_probabilities`.
        embedding_dim: Pattern order m (number of points per pattern).
        delay: Embedding lag.
        normalize: Divide by ``log(embedding_dim!)`` so the result is a
            dimensionless quantity in ``[0, 1]``. ``NaN`` (``0 / 0``) at
            ``embedding_dim = 1``, where there is exactly one possible
            pattern and no normalizing constant.
        weighting: ``None``, ``"variance"`` or ``"amplitude"``.
        amplitude_weight: Parameter A in ``[0, 1]`` of ``"amplitude"``.

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` (batched) entropy, in
        nats (or dimensionless if ``normalize=True``). ``NaN`` under
        the same condition as :func:`ordinal_pattern_probabilities`'s
        own ``NaN`` rows (Shannon entropy of an all-``NaN`` distribution
        is itself ``NaN``, propagated rather than special-cased again
        here).

    Raises:
        ValueError: Same as :func:`ordinal_pattern_probabilities`.
    """
    probability = _probabilities(trajectory, embedding_dim, delay, weighting, amplitude_weight)
    log_probability = torch.log(probability.clamp(min=torch.finfo(probability.dtype).tiny))
    entropy = -(probability * log_probability).sum(dim=-1)
    if normalize:
        entropy = entropy / math.log(math.factorial(embedding_dim))
    return entropy.to(_output_dtype(as_tensor(trajectory)))
