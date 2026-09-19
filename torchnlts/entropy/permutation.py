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
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched
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


def permutation_entropy(
    trajectory,
    embedding_dim: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> torch.Tensor:
    """Permutation entropy: Shannon entropy of ordinal pattern frequencies.

    Gradient class: none (ordinal ranking via ``argsort`` has no useful gradient).

    Reference: Bandt & Pompe (2002), Sec. II-III. Matches
    ComplexityMeasures.jl's ``entropy(Shannon(base=MathConstants.e),
    OrdinalPatterns(m=embedding_dim, τ=delay), x)`` (natural log, forced
    explicitly since that package's own default base is 2).

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (a
            multivariate ``D > 1`` trajectory is rejected -- see module
            docstring). A tensor, numpy array, or pandas Series/DataFrame.
        embedding_dim: Pattern order m (number of points per pattern).
        delay: Embedding lag.
        normalize: Divide by ``log(embedding_dim!)`` so the result is a
            dimensionless quantity in ``[0, 1]``. ``NaN`` (``0 / 0``) at
            ``embedding_dim = 1``, where there is exactly one possible
            pattern and no normalizing constant.

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` (batched) entropy, in
        nats (or dimensionless if ``normalize=True``).

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1, is
            too short for ``embedding_dim``/``delay`` (see
            :func:`torchnlts.embedding.delay.delay_embedding`), or
            contains ``NaN`` in a delay-embedded window (an ordinal pattern is undefined when a
            coordinate can't be compared to the others).
    """
    tensor, _ = _ensure_batched(trajectory)
    _require_univariate(tensor)
    embedded = delay_embedding(trajectory, embedding_dim, delay)
    if torch.isnan(embedded).any():
        raise ValueError("permutation_entropy: trajectory contains NaN, ordinal patterns are undefined for NaN")
    codes = _ordinal_pattern_codes(embedded)
    num_patterns = math.factorial(embedding_dim)
    dtype = embedded.dtype if embedded.is_floating_point() else torch.get_default_dtype()
    counts = torch.nn.functional.one_hot(codes, num_classes=num_patterns).sum(dim=-2).to(dtype)
    probability = counts / counts.sum(dim=-1, keepdim=True)
    log_probability = torch.log(probability.clamp(min=torch.finfo(probability.dtype).tiny))
    entropy = -(probability * log_probability).sum(dim=-1)
    if normalize:
        entropy = entropy / math.log(num_patterns)
    return entropy
