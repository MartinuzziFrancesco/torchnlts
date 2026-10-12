"""Dispersion entropy: Shannon entropy of Gaussian-CDF-classified dispersion patterns.

Reference: Rostaghi & Azami (2016), IEEE Signal Process. Lett. 23(5), 610-614, in the form restated by Azami,
Rostaghi, Abasolo & Escudero (2017), IEEE Trans. Biomed. Eng., arXiv:1606.01379, eqs. 2-4:

1. Map each value to ``y`` in (0, 1) with the Normal CDF, using the trajectory's own mean and standard deviation
   (eq. 2).
2. Assign the class ``z = round(c * y + 0.5)`` in ``{1, ..., c}``, with ``c = num_classes``.
3. Delay-embed the class sequence (dimension m, lag d); each window is a dispersion pattern (``c ** m`` possible
   patterns).
4. Shannon entropy of the pattern frequencies (eq. 4), optionally divided by ``log(c ** m)``.

Unlike permutation entropy it depends on the amplitudes, not only on the order, of nearby values. Univariate input
only. Gradient class: none (rounding into classes, then a histogram). Works under ``torch.vmap``.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _output_dtype,
    _pattern_counts,
    _require_univariate,
    _restore_shape_preserving,
    _shannon_entropy,
    _work_dtype,
)
from torchnlts.embedding.delay import delay_embedding


def _gaussian_cdf_classes(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Eq. 2 and the linear class assignment: each value's class in ``{1, ..., num_classes}`` from the Normal CDF with
    the trajectory's own mean and standard deviation.
    """
    mean = tensor.mean(dim=-2, keepdim=True)
    std = tensor.std(dim=-2, keepdim=True)
    cdf = 0.5 * (1 + torch.erf((tensor - mean) / (std * math.sqrt(2))))
    classes = torch.round(num_classes * cdf + 0.5)
    return classes.clamp(min=1, max=num_classes)


def dispersion_entropy(
    trajectory,
    *,
    num_classes: int = 6,
    embedding_dim: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> torch.Tensor:
    """Dispersion entropy: Shannon entropy of Gaussian-CDF-classified dispersion-pattern frequencies.

    Reference: Rostaghi & Azami (2016), eqs. as restated in Azami et al. (2017), eqs. 2-4.

    Gradient class: none.

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]`` (``D > 1`` is rejected). A tensor, numpy array, or
            pandas Series/DataFrame.
        num_classes: Number of classes c, at least 2. The paper recommends ``2 < c < 9``; the default 6 is the
            follow-up paper's choice.
        embedding_dim: Pattern length m. The pattern histogram has ``num_classes ** embedding_dim`` bins, so memory
            grows with that product rather than with the trajectory length.
        delay: Embedding lag d; the paper recommends 1.
        normalize: Divide by ``log(num_classes ** embedding_dim)`` so the result lies in ``[0, 1]``.

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` entropy in nats (dimensionless if ``normalize``). ``NaN`` for a row
        that contains ``NaN`` or ``inf`` or whose standard deviation is exactly zero, e.g. an exactly constant
        trajectory (ComplexityMeasures.jl's ``Dispersion`` returns ``-0.0`` there). A trajectory that is constant only
        up to rounding (e.g. in float32) can have a tiny nonzero standard deviation and then gives a low finite
        entropy.

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1, ``num_classes`` is not an integer ``>= 2``,
            ``embedding_dim`` or ``delay`` is not a positive integer, or ``trajectory`` is too short for them (see
            :func:`torchnlts.embedding.delay_embedding`).
    """
    num_classes = _as_int(num_classes, "num_classes")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor, "dispersion_entropy")
    out_dtype = _output_dtype(tensor)
    tensor = tensor.to(_work_dtype(tensor.dtype))
    classes = _gaussian_cdf_classes(tensor, num_classes)
    # Restore the caller's [T]/[T, 1]/[B, T, 1] shape so delay_embedding does not treat an unbatched input as batched.
    classes = _restore_shape_preserving(classes, ndim)
    embedded = delay_embedding(classes, embedding_dim, delay=delay)

    # Zero variance gives NaN classes; substitute a dummy class before the int64 cast, which is undefined for NaN.
    # Codes must be integers: float codes collide once num_classes**embedding_dim exceeds 2**24.
    has_nan = torch.isnan(embedded).any(dim=(-2, -1))
    embedded = torch.where(torch.isnan(embedded), torch.ones_like(embedded), embedded).to(torch.int64)
    weights = num_classes ** torch.arange(embedding_dim - 1, -1, -1, dtype=torch.int64, device=embedded.device)
    codes = ((embedded - 1) * weights).sum(dim=-1)

    num_patterns = num_classes**embedding_dim
    counts = _pattern_counts(codes, num_patterns, dtype=tensor.dtype)
    entropy = _shannon_entropy(counts / counts.sum(dim=-1, keepdim=True))
    # NaN is applied to the whole entropy as a tensor op, which keeps this torch.vmap-compatible.
    entropy = torch.where(has_nan, torch.full_like(entropy, float("nan")), entropy)
    if normalize:
        entropy = entropy / math.log(num_patterns)
    return entropy.to(out_dtype)
