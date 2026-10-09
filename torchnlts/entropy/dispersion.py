"""Dispersion entropy: Shannon entropy over a delay-embedded sequence's
pattern-frequency distribution, like :mod:`torchnlts.entropy.permutation`,
but with the symbols coming from Gaussian-CDF amplitude classification
instead of ordinal (rank-order) patterns -- so, unlike permutation
entropy, it's sensitive to the actual magnitude of each sample, not
just the relative order of nearby ones.

Reference: Rostaghi, M. & Azami, H. (2016), "Dispersion Entropy: A
Measure for Time-Series Analysis", IEEE Signal Processing Letters
23(5), 610-614. Equations quoted here are as restated (citing the same
paper) in Azami, Rostaghi, Abasolo & Escudero (2017), "Refined
Composite Multiscale Dispersion Entropy and its Application to
Biomedical Signals", IEEE Trans. Biomed. Eng., DOI
10.1109/TBME.2017.2679136 (arXiv:1606.01379), eqs. 2-4, since the
original IEEE SPL letter wasn't directly accessible -- both papers share an
author and the later one explicitly restates the earlier one's
definition rather than introducing a new one, so this is not a
"closest available substitute", but the identical formula.

Algorithm:
  1) Map each value x_j to y_j in (0, 1) via the Normal CDF using the
     trajectory's own empirical mean/std (eq. 2).
  2) Linearly assign y_j to an integer class z_j in {1, ..., c}:
     z_j = round(c * y_j + 0.5).
  3) Delay-embed the resulting class sequence at dimension m, lag d;
     each embedded window is a "dispersion pattern" (c^m possible values).
  4) Shannon entropy of the observed patterns' relative frequencies
     (eq. 4), optionally normalized by ln(c^m) (in [0, 1]).

Built on already-validated primitives for steps 3-4 (delay embedding,
the same one-hot-count-then-Shannon-entropy pattern
:func:`torchnlts.entropy.permutation_entropy` already
uses) -- step 1-2's Gaussian-CDF classification is this module's one
genuinely new piece.

Univariate only, same reasoning as permutation entropy: no convention
exists for combining multiple channels into one symbol sequence.

Gradient class: none. Classification into discrete classes (a
``round``) and the resulting histogram-based Shannon entropy have the
same fundamental discreteness as permutation entropy's ordinal-pattern
histogram (see its own module docstring for why no soft/kernel variant
exists there) -- no soft variant here either.

Unlike :func:`torchnlts.entropy.lempel_ziv_complexity`
(this subpackage's other recent addition), this function is fully
``torch.vmap``-compatible: the NaN handling for a zero-variance
trajectory (see :func:`~torchnlts.entropy.dispersion_entropy`'s own Returns docstring)
masks via ``torch.where`` before the one required integer conversion,
never branches on tensor content with a Python ``if``, and never
leaves torch tensors for a Python-level loop -- confirmed directly,
not just assumed (see this module's own vmap test).
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _restore_shape_preserving
from torchnlts.embedding.delay import delay_embedding


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(
            f"dispersion_entropy only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}"
        )


def _gaussian_cdf_classes(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Eq. 2 + the linear class assignment: map each value to an
    integer class in ``{1, ..., num_classes}`` via the trajectory's own
    empirical-mean/std Normal CDF.
    """
    mean = tensor.mean(dim=-2, keepdim=True)
    std = tensor.std(dim=-2, keepdim=True)
    cdf = 0.5 * (1 + torch.erf((tensor - mean) / (std * math.sqrt(2))))
    classes = torch.round(num_classes * cdf + 0.5)
    return classes.clamp(min=1, max=num_classes)


def dispersion_entropy(
    trajectory,
    num_classes: int = 6,
    embedding_dim: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> torch.Tensor:
    """Dispersion entropy: Shannon entropy of Gaussian-CDF-classified
    dispersion-pattern frequencies.

    Gradient class: none (see module docstring).

    Reference: Rostaghi & Azami (2016), eqs. as restated in Azami,
    Rostaghi, Abasolo & Escudero (2017, arXiv:1606.01379), eqs. 2-4.

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (a
            multivariate ``D > 1`` trajectory is rejected -- see module
            docstring). A tensor, numpy array, or pandas Series/DataFrame.
        num_classes: Number of Gaussian-CDF classes c. The paper's own
            recommendation is ``2 < c < 9``; this module defaults to
            ``c=6``, matching the follow-up paper's own choice "for all
            signals". Must be at least 2 (``c=1`` gives exactly one
            possible pattern regardless of the data, a degenerate case
            the paper itself notes should be avoided).
        embedding_dim: Pattern length m. Commonly 2 or 3 in the
            literature -- the number of possible patterns is
            ``num_classes ** embedding_dim``, both the size of the
            one-hot-counted intermediate tensor and (via delay
            embedding) a multiplicative factor on compute; keep this
            product modest (matching the literature's own recommended
            range) rather than assuming it scales as freely as
            ``sample_entropy``'s continuous radius comparison does.
        delay: Embedding lag d. The paper recommends ``d=1``
            ("aliasing may occur for d > 1").
        normalize: Divide by ``log(num_classes ** embedding_dim)`` so
            the result lies in ``[0, 1]`` (the paper's NDE).

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` (batched) entropy, in
        nats (or dimensionless if ``normalize=True``). ``NaN`` when the
        Gaussian-CDF classification divides by an exactly-zero ``std``
        -- reliable for an exactly-constant trajectory in a well-behaved
        dtype (e.g. float64), but a trajectory that's constant only up
        to floating-point rounding (confirmed for a "constant" float32
        tensor) can have a tiny nonzero computed ``std`` instead,
        giving a low-but-finite entropy rather than ``NaN`` -- both are
        defensible readings of a near-degenerate signal, but only the
        exactly-zero case is guaranteed to trip this. ComplexityMeasures.jl's
        ``Dispersion`` handles the exactly-constant case differently
        still, returning ``-0.0`` rather than an error or ``NaN`` --
        confirmed live, a genuine deviation from this function, not
        just a difference in convention this function happens to avoid
        testing (see this module's tests).

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1,
            ``num_classes < 2``, or ``trajectory`` is too short for
            ``embedding_dim``/``delay`` (see
            :func:`torchnlts.embedding.delay_embedding`).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes!r}")
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor)
    classes = _gaussian_cdf_classes(tensor, num_classes)
    # classes has _ensure_batched's always-3-D shape; restore the caller's
    # own [T]/[T,1]/[B,T,1] convention before handing it to delay_embedding,
    # which does its own batch-shape handling and would otherwise treat an
    # originally-unbatched [1,T,1] as a genuinely batched input, leaving a
    # spurious size-1 batch dim on the final result.
    classes = _restore_shape_preserving(classes, ndim)
    embedded = delay_embedding(classes, embedding_dim, delay)

    # A zero-variance trajectory divides by std=0 in _gaussian_cdf_classes,
    # giving NaN classes here. Detect that on the still-floating-point
    # classes (NaN propagates through the arithmetic below either way),
    # then substitute a safe dummy class so the rest of this function can
    # move to integer arithmetic -- required for correctness, not just
    # style: computing pattern codes in a floating dtype loses precision
    # once num_classes**embedding_dim exceeds float32's 2**24 exactly-
    # representable-integer ceiling, silently colliding two different
    # patterns onto the same code (confirmed directly: embedding_dim=25,
    # num_classes=2 collides two provably-different digit sequences).
    # Converting a NaN float straight to int64 is itself undefined
    # behavior (crashes one_hot with an out-of-range/negative class), so
    # the substitution has to happen here, before that conversion.
    has_nan = torch.isnan(embedded).any(dim=(-2, -1))
    embedded = torch.where(torch.isnan(embedded), torch.ones_like(embedded), embedded).to(torch.int64)

    weights = num_classes ** torch.arange(embedding_dim - 1, -1, -1, dtype=torch.int64, device=embedded.device)
    codes = ((embedded - 1) * weights).sum(dim=-1)
    # Re-apply NaN to the whole entropy at the end, not per-code -- same
    # pattern as permutation_entropy/approximate_entropy's own NaN
    # handling: a plain tensor op, not a data-dependent `if`, so this
    # stays torch.vmap-compatible.

    num_patterns = num_classes**embedding_dim
    dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
    counts = torch.nn.functional.one_hot(codes, num_classes=num_patterns).sum(dim=-2).to(dtype)
    probability = counts / counts.sum(dim=-1, keepdim=True)
    log_probability = torch.log(probability.clamp(min=torch.finfo(probability.dtype).tiny))
    entropy = -(probability * log_probability).sum(dim=-1)
    entropy = torch.where(has_nan, torch.full_like(entropy, float("nan")), entropy)
    if normalize:
        entropy = entropy / math.log(num_patterns)
    return entropy
