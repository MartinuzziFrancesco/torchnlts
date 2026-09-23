"""Lempel-Ziv complexity: counting distinct "phrases" a sequence
parses into under a specific incremental-copy-or-insert rule, a
combinatorial (not information-theoretic) complexity measure.

Reference: Lempel, A. & Ziv, J. (1976), "On the Complexity of Finite
Sequences", IEEE Trans. Inform. Theory 22(1), 75-81 -- defines the
complexity c(n) via that incremental parsing over a finite-alphabet
sequence, but not as a closed-form/efficiently-computable formula.
Kaspar, F. & Schuster, H.G. (1987), "Easily calculable measure for the
complexity of spatiotemporal patterns", Phys. Rev. A 36(2), 842-848,
give the copy/insert algorithm actually implemented here
(:func:`_lz76_count`) -- the standard practical realization of c(n)
used by ComplexityMeasures.jl's ``LempelZiv76`` (confirmed a
symbol-for-symbol match, see this module's tests) and, per its own
docstring's account of the paper, by essentially every reference
implementation. This straightforward comparison-based version is
**not** linear time in practice, despite that reputation -- measured
directly (not assumed): both an adversarial input and ordinary random
binary data show the comparison count growing distinctly faster than
``n`` as ``n`` grows (closer to quadratic than linear for this simple
version; a suffix-tree/hash-based implementation could do better, but
that isn't what this port -- or Wikipedia's pseudocode, or
ComplexityMeasures.jl's own implementation -- does).

This module applies it to a continuous trajectory the way
ComplexityMeasures.jl's own docstring names as common: binarize first
(default: threshold at the trajectory's own mean -- the algorithm
itself has no opinion on alphabet size or how a continuous signal maps
to one, that step is this module's chosen convention, not the paper's;
other references, e.g. antropy, instead expect an already-symbolized
sequence and leave binarization to the caller).

Univariate only, same reasoning as :mod:`torchnlts.entropy.permutation`:
no convention exists for combining multiple channels into one symbol
sequence.

Gradient class: none, and no soft variant exists in this module (unlike
this subpackage's other entropies) -- the incremental parsing is a
sequential, data-dependent control-flow algorithm (which "phrase" ends
depends on which symbols equal which others), not a threshold applied
to a fixed set of pairwise comparisons the way ``sample_entropy``'s
Heaviside step is; there is no natural kernel relaxation of "does this
substring appear earlier" the way there is for "is this distance within
r". Not vectorized across time for the same reason -- implemented as a
plain Python loop per sequence, batched trajectories loop in Python too.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(f"lempel_ziv_complexity only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}")


def _lz76_count(sequence: list) -> int:
    """Kaspar & Schuster (1987)'s incremental-parsing count c(n) (their
    Fig. 1 algorithm, equivalently Wikipedia's "Lempel-Ziv complexity"
    pseudocode -- the same one ComplexityMeasures.jl's ``LempelZiv76``
    implements, confirmed to match it symbol-for-symbol on the same
    input, including an exhaustive check across every binary string of
    length 2-13). Operates on any sequence of hashable/comparable
    symbols; this module always feeds it a binarized ``{0, 1}``
    sequence, but the counting algorithm itself doesn't care about
    alphabet size. See the module docstring for why this isn't linear
    time despite the reputation.
    """
    n = len(sequence)
    i, c, u, v, vmax = 0, 1, 1, 1, 1
    while u + v <= n:
        if sequence[i + v - 1] == sequence[u + v - 1]:
            v += 1
        else:
            vmax = max(v, vmax)
            i += 1
            if i == u:
                c += 1
                u += vmax
                v = 1
                i = 0
                vmax = v
            else:
                v = 1
    if v != 1:
        c += 1
    return c


def lempel_ziv_complexity(trajectory, threshold="mean", normalize: bool = False) -> torch.Tensor:
    """Lempel-Ziv complexity of a binarized trajectory.

    Gradient class: none (see module docstring).

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (a
            multivariate ``D > 1`` trajectory is rejected -- see module
            docstring). A tensor, numpy array, or pandas Series/DataFrame.
        threshold: ``"mean"`` to binarize at the trajectory's own mean
            (per-batch-element if batched -- this module's chosen
            convention, see module docstring), or an explicit numeric
            threshold applied to every element. Binarization is
            ``value >= threshold -> 1, else 0``; a non-finite
            (``NaN``/``inf``) trajectory value or threshold silently
            binarizes to ``0`` under IEEE 754 comparison rules rather
            than raising or propagating -- not detected or guarded
            against.
        normalize: Divide by ``T / log2(T)`` (Amigo et al. 2004's
            normalization, matching ComplexityMeasures.jl's
            ``complexity_normalized``), making results comparable
            across different trajectory lengths. Close to 0 for very
            regular sequences, close to 1 for random ones (can exceed 1).

    Returns:
        Integer-valued count (``normalize=False``) or float ratio
        (``normalize=True``), as a scalar (unbatched trajectory) or
        ``[B]`` (batched) tensor, dtype following ``trajectory`` when
        ``normalize=True`` and ``torch.long`` otherwise.

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1,
            or fewer than 2 time points.

    Not ``torch.vmap``-compatible, unlike most of this library: the
    binarized sequence is converted to a Python list (``.tolist()``)
    per batch row for the sequential parsing loop, which ``vmap``
    cannot trace through. Inherent to this algorithm (see module
    docstring), not an oversight to fix.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor)
    if tensor.shape[-2] < 2:
        raise ValueError(f"lempel_ziv_complexity needs at least 2 time points, got {tensor.shape[-2]}")
    values = tensor.squeeze(-1)  # [B, T]

    if threshold == "mean":
        cutoff = values.mean(dim=-1, keepdim=True)
    else:
        cutoff = torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    binary = (values >= cutoff).to(torch.int64)

    counts = torch.tensor([_lz76_count(row.tolist()) for row in binary], dtype=torch.int64, device=values.device)
    if normalize:
        length = values.shape[-1]
        result = counts.to(values.dtype) / (length / math.log2(length))
    else:
        result = counts

    if ndim == 3:
        return result
    return result.squeeze(0)
