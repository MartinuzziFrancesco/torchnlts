"""Lempel-Ziv complexity: the number of phrases a binarized sequence parses into under the incremental copy-or-insert
rule, a combinatorial rather than information-theoretic measure.

Reference: Lempel & Ziv (1976), IEEE Trans. Inform. Theory 22(1), 75-81 (the complexity c(n)); the algorithm
implemented in ``_lz76_count`` is Kaspar & Schuster (1987), Phys. Rev. A 36(2), 842-848, the one in
ComplexityMeasures.jl's ``LempelZiv76`` (identical counts).

A continuous trajectory is binarized first; the default threshold is the trajectory's own mean, as
ComplexityMeasures.jl does (antropy instead expects an already-symbolized sequence). Univariate input only.
Gradient class: none (the parsing is sequential and data-dependent, with no smooth relaxation).

Limitation: the comparison-based parsing is closer to quadratic than linear in the length and runs as a Python loop
per row, so it is not vectorized and not ``torch.vmap``-compatible; a suffix-tree or hash-based parser would be faster.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _output_dtype, _require_univariate, _work_dtype


def _lz76_count(sequence: list) -> int:
    """Kaspar & Schuster (1987) phrase count c(n) (their Fig. 1 algorithm) of a sequence of comparable symbols."""
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


def lempel_ziv_complexity(trajectory, *, threshold="mean", normalize: bool = False) -> torch.Tensor:
    """Lempel-Ziv complexity of a binarized trajectory.

    Gradient class: none.

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]`` (``D > 1`` is rejected). A tensor, numpy array, or
            pandas Series/DataFrame.
        threshold: ``"mean"`` to binarize at each row's own mean, or a numeric threshold applied to every element.
            Binarization is ``value >= threshold -> 1, else 0``; a non-finite threshold follows IEEE 754 comparison
            rules and is not checked.
        normalize: Divide by ``T / log2(T)`` (Amigo et al. 2004; ComplexityMeasures.jl's ``complexity_normalized``)
            to make results comparable across lengths: near 0 for very regular sequences, near 1 for random ones
            (can exceed 1).

    Returns:
        Scalar (unbatched trajectory) or ``[B]`` tensor. The phrase count as ``torch.long`` (``normalize=False``), or
        the normalized ratio in the input's floating dtype (the default dtype for integer input). A row containing
        ``NaN`` or ``inf`` gives ``-1`` (``normalize=False``) or ``NaN`` (``normalize=True``).

    Raises:
        ValueError: If ``trajectory`` has a feature dim other than 1 or fewer than 2 time points.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor, "lempel_ziv_complexity")
    if tensor.shape[-2] < 2:
        raise ValueError(f"lempel_ziv_complexity needs at least 2 time points, got {tensor.shape[-2]}")
    out_dtype = _output_dtype(tensor)
    values = tensor.squeeze(-1).to(_work_dtype(tensor.dtype))  # [B, T]
    finite = torch.isfinite(values).all(dim=-1)  # [B]

    if threshold == "mean":
        cutoff = values.mean(dim=-1, keepdim=True)
    else:
        cutoff = torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    binary = (values >= cutoff).to(torch.int64)

    counts = [_lz76_count(row.tolist()) if row_finite else -1 for row, row_finite in zip(binary, finite.tolist())]
    result = torch.tensor(counts, dtype=torch.int64, device=values.device)
    if normalize:
        length = values.shape[-1]
        result = result.to(values.dtype) / (length / math.log2(length))
        result = torch.where(finite, result, torch.full_like(result, float("nan"))).to(out_dtype)

    if ndim == 3:
        return result
    return result.squeeze(0)
