"""Kaplan-Yorke (Lyapunov) dimension from a Lyapunov spectrum.

Reference: Kaplan & Yorke (1979), "Chaotic behavior of multidimensional
difference equations", Lecture Notes in Math. 730, 204-227; Frederickson,
Kaplan, Yorke & Yorke (1983), J. Diff. Eq. 49, 185-207.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _mask_nonfinite_rows, _output_dtype, _work_dtype
from torchnlts.convert import as_tensor


def kaplan_yorke_dimension(spectrum) -> torch.Tensor:
    """Kaplan-Yorke dimension ``k + sum(lambda_1..lambda_k) / |lambda_{k+1}|``.

    Matches FractalDimensions.jl's ``kaplanyorke_dim``: ``k`` is the number
    of leading non-negative partial sums of the descending-sorted spectrum.

    Gradient class: piecewise (continuous, with kinks where a partial sum
    crosses zero or sorted exponents tie; jumps where the largest exponent
    crosses zero).

    Args:
        spectrum: Lyapunov exponents, shape ``[D]`` or ``[B, D]``, as
            returned by ``lyapunov_spectrum`` or ``eckmann_spectrum`` (a
            tensor, numpy array, or pandas object). Any
            order: sorted descending internally (the gradient follows the
            sort).

    Returns:
        Dimension, shape ``[]`` or ``[B]``. ``0`` if the largest exponent is
        negative, ``D`` if no partial sum is negative, and ``NaN`` for a row
        containing NaN/inf (other rows and their gradients are unaffected).

    Raises:
        ValueError: If ``spectrum`` is not 1D or 2D, is empty, or is complex.
    """
    spectrum = as_tensor(spectrum)
    if spectrum.dim() not in (1, 2) or spectrum.shape[-1] == 0:
        raise ValueError(f"expected a non-empty spectrum of shape [D] or [B, D], got {tuple(spectrum.shape)}")
    out_dtype = _output_dtype(spectrum)
    values = spectrum.to(_work_dtype(spectrum.dtype))
    unbatched = values.dim() == 1
    values, nonfinite = _mask_nonfinite_rows(values.reshape(-1, values.shape[-1]))
    ordered = values.sort(dim=-1, descending=True).values
    partial = ordered.cumsum(dim=-1)
    count = (partial >= 0).to(torch.long).cumprod(dim=-1).sum(dim=-1, keepdim=True)
    last = ordered.shape[-1] - 1
    head = partial.gather(-1, (count - 1).clamp(min=0))
    tail = ordered.gather(-1, count.clamp(max=last)).abs()
    # Where count == D or 0 the interpolation term is unused; keep its divisor valid.
    interpolated = (tail > 0) & (count > 0) & (count <= last)
    fraction = torch.where(
        interpolated, head / torch.where(interpolated, tail, torch.ones_like(tail)), torch.zeros_like(head)
    )
    dimension = torch.where(count == 0, torch.zeros_like(head), count.to(head.dtype) + fraction).squeeze(-1)
    dimension = torch.where(nonfinite, torch.full_like(dimension, float("nan")), dimension)
    return (dimension.squeeze(0) if unbatched else dimension).to(out_dtype)
