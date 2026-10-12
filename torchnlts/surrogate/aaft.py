"""Amplitude-adjusted Fourier transform (AAFT) surrogate data.

Reference: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), "Testing for nonlinearity in time series: the method
of surrogate data", Physica D 58(1-4), 77-94. Phase randomization
(:func:`~torchnlts.surrogate.phase_randomized_surrogate`) implicitly assumes Gaussian marginals; AAFT also keeps the
exact amplitude distribution: (1) reorder Gaussian white noise to the rank order of the data, (2) phase-randomize that
series, (3) reorder the sorted data to the rank order of the result. The surrogate is a permutation of the data (per
channel) with approximately its power spectrum.

TimeseriesSurrogates.jl's ``AAFT`` omits step (1): it phase-randomizes the data directly and then restores the ranks.
This module follows the paper, so the two give different surrogates for non-Gaussian data.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _output_dtype, _restore_shape_preserving, _work_dtype
from torchnlts.surrogate.phase import phase_randomized_surrogate

_TIME_DIM = -2


def _rank(tensor: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(tensor, dim=_TIME_DIM, stable=True), dim=_TIME_DIM, stable=True)


def aaft_surrogate(trajectory, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Amplitude-adjusted Fourier transform surrogate of ``trajectory``.

    Reference: Theiler et al. (1992), Physica D 58, 77. Unlike TimeseriesSurrogates.jl's ``AAFT``, it includes the
    paper's Gaussian rank-matching step.

    Gradient class: none, by policy (the subpackage is analysis-only; the rank restoration is also a discrete
    permutation).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]`` with ``T >= 1``, computed in the input's floating dtype
            (the default dtype for integer input; half precision in float32, cast back). A tensor, numpy array, or
            pandas Series/DataFrame. Each channel and
            batch element is surrogated independently.
        generator: Optional ``torch.Generator`` for the Gaussian-noise draw and then the phase draw; it advances
            through both (see :func:`~torchnlts.surrogate.phase_randomized_surrogate`).

    Returns:
        Surrogate of the same shape as ``trajectory``, in its floating dtype. Its values are permuted along time (per
        channel) to approximately match its power spectrum.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``, or ``[B, T, D]``, or is complex.
    """
    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    batched = batched.to(_work_dtype(batched.dtype))

    noise = torch.randn(batched.shape, generator=generator, dtype=batched.dtype, device=batched.device)
    sorted_noise = torch.sort(noise, dim=_TIME_DIM, stable=True).values
    gaussianized = torch.gather(sorted_noise, _TIME_DIM, _rank(batched))

    ft_surrogate = phase_randomized_surrogate(gaussianized, generator=generator)

    sorted_data = torch.sort(batched, dim=_TIME_DIM, stable=True).values
    surrogate = torch.gather(sorted_data, _TIME_DIM, _rank(ft_surrogate))

    return _restore_shape_preserving(surrogate.to(out_dtype), ndim)
