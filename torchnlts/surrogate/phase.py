"""Phase-randomized (random Fourier phase) surrogate data.

Reference: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), "Testing for nonlinearity in time series: the method
of surrogate data", Physica D 58(1-4), 77-94. The Fourier magnitude (hence the power spectrum and autocorrelation) is
kept and the phases are randomized, which preserves linear statistics and destroys nonlinear structure. The null
hypothesis is a linear Gaussian process.

Algorithm: center the signal, ``rfft``, draw a phase uniform on ``[0, 2*pi)`` for every bin, ``irfft``, add the mean
back. The DC phase is set to 0 and, for even length, the Nyquist phase is set to 0 or ``pi`` (a random sign), as
``phi(f) = -phi(-f)`` requires; the mean and every Fourier magnitude are preserved. TimeseriesSurrogates.jl's
``RandomFourier`` draws a uniform phase for the Nyquist bin too, so ``irfft`` scales its magnitude by ``|cos(phase)|``.
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _ensure_batched, _output_dtype, _restore_shape_preserving, _work_dtype


def phase_randomized_surrogate(trajectory, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Phase-randomized surrogate of ``trajectory``.

    Reference: Theiler et al. (1992), Physica D 58, 77.

    Gradient class: none, by policy (the subpackage is analysis-only; for a fixed phase draw the map is ordinary
    autodiff).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]`` with ``T >= 1``, computed in the input's floating dtype
            (the default dtype for integer input; half precision in float32, cast back). A tensor, numpy array, or
            pandas Series/DataFrame. Each channel and
            batch element is randomized independently, so cross-channel correlation is not preserved.
        generator: Optional ``torch.Generator`` for the phase draw (default: the global RNG). It advances on every
            call; re-seed it, or the global seed, before each call for a reproducible surrogate.

    Returns:
        Surrogate of the same shape as ``trajectory``, in its floating dtype. The mean and every Fourier magnitude,
        Nyquist included, are preserved. Unlike TimeseriesSurrogates.jl's ``RandomFourier``, the Nyquist phase of an
        even-length series is 0 or ``pi``.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``, or ``[B, T, D]``, or is complex.
    """
    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    batched = batched.to(_work_dtype(batched.dtype))
    num_points = batched.shape[-2]
    mean = batched.mean(dim=-2, keepdim=True)
    spectrum = torch.fft.rfft(batched - mean, dim=-2)
    magnitude = spectrum.abs()
    draw = torch.rand(magnitude.shape, generator=generator, dtype=magnitude.dtype, device=magnitude.device)
    phase = draw * (2.0 * math.pi)
    phase[..., 0, :] = 0.0
    if num_points % 2 == 0:
        phase[..., -1, :] = math.pi * (draw[..., -1, :] >= 0.5).to(phase.dtype)
    shuffled = magnitude * torch.exp(1j * phase)
    surrogate = torch.fft.irfft(shuffled, n=num_points, dim=-2) + mean
    return _restore_shape_preserving(surrogate.to(out_dtype), ndim)
