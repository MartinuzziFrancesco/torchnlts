"""Phase-randomized (random Fourier phase) surrogate data.

Reference: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992), "Testing for nonlinearity in time series: the method
of surrogate data", Physica D 58(1-4), 77-94. The Fourier magnitude (hence the power spectrum and autocorrelation) is
kept and the phases are randomized, which preserves linear statistics and destroys nonlinear structure. The null
hypothesis is a linear Gaussian process.

Matches TimeseriesSurrogates.jl's ``RandomFourier`` (``phases=true``; the random-amplitude variant is not
implemented): center the signal, ``rfft``, draw a phase uniform on ``[0, 2*pi)`` for every bin including DC and (for
even length) Nyquist, ``irfft``, add the mean back. Neither package special-cases the DC and Nyquist bins, which
Hermitian symmetry requires to be real: ``irfft`` drops the imaginary part, so the Nyquist magnitude is multiplied by
``|cos(phase)|`` and is not preserved. The mean and all interior bins are preserved.
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
        Surrogate of the same shape as ``trajectory``, in its floating dtype. The mean and the interior Fourier magnitudes are
        preserved; the Nyquist magnitude of an even-length series is not.

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
    phase = torch.rand(magnitude.shape, generator=generator, dtype=magnitude.dtype, device=magnitude.device)
    phase = phase * (2.0 * math.pi)
    shuffled = magnitude * torch.exp(1j * phase)
    surrogate = torch.fft.irfft(shuffled, n=num_points, dim=-2) + mean
    return _restore_shape_preserving(surrogate.to(out_dtype), ndim)
