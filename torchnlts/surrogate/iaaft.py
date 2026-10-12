"""Iterative amplitude-adjusted Fourier transform (IAAFT) surrogate data.

Reference: Schreiber, T. & Schmitz, A. (1996), "Improved Surrogate Data for Nonlinearity Tests", Phys. Rev. Lett.
77(4), 635-638. AAFT (:func:`~torchnlts.surrogate.aaft_surrogate`) matches the amplitude distribution exactly but the
spectrum only loosely. IAAFT starts from a random permutation of the data and repeats ``max_iterations`` times: (1)
replace the Fourier magnitude of the current surrogate by the data's, keeping its phase; (2) reorder the sorted data to
the rank order of the result. The amplitude distribution is exact (the surrogate permutes the data); the spectrum is
approximate, since exact simultaneous matching is generally unattainable for a nontrivial surrogate. The residual
relative spectral error depends on the random start and can vary by about two orders of magnitude across seeds on short
series; TimeseriesSurrogates.jl shows the same spread.

Differences from TimeseriesSurrogates.jl's ``IAAFT``: the start is a true random permutation of the data (Julia samples
indices with replacement, which only changes the path to the fixed point, since the first rank restoration yields a
permutation either way), and exactly ``max_iterations`` rounds run (Julia also stops once a binned-periodogram error
plateaus).

Limitation: fixed iteration count, no per-channel early stopping; add a convergence check on the relative spectral
error if the default ``max_iterations`` proves wasteful.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _output_dtype, _restore_shape_preserving, _work_dtype
from torchnlts.surrogate.aaft import _TIME_DIM, _rank


def iaaft_surrogate(
    trajectory,
    *,
    max_iterations: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Iterative amplitude-adjusted Fourier transform surrogate of ``trajectory``.

    Reference: Schreiber & Schmitz (1996), Phys. Rev. Lett. 77, 635.

    Gradient class: none, by policy (the subpackage is analysis-only; the rank restoration is also a discrete
    permutation).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]`` with ``T >= 1``, computed in the input's floating dtype
            (the default dtype for integer input; half precision in float32, cast back). A tensor, numpy array, or
            pandas Series/DataFrame. Each channel and
            batch element is surrogated independently.
        max_iterations: Number of spectral/amplitude adjustment rounds, ``>= 1`` (TimeseriesSurrogates.jl's ``M``,
            same default). There is no early stopping.
        generator: Optional ``torch.Generator`` for the initial random permutation.

    Returns:
        Surrogate of the same shape as ``trajectory``, in its floating dtype. Its values are permuted along time (per
        channel), in general matching its power spectrum more closely than :func:`~torchnlts.surrogate.aaft_surrogate`.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``, or ``[B, T, D]``, or is complex, or
            ``max_iterations`` is not an integer ``>= 1``.
    """
    max_iterations = _as_int(max_iterations, "max_iterations")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    batched = batched.to(_work_dtype(batched.dtype))
    num_points = batched.shape[_TIME_DIM]

    target_magnitude = torch.fft.rfft(batched, dim=_TIME_DIM).abs()
    sorted_data = torch.sort(batched, dim=_TIME_DIM, stable=True).values

    permutation_keys = torch.rand(batched.shape, generator=generator, dtype=batched.dtype, device=batched.device)
    permutation = torch.argsort(permutation_keys, dim=_TIME_DIM)
    surrogate = torch.gather(batched, _TIME_DIM, permutation)

    for _ in range(max_iterations):
        phase = torch.angle(torch.fft.rfft(surrogate, dim=_TIME_DIM))
        spectrally_adjusted = target_magnitude * torch.exp(1j * phase)
        surrogate = torch.fft.irfft(spectrally_adjusted, n=num_points, dim=_TIME_DIM)
        surrogate = torch.gather(sorted_data, _TIME_DIM, _rank(surrogate))

    return _restore_shape_preserving(surrogate.to(out_dtype), ndim)
