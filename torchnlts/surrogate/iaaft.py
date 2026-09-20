"""Iterative amplitude-adjusted Fourier transform (IAAFT) surrogate
data, from observed data.

Reference: Schreiber, T. & Schmitz, A. (1996), "Improved Surrogate Data
for Nonlinearity Tests", Phys. Rev. Lett. 77(4), 635-638: AAFT
(:func:`torchnlts.surrogate.aaft.aaft_surrogate`) already matches the
original data's amplitude distribution exactly (both methods do, by the
same rank-restoration construction) but reaches its spectral match in a
*single* amplitude-adjustment pass, which is generally loose; IAAFT
repeats the same rank-restoration step across many *iterations*, each
preceded by a fresh spectral-adjustment pass, so the surrogate converges
toward a fixed point whose spectrum is a much closer match. Each
iteration: (1) spectral adjustment -- replace the current surrogate's
Fourier magnitude with the *original data's* magnitude, keeping the
current surrogate's own phase, and invert; (2) amplitude adjustment --
rank-order the original data's own values according to that result's
rank order (identical to AAFT's own final step). Repeat for
``max_iterations`` rounds. Exact simultaneous matching of both the
amplitude distribution and the power spectrum isn't generally achievable
for a genuinely decorrelating (nontrivial) surrogate -- trivial cases
like the original data itself, a cyclic shift of it, or its time reversal
satisfy both exactly, as the cited paper notes, but those aren't useful
surrogates. The relative spectral error generally decreases over the
first several iterations and then plateaus at a small but
non-repeatably-sized value that depends on which fixed point the random
initial permutation happens to land near -- confirmed empirically (see
this module's tests) to vary by roughly two orders of magnitude across
otherwise-identical reruns with different seeds (e.g. ~4e-6 to ~3e-3 for
one fixed 10-point test series), and confirmed to be a property of the
method itself, not this port: TimeseriesSurrogates.jl's own ``IAAFT``
shows a comparable, same-order-of-magnitude seed-dependent spread on the
same input (not claimed to be an identical distribution -- the two
implementations' RNG streams are unrelated).

Confirmed deviations from TimeseriesSurrogates.jl's own ``IAAFT``,
reading its source:

- **Initial condition.** This module starts from a true random
  permutation of the data (``argsort`` of random keys, no repeats or
  omissions). TimeseriesSurrogates.jl instead starts from
  ``StatsBase.sample(rng, 1:n, n)``, which samples *with* replacement by
  default (confirmed empirically: it produces both duplicate and missing
  indices) -- so its very first iteration begins from a resampled
  sequence that generally isn't a permutation of the data at all. This
  doesn't affect either implementation's final *validity* (the very
  first iteration's amplitude-adjustment step already rank-restores to
  the data's own values regardless of the starting point), only the
  specific path taken to the fixed point -- a permutation is the more
  standard choice in the literature and was chosen deliberately here,
  not an attempt to replicate Julia's specific choice.
- **Iteration count.** This module always runs exactly
  ``max_iterations`` rounds. TimeseriesSurrogates.jl additionally tracks
  a coarse-grained (binned) periodogram and, each iteration, the
  normalized squared deviation between the *current* surrogate's binned
  periodogram and the *original data's* -- stopping early once that
  deviation metric itself changes by less than a tolerance between
  successive iterations (or ``M`` is reached). Skipped here as a
  deliberate simplification --
  running more iterations than strictly necessary costs extra compute
  but is not a correctness problem, whereas replicating a binned/early
  -stopping criterion that must independently decide convergence *per
  channel and per batch element* (since this module is batched and
  Julia's isn't) is real added complexity for a compute-time optimization,
  not a numerical-quality one.
  ``Limitation: fixed iteration count, no per-channel early stopping; add
  a convergence check (e.g. per-channel relative spectral-error
  plateauing) if the default max_iterations proves wasteful in practice.``
"""

from __future__ import annotations

import operator

import torch

from torchnlts._shape import _ensure_batched, _restore_shape_preserving
from torchnlts.surrogate.aaft import _TIME_DIM, _rank


def iaaft_surrogate(
    trajectory,
    max_iterations: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Iterative amplitude-adjusted Fourier transform surrogate of
    ``trajectory``.

    Gradient class: none, by policy (see
    :func:`torchnlts.surrogate.phase.phase_randomized_surrogate`'s
    docstring). Like :func:`torchnlts.surrogate.aaft.aaft_surrogate`,
    the rank-restoration step run every iteration is itself a discrete
    permutation, not merely policy-excluded from autodiff.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``, with
            ``T >= 1``, real floating-point dtype (``float32``/
            ``float64``). A tensor, numpy array, or pandas
            Series/DataFrame. Each feature channel (and batch element)
            is surrogated independently.
        max_iterations: Number of alternating spectral/amplitude
            adjustment rounds (TimeseriesSurrogates.jl's ``M``, default
            matches its own default of ``100``). Must be positive; see the
            module docstring for why this module doesn't also offer
            early stopping.
        generator: Optional ``torch.Generator`` for the initial random
            permutation, forwarded to ``torch.rand``.

    Returns:
        Surrogate of the same shape and dtype as ``trajectory``: exactly
        ``trajectory``'s own values, permuted along the time axis (per
        channel) to approximately match its power spectrum -- more
        closely than :func:`~torchnlts.surrogate.aaft.aaft_surrogate`'s
        single amplitude-adjustment pass, in general. A fresh random
        draw every call.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``,
            or ``[B, T, D]``, or if ``max_iterations`` is not a positive
            integer.
    """
    max_iterations = operator.index(max_iterations)
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")

    batched, ndim = _ensure_batched(trajectory)
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

    return _restore_shape_preserving(surrogate, ndim)
