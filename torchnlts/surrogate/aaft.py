"""Amplitude-adjusted Fourier transform (AAFT) surrogate data, from
observed data.

Reference: Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer,
J.D. (1992), "Testing for nonlinearity in time series: the method of
surrogate data", Physica D 58(1-4), 77-94: extend phase randomization
(:func:`torchnlts.surrogate.phase.phase_randomized_surrogate`) to also
preserve a signal's exact amplitude distribution (its set of values, not
merely its power spectrum), for signals whose own marginal distribution
isn't Gaussian -- plain phase randomization implicitly assumes
near-Gaussian data, and its amplitude distribution isn't well preserved
by that assumption when the data genuinely isn't Gaussian. Three steps:
(1) rank-order Gaussian white noise to match the original data's own
rank order, giving a Gaussian-shaped series with the data's temporal
structure; (2) phase-randomize *that* series; (3) rank-order the
original data's own values according to *that* surrogate's rank order.
The result has exactly the original data's amplitude distribution (a
permutation of its own values, per channel) and approximately its power
spectrum.

Confirmed, deliberate deviation from TimeseriesSurrogates.jl's own
``AAFT``: reading its source (both v2.7.1 and v2.8.0, the two versions
installed for this repo's Julia validation) shows it skips step (1)
entirely -- it phase-randomizes the original data directly (identical to
its own ``RandomFourier``) and then only does step (3), never
constructing the Gaussian-shaped intermediate series the cited paper
defines. Confirmed live, not just by reading source: running
``AAFT`` and ``RandomFourier`` from the same seed on the same input in
``juliaenv`` and rank-restoring the ``RandomFourier`` surrogate to the
data by hand reproduces ``AAFT``'s output exactly -- see this module's
test. Locked in as a disagreement rather than reconciled, same playbook
as :func:`torchnlts.lyapunov.kantz.kantz_divergence`'s own confirmed
Julia deviation.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_shape_preserving
from torchnlts.surrogate.phase import phase_randomized_surrogate

_TIME_DIM = -2


def _rank(tensor: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(tensor, dim=_TIME_DIM, stable=True), dim=_TIME_DIM, stable=True)


def aaft_surrogate(trajectory, generator: torch.Generator | None = None) -> torch.Tensor:
    """Amplitude-adjusted Fourier transform surrogate of ``trajectory``.

    Gradient class: none, by policy (see
    :func:`torchnlts.surrogate.phase.phase_randomized_surrogate`'s
    docstring for why a fixed random draw doesn't itself block gradient
    flow through *that* step). This function's own final step is a
    further, distinct source of non-smoothness beyond the policy: rank
    restoration via ``torch.gather`` on integer rank indices is a
    discrete permutation, so unlike a fixed phase draw's ordinary
    differentiability, the map here is only piecewise-defined even
    before the "none" policy is applied -- a perturbation of
    ``trajectory`` that changes any pair's rank order changes which
    value gets gathered where, discontinuously.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``, with
            ``T >= 1`` (an empty time axis reaches ``rfft`` inside
            :func:`~torchnlts.surrogate.phase.phase_randomized_surrogate`
            and raises from there), real floating-point dtype
            (``float32``/``float64``). A tensor, numpy array, or pandas
            Series/DataFrame. Each feature channel (and batch element)
            is surrogated independently.
        generator: Optional ``torch.Generator``, forwarded in sequence
            to ``torch.randn`` (the Gaussian-noise draw) and then to
            :func:`~torchnlts.surrogate.phase.phase_randomized_surrogate`
            (the phase draw) -- both advance the same generator's state,
            see that function's docstring for reproducibility (including
            its caveat that reuse across calls is only *generally*, not
            absolutely, guaranteed to differ).

    Returns:
        Surrogate of the same shape and dtype as ``trajectory``: exactly
        ``trajectory``'s own values, permuted along the time axis (per
        channel) to approximately match its power spectrum. A fresh
        random draw every call.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``,
            or ``[B, T, D]``.
    """
    batched, ndim = _ensure_batched(trajectory)

    noise = torch.randn(batched.shape, generator=generator, dtype=batched.dtype, device=batched.device)
    sorted_noise = torch.sort(noise, dim=_TIME_DIM, stable=True).values
    gaussianized = torch.gather(sorted_noise, _TIME_DIM, _rank(batched))

    ft_surrogate = phase_randomized_surrogate(gaussianized, generator=generator)

    sorted_data = torch.sort(batched, dim=_TIME_DIM, stable=True).values
    surrogate = torch.gather(sorted_data, _TIME_DIM, _rank(ft_surrogate))

    return _restore_shape_preserving(surrogate, ndim)
