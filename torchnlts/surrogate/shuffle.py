"""Random-shuffle and random circular-shift surrogates.

Reference: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992),
"Testing for nonlinearity in time series: the method of surrogate data",
Physica D 58(1-4), 77-94. A shuffle surrogate permutes the samples
uniformly at random (null: independent, identically distributed
values, possibly observed through a static nonlinearity); a shift
surrogate rotates the series by a random number of steps (preserves the
amplitude distribution, the autocorrelation up to the wrap-around point
and every cyclic window of the data; null: no relation to an external
time-locked signal).

Matches TimeseriesSurrogates.jl's ``RandomShuffle`` (a uniform random
permutation of the values) and ``CircShift``. ``CircShift`` takes the
admissible shifts from the caller; here the shift is uniform on
``0, ..., T - 1``, the whole cyclic group including the identity, which
makes the data exchangeable with its surrogates under the null and the
rank test exact (excluding the identity inflates false rejections). Each feature channel and batch element gets its own
permutation or shift, like the other generators.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_shape_preserving

_TIME_DIM = -2


def random_shuffle_surrogate(trajectory, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random-shuffle surrogate of ``trajectory``.

    Reference: Theiler et al. (1992), Physica D 58, 77.

    Gradient class: none, by policy (see
    :func:`torchnlts.surrogate.phase_randomized_surrogate`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``; any
            dtype. A tensor, numpy array, or pandas Series/DataFrame.
            Each feature channel and batch element is permuted
            independently.
        generator: Optional ``torch.Generator`` for the permutation
            draw. Defaults to the global RNG.

    Returns:
        Tensor of the same shape and dtype as ``trajectory``: its values
        permuted uniformly at random along time. Non-finite values are
        permuted like any other.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``,
            or ``[B, T, D]``.
    """
    batched, ndim = _ensure_batched(trajectory)
    # float64 keys: float32 keys tie often enough for long series to bias the permutation.
    keys = torch.rand(batched.shape, generator=generator, dtype=torch.float64, device=batched.device)
    permutation = torch.argsort(keys, dim=_TIME_DIM)
    return _restore_shape_preserving(torch.gather(batched, _TIME_DIM, permutation), ndim)


def random_shift_surrogate(trajectory, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random circular-shift surrogate of ``trajectory``.

    Reference: Theiler et al. (1992), Physica D 58, 77.

    Gradient class: none, by policy (see
    :func:`torchnlts.surrogate.phase_randomized_surrogate`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``; any
            dtype. A tensor, numpy array, or pandas Series/DataFrame.
            Each feature channel and batch element is shifted
            independently.
        generator: Optional ``torch.Generator`` for the shift draw.
            Defaults to the global RNG.

    Returns:
        Tensor of the same shape and dtype as ``trajectory``: the series
        rotated by ``s`` steps (``surrogate[t] = trajectory[(t - s) mod T]``)
        with ``s`` uniform on ``0, ..., T - 1`` (``s = 0`` returns the data). Non-finite values are
        rotated like any other.

    Raises:
        ValueError: If ``trajectory`` is not shape ``[T]``, ``[T, D]``,
            or ``[B, T, D]``.
    """
    batched, ndim = _ensure_batched(trajectory)
    num_points = batched.shape[_TIME_DIM]
    shift = torch.randint(0, num_points, (batched.shape[0], 1, batched.shape[-1]), generator=generator, device=batched.device)
    index = (torch.arange(num_points, device=batched.device).view(1, -1, 1) - shift) % num_points
    return _restore_shape_preserving(torch.gather(batched, _TIME_DIM, index), ndim)
