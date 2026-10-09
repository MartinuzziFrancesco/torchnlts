"""Rank-based surrogate hypothesis test.

References: Theiler, Eubank, Longtin, Galdrikian & Farmer (1992),
Physica D 58, 77; Schreiber & Schmitz (2000), "Surrogate time series",
Physica D 142, 346. A discriminating statistic is computed on the data
and on ``M`` surrogates; the data's rank among the surrogates gives a
Monte Carlo p-value that does not assume a distribution of the statistic.

With ``k`` the number of surrogate statistics at least as extreme as the
data's (ties count as extreme), the p-value is ``(1 + k) / (M + 1)``,
the exact p-value of the rank test when the data is exchangeable with
the surrogates (Davison & Hinkley 1997); it is never zero.
TimeseriesSurrogates.jl's ``pvalue(SurrogateTest(...))`` uses
``k / M`` with the same ``k`` for tails ``:left`` and ``:right``; the two
differ by the added data sample. For ``tail="both"`` the p-value is
``2 * (1 + min(k_left, k_right)) / (M + 1)``, clipped to 1 (Julia's
``2 * min(k_left, k_right) / M`` is not clipped).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

from torchnlts._shape import _as_int, _ensure_batched, _output_dtype, _restore_batch

_ELEMENT_BUDGET = 2**24  # max surrogate elements generated per statistic call

_TAILS = ("left", "right", "both")


class SurrogateTestResult(NamedTuple):
    """``p_value`` ``[B]``, ``statistic`` ``[B]`` and ``surrogate_statistics``
    ``[B, M]``; the leading ``B`` is dropped for an input without a batch
    dimension."""

    p_value: torch.Tensor
    statistic: torch.Tensor
    surrogate_statistics: torch.Tensor


def surrogate_test(
    trajectory,
    statistic,
    surrogate,
    num_surrogates: int = 99,
    tail: str = "right",
    generator: torch.Generator | None = None,
    **surrogate_kwargs,
) -> SurrogateTestResult:
    """Rank-based surrogate test of ``statistic`` on ``trajectory``.

    Reference: Theiler et al. (1992), Physica D 58, 77; Schreiber &
    Schmitz (2000), Physica D 142, 346.

    Gradient class: none (a rank count; all computation is detached).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        statistic: Callable mapping a ``[B, T, D]`` tensor to one value
            per row (shape ``[B]``). Called on the data and on stacks
            of surrogates.
        surrogate: Surrogate generator with the signature
            ``surrogate(trajectory, generator=None, **kwargs)``, e.g.
            :func:`~torchnlts.surrogate.iaaft_surrogate`.
        num_surrogates: Number of surrogates ``M`` per row, ``>= 1``.
            The smallest possible p-value is ``1 / (M + 1)``.
        tail: ``"right"`` rejects when the data statistic is
            large (counts surrogates ``>=`` it), ``"left"`` when it is
            small (counts ``<=``), ``"both"`` uses twice the smaller
            one-sided count.
        generator: Optional ``torch.Generator`` forwarded to
            ``surrogate``.
        **surrogate_kwargs: Forwarded to ``surrogate``.

    Returns:
        :class:`~torchnlts.surrogate.SurrogateTestResult`. The p-value has the dtype of
        ``trajectory`` (the default float dtype for non-float input);
        the statistics keep their own dtype, and ranking is done in the
        promoted common dtype of the data and surrogate statistics. A
        row whose trajectory, data statistic or any surrogate statistic
        is non-finite has a NaN p-value; other rows are unaffected. Surrogates are generated
        in chunks of about ``2**24`` elements per statistic call.
        Limitation: the surrogates drawn for a given generator state
        depend on the chunking whenever the generator's draws depend on
        the batch shape; pass a fixed ``generator`` state and keep the
        input size fixed for reproducibility across machines.

    Raises:
        ValueError: If ``num_surrogates < 1``, ``tail`` is unknown,
            ``statistic`` or ``surrogate`` is not callable, or
            ``statistic`` does not return one value per row.
    """
    num_surrogates = _as_int(num_surrogates, "num_surrogates")
    if num_surrogates < 1:
        raise ValueError(f"num_surrogates must be >= 1, got {num_surrogates}")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
    if not callable(statistic) or not callable(surrogate):
        raise ValueError("statistic and surrogate must be callable")

    batched, ndim = _ensure_batched(trajectory)
    batch_size = batched.shape[0]

    def evaluate(data: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(statistic(data))
        if values.numel() != data.shape[0]:
            raise ValueError(f"statistic must return one value per row ({data.shape[0]}), got shape {tuple(values.shape)}")
        # clone: a view (e.g. data[:, 0, 0]) would keep the whole surrogate chunk alive.
        return values.detach().reshape(-1).clone()

    with torch.no_grad():
        p_dtype = _output_dtype(batched)
        data_statistic = evaluate(batched)
        chunk = max(1, _ELEMENT_BUDGET // max(1, batched.numel()))
        chunks = []
        for first in range(0, num_surrogates, chunk):
            count = min(chunk, num_surrogates - first)
            surrogates = surrogate(batched.repeat(count, 1, 1), generator=generator, **surrogate_kwargs)
            chunks.append(evaluate(surrogates).reshape(count, batch_size))
            del surrogates
        common = functools.reduce(torch.promote_types, [item.dtype for item in chunks], data_statistic.dtype)
        surrogate_statistics = torch.cat([item.to(common) for item in chunks], dim=0).t()  # [B, M]
        data_statistic = data_statistic.to(common)

        reference = data_statistic.unsqueeze(-1)
        # Counts include the data itself, which ties with itself.
        right = (surrogate_statistics >= reference).sum(dim=-1) + 1
        left = (surrogate_statistics <= reference).sum(dim=-1) + 1
        extreme = {"right": right, "left": left, "both": 2 * torch.minimum(right, left)}[tail]
        # The ratio in float64: integer counts can exceed the half-precision range.
        p_value = (extreme.double() / (num_surrogates + 1)).clamp(max=1.0).to(p_dtype)
        valid = (
            torch.isfinite(batched).flatten(1).all(dim=-1) & torch.isfinite(data_statistic) & torch.isfinite(surrogate_statistics).all(dim=-1)
        )
        p_value = torch.where(valid, p_value, torch.full_like(p_value, float("nan")))

    return SurrogateTestResult(
        _restore_batch(p_value, ndim), _restore_batch(data_statistic, ndim), _restore_batch(surrogate_statistics, ndim)
    )
