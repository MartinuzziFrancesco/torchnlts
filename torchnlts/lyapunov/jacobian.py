"""Per-state Jacobian of a differentiable step function, via
``torch.func``'s reverse-mode ``jacrev`` (batched with ``vmap``) -- the
shared building block for :mod:`torchnlts.lyapunov.spectrum`'s
Benettin/QR Lyapunov spectrum estimator.

``step_fn`` is any differentiable PyTorch callable mapping one state to
the next: a hand-written map, a ``torch.nn.Module``'s ``forward``, or one
fixed-step continuous-flow integration (e.g. ``torchdiffeq.odeint`` with
a fixed-step method like ``"rk4"``). Not every such callable composes
with the ``torch.func`` transforms (``jacrev``/``vmap``) used here:
confirmed by direct test, ``torchdiffeq.odeint`` with a fixed-step method
works under ``jacrev``, but its adaptive-step ``"dopri5"`` does not --
it raises, since its internal step-size-selection routes through an
``autograd.Function`` that doesn't override ``setup_context``, which
``torch.func`` transforms require. This module otherwise makes no
assumption beyond "differentiable ``[D] -> [D]`` function" -- it does not
itself roll out a trajectory, and has no dependency on ``dysts`` or
``torchdiffeq`` (this package doesn't reimplement ODE solvers or
reference systems).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import jacrev, vmap


def step_jacobian(step_fn: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor) -> torch.Tensor:
    """Jacobian d(step_fn(state))/d(state).

    Gradient class: exact (``torch.func.jacrev`` is reverse-mode
    autodiff; composing it under further autograd -- to backpropagate
    through the Jacobian itself, as
    :func:`torchnlts.lyapunov.lyapunov_spectrum` does when
    ``state`` or ``step_fn`` depends on upstream parameters -- is a
    double backward through ``step_fn``, only as well-behaved as
    ``step_fn``'s own second derivatives).

    Args:
        step_fn: Maps an unbatched ``[D]`` state to the next ``[D]``
            state. Always called with an unbatched state, even when
            ``state`` here is batched -- batching is handled by
            ``vmap`` in this function, not inside ``step_fn``.
        state: Shape ``[D]`` or ``[B, D]``.

    Returns:
        Jacobian, shape ``[D, D]`` for an unbatched state or
        ``[B, D, D]`` for a batched one.

    Raises:
        ValueError: If ``state`` is not 1D or 2D.
    """
    if state.dim() == 1:
        return jacrev(step_fn)(state)
    if state.dim() == 2:
        return vmap(jacrev(step_fn))(state)
    raise ValueError(f"expected state of shape [D] or [B, D], got {tuple(state.shape)}")
