"""Per-state Jacobian of a differentiable step function, by ``torch.func.jacrev`` batched with ``vmap``; the building
block of :func:`~torchnlts.lyapunov.lyapunov_spectrum`.

``step_fn`` is any differentiable ``[D] -> [D]`` PyTorch callable: a hand-written map, a ``torch.nn.Module``, or one
fixed-step integration of a flow (e.g. ``torchdiffeq.odeint`` with ``"rk4"``). It must compose with ``jacrev`` and
``vmap``; ``torchdiffeq``'s adaptive ``"dopri5"`` does not (its step-size selection goes through an
``autograd.Function`` without ``setup_context``).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import jacrev, vmap


def _mask_nonfinite_states(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``state`` ``[D]`` or ``[B, D]`` with every state containing NaN/inf replaced by zeros, so ``step_fn`` is never
    evaluated at a non-finite state, and the mask of those states (``[]`` or ``[B]``). No data-dependent control flow,
    so ``torch.vmap`` works."""
    nonfinite = ~torch.isfinite(state).all(dim=-1)
    return torch.where(nonfinite[..., None], torch.zeros_like(state), state), nonfinite


def step_jacobian(step_fn: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor) -> torch.Tensor:
    """Jacobian ``d step_fn(state) / d state``.

    Gradient class: exact (``jacrev`` is reverse-mode autodiff; backpropagating through the Jacobian is a double
    backward through ``step_fn``, as well-behaved as its second derivatives).

    ``step_fn`` is evaluated at the zero state in place of a state containing NaN or inf.
    Limitation: a ``step_fn`` that is singular at the zero state (e.g. ``1 / x``) can still give non-finite parameter
    gradients for a non-finite state.

    Args:
        step_fn: Maps an unbatched ``[D]`` state to the next ``[D]`` state. It is always called unbatched; ``vmap``
            does the batching.
        state: Shape ``[D]`` or ``[B, D]``, a real floating-point tensor (float16 and bfloat16 included).

    Returns:
        The Jacobian, shape ``[D, D]`` or ``[B, D, D]``. NaN for a state (or batch row) containing NaN or inf, without
        affecting the other rows or their gradients.

    Raises:
        ValueError: If ``state`` is not 1-D or 2-D, or not a real floating-point tensor.
    """
    if state.dim() not in (1, 2):
        raise ValueError(f"expected state of shape [D] or [B, D], got {tuple(state.shape)}")
    if not state.is_floating_point():
        raise ValueError(f"state must be a real floating-point tensor, got {state.dtype}")
    state, nonfinite = _mask_nonfinite_states(state)
    jacobian = jacrev(step_fn)(state) if state.dim() == 1 else vmap(jacrev(step_fn))(state)
    return torch.where(nonfinite[..., None, None], torch.full_like(jacobian, float("nan")), jacobian)
