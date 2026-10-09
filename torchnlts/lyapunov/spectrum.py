"""Full Lyapunov spectrum of a differentiable dynamics model, via the
Benettin/QR method (Geist, Parlitz & Lauterborn's "H2" algorithm).

Reference: Benettin, Galgani, Giorgilli & Strelcyn (1980), "Lyapunov
Characteristic Exponents for smooth dynamical systems and for
Hamiltonian systems; a method for computing all of them", Meccanica
15(1), 9-20 & 21-30 (the original method); Geist, K., Parlitz, U. &
Lauterborn, W. (1990), "Comparison of Different Methods for Computing
Lyapunov Exponents", Progress of Theoretical Physics 83(5), 875-893 (the
specific QR-based "H2" variant implemented here).

Matches ChaosTools.jl's ``lyapunovspectrum``: evolve the state one step
via ``step_fn``, propagate a ``[D, num_exponents]`` deviation matrix
through ``step_fn``'s Jacobian at the *pre-step* state, QR-decompose it,
accumulate ``log(abs(diag(R)))``, then replace the deviations with the
orthonormal ``Q`` before the next step. After ``num_steps`` (ChaosTools.jl's
``N``), divide by ``num_steps * dt`` (its ``current_time(tands) - t0``).
Cross-checked live against ``lyapunovspectrum`` in ``juliaenv`` on the
Hénon map (full spectrum, a partial spectrum, and with a ``Ttr``
transient) in ``tests/test_lyapunov_spectrum.py``, matching this
implementation to float64 precision for the same map, initial condition,
and step count.

Doesn't sort the result. The algorithm's Gram-Schmidt-style
orthonormalization yields it in descending order whenever the deviation
vectors mix across steps -- true generically, including every chaotic
system this has been tested against -- but this isn't unconditional:
confirmed by direct comparison, ChaosTools.jl's own ``lyapunovspectrum``
returns the *same* non-descending spectrum as this function on a
diagonal linear map whose diagonal entries aren't already sorted (e.g.
``step_fn(x) = [0.5, 2.0] * x``), since the coordinate-axis deviation
vectors both implementations start from are then exact invariant
directions of the map with no mixing between them, so neither QR
decomposition ever has a chance to reorder them. This is a shared,
inherited limitation of the Benettin/QR method's initial-condition
sensitivity, not a divergence between this implementation and its
reference -- see ``test_lyapunov_spectrum_diagonal_map_can_be_non_descending``.

Gradient stability note: differentiating this w.r.t. anything upstream
of ``initial_state`` or ``step_fn`` (e.g. training a differentiable
dynamics model with a Lyapunov-based loss) means backpropagating through
``num_steps`` chained applications of ``step_fn``, its Jacobian, and a QR
decomposition each -- exactly the products-of-state-Jacobians structure
Metz et al., "Gradients are Not All You Need" (2021/2022,
https://arxiv.org/abs/2111.05803), show produces exploding-variance
gradients in chaotic systems. This function applies no mitigation of its
own (no truncated BPTT, no gradient clipping) and exposes no hook to add
one inside its own ``num_steps`` loop: keep ``num_steps`` modest, clip
gradients on the parameters being trained in the surrounding optimizer
step, or -- if ``initial_state`` is itself produced by a longer outer
rollout the caller controls -- detach state periodically in *that*
rollout (truncated BPTT), upstream of ever calling this function.
"""

from __future__ import annotations

import operator
from typing import Callable

import torch
from torch.func import vmap

from torchnlts.lyapunov.jacobian import step_jacobian


def lyapunov_spectrum(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    num_steps: int,
    num_exponents: int | None = None,
    dt: float = 1.0,
    transient_steps: int = 0,
) -> torch.Tensor:
    """Lyapunov exponent spectrum via the Benettin/QR method.

    Gradient class: exact (see the module docstring's gradient-stability
    caveat for chaotic ``step_fn``; ``torch.linalg.qr``'s backward
    additionally requires the deviation matrix to have full column rank
    at every step).

    Args:
        step_fn: See :func:`torchnlts.lyapunov.step_jacobian` --
            one discrete step of the dynamics, or one fixed-step
            continuous-flow integration (e.g. via ``torchdiffeq``).
        initial_state: Shape ``[D]`` or ``[B, D]``.
        num_steps: Number of steps to average the QR growth rates over
            (ChaosTools.jl's ``N``). Must be positive.
        num_exponents: Number of exponents to compute,
            ``1 <= num_exponents <= D``. Defaults to the full spectrum
            (``D``).
        dt: Time per step, dividing the accumulated log-growth
            (ChaosTools.jl's ``Δt``; ``1.0`` for a discrete map). Must be
            positive.
        transient_steps: Extra steps to evolve the state and orient the
            deviation vectors before accumulating growth rates
            (ChaosTools.jl's ``Ttr``), discarded from the average. Must
            be non-negative.

    Returns:
        Lyapunov exponents, shape ``[num_exponents]`` for an unbatched
        ``initial_state`` or ``[B, num_exponents]`` for a batched one.
        Descending order in general (see the module docstring), except
        in the confirmed non-generic case documented there. ``-inf`` at
        any exponent whose growth rate is exactly ``0`` at some step
        (e.g. a degenerate ``step_fn``), matching ``log(0)`` rather than
        silently flooring it, and ``NaN`` if ``step_fn`` or the QR
        decomposition itself ever produces one -- including from a
        rank-deficient deviation matrix, which the forward pass tolerates
        but whose backward the Gradient class note above does not.

    Raises:
        ValueError: If ``initial_state`` is not 1D or 2D, ``num_steps``
            is not positive, ``num_exponents`` is out of ``[1, D]``,
            ``dt`` is not positive, or ``transient_steps`` is negative.
    """
    if initial_state.dim() not in (1, 2):
        raise ValueError(f"expected initial_state of shape [D] or [B, D], got {tuple(initial_state.shape)}")
    state_dim = initial_state.shape[-1]
    num_steps = operator.index(num_steps)
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    num_exponents = state_dim if num_exponents is None else operator.index(num_exponents)
    if not (1 <= num_exponents <= state_dim):
        raise ValueError(f"num_exponents must be in [1, {state_dim}], got {num_exponents}")
    dt = float(dt)
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    transient_steps = operator.index(transient_steps)
    if transient_steps < 0:
        raise ValueError(f"transient_steps must be non-negative, got {transient_steps}")

    batched = initial_state.dim() == 2
    advance = vmap(step_fn) if batched else step_fn

    deviations = torch.eye(state_dim, num_exponents, dtype=initial_state.dtype, device=initial_state.device)
    if batched:
        deviations = deviations.unsqueeze(0).expand(initial_state.shape[0], -1, -1).clone()

    state = initial_state
    log_growth = torch.zeros(
        deviations.shape[:-2] + (num_exponents,), dtype=initial_state.dtype, device=initial_state.device
    )

    for step in range(transient_steps + num_steps):
        jacobian = step_jacobian(step_fn, state)
        deviations = jacobian @ deviations
        q, r = torch.linalg.qr(deviations, mode="reduced")
        if step >= transient_steps:
            log_growth = log_growth + torch.diagonal(r, dim1=-2, dim2=-1).abs().log()
        deviations = q
        state = advance(state)

    return log_growth / (num_steps * dt)
