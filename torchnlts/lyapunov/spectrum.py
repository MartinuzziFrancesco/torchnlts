"""Full Lyapunov spectrum of a differentiable dynamics model, by the Benettin/QR method.

Reference: Benettin, Galgani, Giorgilli & Strelcyn (1980), "Lyapunov Characteristic Exponents for smooth dynamical
systems and for Hamiltonian systems; a method for computing all of them", Meccanica 15(1), 9-20 and 21-30; the QR
variant ("H2") is Geist, K., Parlitz, U. & Lauterborn, W. (1990), "Comparison of Different Methods for Computing
Lyapunov Exponents", Prog. Theor. Phys. 83(5), 875-893. The state is advanced one step with ``step_fn``, a
``[D, num_exponents]`` deviation matrix is propagated through the Jacobian of ``step_fn`` at the pre-step state and
QR-decomposed, ``log |diag(R)|`` is accumulated, and the deviations are replaced by ``Q``. After ``num_steps`` the sum
is divided by ``num_steps * dt``.

Matches ChaosTools.jl's ``lyapunovspectrum`` (``N = num_steps``, ``Ttr = transient_steps``) to float64 precision on the
Hénon map.

The spectrum is not sorted. QR yields descending exponents whenever the deviation vectors mix across steps, which is
generic for chaotic maps. When the coordinate axes are invariant directions, e.g. ``step_fn(x) = [0.5, 2.0] * x``,
QR never reorders them and the diagonal's order is kept, as in ChaosTools.jl.

Gradient class: exact. Differentiating with respect to anything upstream of ``initial_state`` or ``step_fn``
backpropagates through ``num_steps`` chained applications of ``step_fn``, its Jacobian and a QR decomposition, the
products-of-Jacobians structure whose gradient variance explodes for chaotic systems (Metz et al., "Gradients are Not
All You Need", 2021, https://arxiv.org/abs/2111.05803). Nothing here mitigates it: keep ``num_steps`` modest, clip
gradients in the optimizer, or detach the state in an outer rollout upstream of this call.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import vmap

from torchnlts._shape import _as_int, _real_scalar
from torchnlts.lyapunov.jacobian import _mask_nonfinite_states, step_jacobian


def lyapunov_spectrum(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    num_steps: int,
    *,
    num_exponents: int | None = None,
    dt: float = 1.0,
    transient_steps: int = 0,
) -> torch.Tensor:
    """Lyapunov exponent spectrum of ``step_fn`` along the orbit from ``initial_state``.

    Reference: Benettin et al. (1980), Meccanica 15(1), 9-20; Geist, Parlitz & Lauterborn (1990), Prog. Theor. Phys.
    83(5), 875-893.

    Gradient class: exact (see the module docstring for chaotic ``step_fn``; ``torch.linalg.qr``'s backward also
    needs a deviation matrix of full column rank at every step).

    Args:
        step_fn: One discrete step of the dynamics, or one fixed-step integration of a flow: a differentiable
            ``[D] -> [D]`` callable (see :func:`~torchnlts.lyapunov.step_jacobian`).
        initial_state: Shape ``[D]`` or ``[B, D]``, a float32 or float64 tensor (QR has no half-precision kernel).
        num_steps: Number of steps the growth rates are averaged over (ChaosTools.jl's ``N``). Must be positive.
        num_exponents: Number of exponents, in ``[1, D]``. Default: ``D``.
        dt: Time per step, dividing the log-growth (``1.0`` for a map). Must be positive.
        transient_steps: Steps evolved before the accumulation starts (ChaosTools.jl's ``Ttr``). Must be ``>= 0``.

    Returns:
        Exponents of shape ``[num_exponents]`` or ``[B, num_exponents]``, in ``initial_state``'s dtype, generally in
        descending order (see the module docstring). ``-inf`` for an exponent whose growth rate is exactly 0 at some
        step; ``NaN`` if ``step_fn`` or the QR produces one, including for a rank-deficient deviation matrix, and for
        an ``initial_state`` (or batch row) containing NaN or inf, where ``step_fn`` is evaluated at the zero state
        instead (the other rows and their gradients are unaffected; see :func:`~torchnlts.lyapunov.step_jacobian`).

    Raises:
        ValueError: If ``initial_state`` is not 1-D or 2-D or not float32 or float64, ``num_steps`` is not a
            positive integer, ``num_exponents`` is not an integer in ``[1, D]``, ``dt`` is not a positive real
            number, or ``transient_steps`` is not a non-negative integer.
    """
    if initial_state.dim() not in (1, 2):
        raise ValueError(f"expected initial_state of shape [D] or [B, D], got {tuple(initial_state.shape)}")
    if initial_state.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"initial_state must be a float32 or float64 tensor, got {initial_state.dtype}")
    state_dim = initial_state.shape[-1]
    num_steps = _as_int(num_steps, "num_steps")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    num_exponents = state_dim if num_exponents is None else _as_int(num_exponents, "num_exponents")
    if not (1 <= num_exponents <= state_dim):
        raise ValueError(f"num_exponents must be in [1, {state_dim}], got {num_exponents}")
    dt = _real_scalar(dt, "dt")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    transient_steps = _as_int(transient_steps, "transient_steps")
    if transient_steps < 0:
        raise ValueError(f"transient_steps must be non-negative, got {transient_steps}")

    batched = initial_state.dim() == 2
    advance = vmap(step_fn) if batched else step_fn
    initial_state, nonfinite = _mask_nonfinite_states(initial_state)

    identity = torch.eye(state_dim, dtype=initial_state.dtype, device=initial_state.device)
    deviations = identity[:, :num_exponents]
    if batched:
        deviations = deviations.unsqueeze(0).expand(initial_state.shape[0], -1, -1).clone()

    state = initial_state
    log_growth = torch.zeros(
        deviations.shape[:-2] + (num_exponents,), dtype=initial_state.dtype, device=initial_state.device
    )

    for step in range(transient_steps + num_steps):
        # A masked row stays at zeros with an identity Jacobian: a singular Jacobian there would give its QR and log an
        # infinite gradient that reaches shared parameters as 0 * inf.
        jacobian = torch.where(nonfinite[..., None, None], identity, step_jacobian(step_fn, state))
        deviations = jacobian @ deviations
        q, r = torch.linalg.qr(deviations, mode="reduced")
        if step >= transient_steps:
            log_growth = log_growth + torch.diagonal(r, dim1=-2, dim2=-1).abs().log()
        deviations = q
        state = torch.where(nonfinite[..., None], initial_state, advance(state))

    exponents = log_growth / (num_steps * dt)
    return torch.where(nonfinite[..., None], torch.full_like(exponents, float("nan")), exponents)
