"""TrajectoryKernel: temperature-controlled kernel weights over distances.

Gradient class: smooth.
References: Judd (1992), Physica D 56(2-3), 216-228 (gaussian, a smooth
stand-in for the Grassberger-Procaccia Heaviside step); Sugihara et al.
(2012), Science 338(6106), 496-500 (exponential, CCM's neighbor weight
``u_i = exp(-d_i / d_1)``, recovered by passing ``d_1`` as ``temperature``).

Limitation: distance and temperature both within about one
order of magnitude of float32's representable floor (~1.2e-38) can still
give an infinite temperature gradient (temperature**2 underflows to 0
independent of formulation). float64 is unaffected; upgrade path for
that regime is float64.
"""

from __future__ import annotations

import torch


def trajectory_kernel(
    distance: torch.Tensor, temperature: torch.Tensor | float, kind: str = "gaussian"
) -> torch.Tensor:
    """Kernel weights over a distance matrix.

    Args:
        distance: Non-negative distances, any shape.
        temperature: Bandwidth, scalar or broadcastable against ``distance``.
            Must be positive.
        kind: ``"gaussian"`` or ``"exponential"``.

    Returns:
        Kernel weights, same shape as ``distance``, in ``(0, 1]``.
    """
    if kind not in ("gaussian", "exponential"):
        raise ValueError(f"kind must be 'gaussian' or 'exponential', got {kind!r}")

    # Finite placeholder for inf distances, 0 weight selected after: exp(-inf)'s backward w.r.t. temperature is NaN.
    finite = torch.isfinite(distance)
    safe_distance = torch.where(finite, distance, torch.zeros_like(distance))

    # Floor temperature relative to distance so their ratio stays below ratio_max; the kernel already underflows to 0
    # there, and the division's backward stays finite.
    ratio_max = 1000.0
    floor = safe_distance.abs() / ratio_max
    safe_temperature = torch.clamp(floor, min=temperature)

    if kind == "gaussian":
        ratio = safe_distance / safe_temperature
        weight = torch.exp(-(ratio**2) / 2)
    else:
        weight = torch.exp(-safe_distance / safe_temperature)
    return torch.where(finite, weight, torch.zeros_like(weight))
