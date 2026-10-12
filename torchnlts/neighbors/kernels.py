"""Kernel weights over distances: a smooth stand-in for a hard neighbor indicator.

References: Judd (1992), Physica D 56(2-3), 216-228 (gaussian, a smooth stand-in for the Grassberger-Procaccia
Heaviside step); Sugihara et al. (2012), Science 338(6106), 496-500 (exponential, CCM's neighbor weight
``u_i = exp(-d_i / d_1)``, recovered by passing ``d_1`` as ``temperature``).

Gradient class: smooth.

Limitation: when distance and temperature are both within about one order of magnitude of float32's smallest normal
number (~1.2e-38), the temperature gradient can be infinite (``temperature**2`` underflows); use float64.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _real_scalar


def _validate_temperature(temperature) -> None:
    if not torch.is_tensor(temperature):
        positive = _real_scalar(temperature, "temperature") > 0
    elif temperature.is_complex():
        raise ValueError(f"temperature must be real, got {temperature!r}")
    else:
        # A tensor wrapped by torch.vmap or torch.func has no concrete value to test.
        positive = torch._C._functorch.is_functorch_wrapped_tensor(temperature) or bool((temperature > 0).all())
    if not positive:
        raise ValueError(f"temperature must be > 0 and not NaN, got {temperature!r}")


def trajectory_kernel(
    distance: torch.Tensor, temperature: torch.Tensor | float, *, kind: str = "gaussian"
) -> torch.Tensor:
    """Kernel weights over a distance matrix.

    Reference: Judd (1992), Physica D 56(2-3), 216-228 (gaussian); Sugihara et al. (2012), Science 338(6106), 496-500
    (exponential).

    Gradient class: smooth.

    Args:
        distance: Non-negative distances, any shape. NaN and infinite entries get weight 0.
        temperature: Bandwidth, scalar or broadcastable against ``distance``. Every entry must be > 0 and not NaN
            (not checked for a tensor wrapped by ``torch.vmap`` or ``torch.func``).
        kind: ``"gaussian"``, ``exp(-d**2 / (2 * temperature**2))``, or ``"exponential"``,
            ``exp(-d / temperature)``.

    Returns:
        Kernel weights in ``[0, 1]``, shaped like ``distance``. A weight below the dtype's range is exactly 0.

    Raises:
        ValueError: If ``kind`` is unknown, or ``temperature`` is complex, not a number or tensor, <= 0 or NaN.
    """
    if kind not in ("gaussian", "exponential"):
        raise ValueError(f"kind must be 'gaussian' or 'exponential', got {kind!r}")
    _validate_temperature(temperature)

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
