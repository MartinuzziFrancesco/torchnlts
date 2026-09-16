"""Thin adapter over dysts reference dynamical systems.

Wraps `dysts <https://github.com/williamgilpin/dysts>`_ output as
PyTorch tensors; no dynamics of its own to validate.
"""

from __future__ import annotations

import dysts.flows as _flows
import dysts.maps as _maps
import torch


def load_system(name: str):
    """Look up a dysts flow (e.g. ``"Lorenz"``) or map (e.g.
    ``"Logistic"``) class by name and instantiate it.
    """
    if hasattr(_flows, name):
        return getattr(_flows, name)()
    if hasattr(_maps, name):
        return getattr(_maps, name)()
    raise ValueError(f"{name!r} is not a known dysts flow or map")


def trajectory(name: str, num_points: int, **kwargs) -> torch.Tensor:
    """Generate a trajectory from a named dysts system as a tensor.

    Args:
        name: See :func:`load_system`.
        num_points: Number of time points, forwarded to ``make_trajectory``.
        **kwargs: Forwarded to ``make_trajectory`` (e.g. ``resample``,
            ``pts_per_period``, ``init_cond`` as a numpy array).

    Returns:
        Tensor of shape ``[num_points]`` (scalar maps) or
        ``[num_points, D]`` (multivariate), dtype inferred from dysts'
        own output.

    Raises:
        ValueError: If ``return_times=True`` is passed -- that changes
            the return value to a ``(times, values)`` tuple; call
            ``load_system(name).make_trajectory(..., return_times=True)``
            directly for timestamps.
    """
    if kwargs.get("return_times"):
        raise ValueError(
            "trajectory() always returns just the state array; call "
            "load_system(name).make_trajectory(..., return_times=True) "
            "directly for timestamps too"
        )
    system = load_system(name)
    raw = system.make_trajectory(num_points, **kwargs)
    result = torch.as_tensor(raw)
    # dysts squeezes the time axis away for num_points=1; restore it so
    # the shape always matches [num_points] / [num_points, D].
    if result.dim() == 0 or (result.dim() == 1 and result.shape[0] != num_points):
        result = result.unsqueeze(0)
    return result
