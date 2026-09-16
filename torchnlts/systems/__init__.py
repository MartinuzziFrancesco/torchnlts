"""Thin adapter over dysts reference dynamical systems.

torchnlts does not hand-write reference
dynamical systems (Lorenz, Rossler, the logistic map, Henon, Mackey-Glass,
etc.) -- it depends on `dysts` (https://github.com/williamgilpin/dysts)
for those and just wraps its output as a PyTorch tensor. This module has
no numerical algorithm of its own to validate against a paper equation or
a Julia sibling package the way the core primitives do: it is data
conversion, not computation, so its tests (tests/test_systems.py) check
the wrapping (shape, dtype, exact value passthrough, unknown-name
handling) rather than re-deriving any system's dynamics.
"""

from __future__ import annotations

import dysts.flows as _flows
import dysts.maps as _maps
import torch


def load_system(name: str):
    """Look up a dysts flow or map class by name and instantiate it.

    Args:
        name: Class name as it appears in ``dysts.flows`` (continuous
            systems, e.g. ``"Lorenz"``, ``"Rossler"``) or ``dysts.maps``
            (discrete systems, e.g. ``"Logistic"``, ``"Henon"``).

    Returns:
        The instantiated dysts system object (its own ``make_trajectory``
        etc. are unchanged; use :func:`trajectory` for a tensor output).
    """
    if hasattr(_flows, name):
        return getattr(_flows, name)()
    if hasattr(_maps, name):
        return getattr(_maps, name)()
    raise ValueError(f"{name!r} is not a known dysts flow or map")


def trajectory(name: str, n: int, **kwargs) -> torch.Tensor:
    """Generate a trajectory from a named dysts system as a tensor.

    Args:
        name: See :func:`load_system`.
        n: Number of time points, forwarded to dysts' ``make_trajectory``.
        **kwargs: Forwarded to ``make_trajectory`` (e.g. ``resample``,
            ``pts_per_period``, ``init_cond`` for a custom initial
            condition -- as a numpy array, not a plain list).

    Returns:
        Float64 tensor of shape ``[n]`` (scalar maps) or ``[n, D]``
        (multivariate flows/maps), the same values dysts produced.

    Raises:
        ValueError: If ``return_times=True`` is passed. That changes
            dysts' return value to a ``(times, values)`` tuple, which
            doesn't fit this function's single-tensor contract; call
            ``load_system(name).make_trajectory(..., return_times=True)``
            directly if timestamps are needed too.
    """
    if kwargs.get("return_times"):
        raise ValueError(
            "trajectory() always returns just the state array; call "
            "load_system(name).make_trajectory(..., return_times=True) "
            "directly for timestamps too"
        )
    system = load_system(name)
    traj = system.make_trajectory(n, **kwargs)
    t = torch.as_tensor(traj, dtype=torch.float64)
    # dysts squeezes away the time axis for n=1: a scalar map returns a
    # bare 0-d value instead of shape [1], and a multivariate map/flow
    # returns shape [D] instead of [1, D]. Restore the leading time axis
    # so the output always matches this function's documented [n] / [n, D]
    # contract, regardless of n.
    if t.dim() == 0 or (t.dim() == 1 and t.shape[0] != n):
        t = t.unsqueeze(0)
    return t
