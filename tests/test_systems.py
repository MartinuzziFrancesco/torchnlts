"""Tests for torchnlts.systems (the dysts adapter).

Per systems/__init__.py's docstring: this module is data conversion, not
a numerical algorithm, so there's no paper/Julia dual ground truth to
validate against here (torchnlts wraps dysts rather than re-deriving its
dynamics). These tests instead check the wrapping is correct.
"""

import dysts.flows as flows
import dysts.maps as maps
import numpy as np
import torch

from torchnlts.systems import load_system, trajectory


def test_trajectory_flow_shape_and_exact_passthrough():
    torch.manual_seed(0)
    t = trajectory("Lorenz", 50)
    assert t.shape == (50, 3)
    assert t.dtype == torch.float64
    # Exact passthrough: same values dysts itself produces for the same
    # (deterministic) default initial condition.
    raw = flows.Lorenz().make_trajectory(50)
    assert np.allclose(t.numpy(), raw)


def test_trajectory_map_shape():
    t = trajectory("Logistic", 30)
    assert t.shape == (30,)
    assert t.dtype == torch.float64


def test_trajectory_single_point_preserves_time_axis():
    # dysts squeezes away the time axis for n=1; the wrapper must restore
    # it so the shape contract ([n] / [n, D]) holds for every n.
    scalar = trajectory("Logistic", 1)
    assert scalar.shape == (1,)
    multivariate = trajectory("Henon", 1)
    assert multivariate.shape == (1, 2)


def test_trajectory_unknown_name_raises():
    try:
        trajectory("NotARealDynamicalSystem", 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown system name")


def test_trajectory_rejects_return_times():
    # return_times=True changes dysts' return value to a (times, values)
    # tuple, which doesn't fit trajectory()'s single-tensor contract; it
    # must be rejected explicitly rather than mis-converted.
    for name in ("Lorenz", "Logistic"):
        try:
            trajectory(name, 10, return_times=True)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for return_times=True with {name}")


def test_load_system_returns_matching_class():
    assert isinstance(load_system("Lorenz"), flows.Lorenz)
    assert isinstance(load_system("Henon"), maps.Henon)


def test_trajectory_forwards_kwargs():
    # `init_cond` overrides the initial condition; the wrapper must
    # forward it rather than silently ignoring extra kwargs.
    custom_ic = np.array([1.0, 1.0, 1.0])
    t = trajectory("Lorenz", 20, init_cond=custom_ic)
    raw_traj = flows.Lorenz().make_trajectory(20, init_cond=custom_ic)
    assert np.allclose(t.numpy(), raw_traj)
