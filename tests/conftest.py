"""Shared test helpers.

Not a torchnlts module -- pytest loads this file automatically, but tests
explicitly import the ordinary helper functions defined here.
"""

import numpy as np
import torch


def logistic_map_trajectory(num_points: int, x0: float = 0.4) -> torch.Tensor:
    """The r=4 logistic map, whose largest Lyapunov exponent is exactly
    ln(2) -- the standard analytic benchmark for this kind of estimator. Via
    ``dysts.maps.Logistic`` (not hand-rolled: reference dynamical systems
    come from dysts; torchnlts.systems' own thin wrapper doesn't expose a
    custom ``r``/initial condition, so this imports ``dysts`` directly
    rather than reimplementing the map's iteration).
    """
    import dysts.maps as maps

    system = maps.Logistic(parameters={"r": 4.0})
    trajectory = system.make_trajectory(num_points, init_cond=np.array([x0]))
    return torch.as_tensor(trajectory, dtype=torch.float64)
