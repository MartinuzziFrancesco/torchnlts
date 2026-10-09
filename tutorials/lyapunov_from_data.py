"""Reproduces Rosenstein, Collins & De Luca (1993) and Kantz (1994):
estimating the largest Lyapunov exponent from an observed scalar time
series alone, no equations of motion needed.

The r=4 logistic map is the standard analytic benchmark for this kind
of estimator: its largest Lyapunov exponent is exactly ln(2), derivable
directly from the map's own definition, independent of either
algorithm below.
"""

import math

import numpy as np
import torch
from dysts.maps import Logistic

from torchnlts.lyapunov import kantz_divergence, rosenstein_divergence, rosenstein_lyapunov

system = Logistic(parameters={"r": 4.0})
trajectory = torch.as_tensor(system.make_trajectory(2000, init_cond=np.array([0.4])), dtype=torch.float64)

k_values = torch.arange(1, 12, dtype=torch.float64)
fit_range = slice(0, 6)  # the visually linear region of E(k)/S(k) before saturation

rosenstein_curve = rosenstein_divergence(trajectory, k_values, embedding_dim=2, delay=1, theiler_window=10)
rosenstein_estimate = rosenstein_lyapunov(k_values[fit_range], rosenstein_curve[fit_range])

kantz_curve = kantz_divergence(trajectory, k_values, radius=0.01, embedding_dim=2, delay=1, theiler_window=10)
kantz_estimate = rosenstein_lyapunov(k_values[fit_range], kantz_curve[fit_range])  # same OLS-slope step, either curve

print(f"Rosenstein estimate: {rosenstein_estimate.item():.4f}")
print(f"Kantz estimate:      {kantz_estimate.item():.4f}")
print(f"true ln(2):           {math.log(2):.4f}")
