"""Reproduces Grassberger & Procaccia (1983): the correlation dimension
of a strange attractor, estimated purely from a trajectory sampled on
it -- no knowledge of the governing equations needed.

The Lorenz attractor's correlation dimension is a well-known
non-integer value, commonly cited around 2.05 -- neither a 2D surface
nor a 3D volume, the fractal signature of a strange attractor.
"""

import torch

from torchnlts.dimension import correlation_dimension, correlation_sum
from torchnlts.scaling import largest_linear_region, scaling_slope
from torchnlts.systems import trajectory

state = trajectory("Lorenz", 5000).to(torch.float64)
state = (state - state.mean(dim=0)) / state.std(dim=0)  # normalize each coordinate's scale

radius = torch.logspace(-1.8, 0.2, 20, dtype=torch.float64)  # wide sweep: includes the noisy and saturated ends
correlation_sums = correlation_sum(state, radius, theiler_window=10)
dimension = correlation_dimension(radius, correlation_sums)  # fits every radius
hand_picked = correlation_dimension(radius[4:15], correlation_sums[4:15])

# Automatic scaling region: the largest run of radii with a stable log-log slope.
log_radius, log_correlation = radius.log(), correlation_sums.log()
start, stop = largest_linear_region(log_radius, log_correlation)
detected = scaling_slope(log_radius, log_correlation)

print("radius sweep:", [f"{r:.3f}" for r in radius.tolist()])
print("C(r):        ", [f"{c:.4f}" for c in correlation_sums.tolist()])
print(f"\ncorrelation dimension, all radii:           {dimension.item():.3f}")
print(f"correlation dimension, hand-picked [4:15]:  {hand_picked.item():.3f}")
print(f"correlation dimension, detected [{start.item()}:{stop.item()}]:    {detected.item():.3f}")
print("(literature value for the Lorenz attractor is commonly cited around 2.05)")
