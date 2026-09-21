"""Reproduces Grassberger & Procaccia (1983): the correlation dimension
of a strange attractor, estimated purely from a trajectory sampled on
it -- no knowledge of the governing equations needed.

The Lorenz attractor's correlation dimension is a well-known
non-integer value, commonly cited around 2.05 -- neither a 2D surface
nor a 3D volume, the fractal signature of a strange attractor.
"""

import torch

from torchnlts.dimension.correlation import correlation_dimension, correlation_sum
from torchnlts.systems import trajectory

state = trajectory("Lorenz", 5000).to(torch.float64)
state = (state - state.mean(dim=0)) / state.std(dim=0)  # normalize each coordinate's scale

radius = torch.logspace(-1.5, -0.3, 12, dtype=torch.float64)
correlation_sums = correlation_sum(state, radius, theiler_window=10)
dimension = correlation_dimension(radius, correlation_sums)

print("radius sweep:", [f"{r:.3f}" for r in radius.tolist()])
print("C(r):        ", [f"{c:.4f}" for c in correlation_sums.tolist()])
print(f"\nestimated correlation dimension: {dimension.item():.3f}")
print("(literature value for the Lorenz attractor is commonly cited around 2.05)")
