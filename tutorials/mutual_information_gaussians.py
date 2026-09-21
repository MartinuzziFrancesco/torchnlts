"""Reproduces Kraskov, Stogbauer & Grassberger (2004): their KSG1
k-nearest-neighbor mutual information estimator, checked against the
standard benchmark where the true value is known analytically -- a pair of
correlated Gaussians, whose mutual information is exactly
``-0.5 * log(1 - rho**2)``.
"""

import math

import torch

from torchnlts.entropy.mutual_information import mutual_information

torch.manual_seed(0)
num_points = 5000
rho = 0.8

z1 = torch.randn(num_points, dtype=torch.float64)
z2 = torch.randn(num_points, dtype=torch.float64)
x = z1
y = rho * z1 + math.sqrt(1 - rho**2) * z2

estimated = mutual_information(x, y, num_neighbors=5)
true_value = -0.5 * math.log(1 - rho**2)

print(f"correlation rho = {rho}")
print(f"estimated mutual information: {estimated.item():.4f}")
print(f"true mutual information:      {true_value:.4f}")
