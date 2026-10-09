"""Reproduces Kraskov, Stogbauer & Grassberger (2004): their KSG1 and KSG2
k-nearest-neighbor mutual information estimators, checked against the
standard benchmark where the true value is known analytically -- a pair of
correlated Gaussians, whose mutual information is exactly
``-0.5 * log(1 - rho**2)``.
"""

import math

import torch

from torchnlts.entropy import mutual_information

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

# KSG2 (eq. 9) uses the marginal extents of the k joint neighbors; compare
# both estimators against the analytic value over a range of correlations.
print("\n rho   analytic   KSG1     KSG2")
for rho_i in (0.0, 0.5, 0.8, 0.95):
    y_i = rho_i * z1 + math.sqrt(1 - rho_i**2) * z2
    analytic = -0.5 * math.log(1 - rho_i**2)
    ksg1 = mutual_information(x, y_i, num_neighbors=5, estimator="ksg1").item()
    ksg2 = mutual_information(x, y_i, num_neighbors=5, estimator="ksg2").item()
    print(f"{rho_i:5.2f}  {analytic:8.4f}  {ksg1:7.4f}  {ksg2:7.4f}")
