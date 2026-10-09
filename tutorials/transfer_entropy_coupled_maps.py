"""Illustrates Schreiber (2000)'s transfer-entropy concept:
directional coupling, using the same unidirectionally-coupled logistic
maps as `convergent_cross_mapping.py` -- a different method (conditional
mutual information between one series' future and the other's past,
rather than cross-mapped nearest-neighbor prediction skill) checking the
known direction of predictive information. These are synthetic maps,
not an exact reproduction of Schreiber's numerical experiment. The API
uses the later Frenzel-Pompe (2007) nearest-neighbor CMI estimator.

Transfer entropy is asymmetric by construction: TE(X -> Y) measures how
much of Y's future the recent past of X explains beyond what Y's own
past already explains, and vice versa for TE(Y -> X). Since X drives Y
here and not the other way around, TE(X -> Y) should be substantially
larger.
"""

import torch

from torchnlts.causality import transfer_entropy

torch.manual_seed(0)
num_points = 2000
x = torch.zeros(num_points, dtype=torch.float64)
y = torch.zeros(num_points, dtype=torch.float64)
x[0], y[0] = 0.4, 0.3
rate_x, rate_y, coupling = 3.8, 3.5, 0.3
for t in range(num_points - 1):
    x[t + 1] = x[t] * rate_x * (1 - x[t])
    y[t + 1] = (y[t] * rate_y * (1 - y[t]) + coupling * x[t]) % 1.0

te_x_to_y = transfer_entropy(x, y, embedding_dim=2, delay=1, theiler_window=5)
te_y_to_x = transfer_entropy(y, x, embedding_dim=2, delay=1, theiler_window=5)

print(f"TE(X -> Y) (true direction):  {te_x_to_y.item():.3f}")
print(f"TE(Y -> X) (false direction): {te_y_to_x.item():.3f}")
print("\nLarger TE(X -> Y) is consistent with the known coupling in these maps.")
print("TE measures additional predictive information, not causation in general.")
print("Both this estimator and CCM use nearest neighbors, but estimate different quantities.")
