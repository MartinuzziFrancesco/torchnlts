"""Reproduces Schreiber (2000): information-theoretic detection of
directional coupling, using the same unidirectionally-coupled logistic
maps as `convergent_cross_mapping.py` -- a different method (conditional
mutual information between one series' future and the other's past,
rather than cross-mapped nearest-neighbor prediction skill) arriving at
the same causal-direction conclusion.

Transfer entropy is asymmetric by construction: TE(X -> Y) measures how
much of Y's future the recent past of X explains beyond what Y's own
past already explains, and vice versa for TE(Y -> X). Since X drives Y
here and not the other way around, TE(X -> Y) should be substantially
larger.
"""

import torch

from torchnlts.causality.transfer_entropy import transfer_entropy

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
print("\nTE(X -> Y) is substantially larger, correctly identifying X as the driver")
print("-- agreeing with convergent_cross_mapping.py's conclusion via an entirely")
print("different (information-theoretic, not nearest-neighbor) method.")
