"""Reproduces Sugihara, May, Ye, Hsieh, Deyle, Fogarty & Munch (2012):
detecting which of two coupled variables drives the other, using only
their time series -- no intervention needed.

X drives Y unidirectionally here (Y's update depends on X; X's own
update never depends on Y). Per convergent cross mapping's own logic --
counterintuitive on first encounter, see `torchnlts.causality.ccm`'s
module docstring -- embedding the *driven* variable and cross-mapping
to estimate the *driver* should show much higher skill than the reverse,
since the driven variable's own dynamics have the driver's signature
left in them.
"""

import torch

from torchnlts.causality.ccm import ccm_skill, cross_map_estimate

torch.manual_seed(0)
num_points = 1000
x = torch.zeros(num_points, dtype=torch.float64)
y = torch.zeros(num_points, dtype=torch.float64)
x[0], y[0] = 0.4, 0.3
rate_x, rate_y, coupling = 3.8, 3.5, 0.3
for t in range(num_points - 1):
    x[t + 1] = x[t] * rate_x * (1 - x[t])
    y[t + 1] = (y[t] * rate_y * (1 - y[t]) + coupling * x[t]) % 1.0

embedding_dim, delay, theiler_window = 3, 1, 5
offset = (embedding_dim - 1) * delay

estimated_x = cross_map_estimate(y, x, embedding_dim, delay, theiler_window=theiler_window)
skill_recovering_x_from_y = ccm_skill(estimated_x, x[offset : offset + estimated_x.shape[0]].unsqueeze(-1))

estimated_y = cross_map_estimate(x, y, embedding_dim, delay, theiler_window=theiler_window)
skill_recovering_y_from_x = ccm_skill(estimated_y, y[offset : offset + estimated_y.shape[0]].unsqueeze(-1))

print(f"skill recovering X from Y's manifold (tests 'X drives Y', true):  {skill_recovering_x_from_y.item():.3f}")
print(f"skill recovering Y from X's manifold (tests 'Y drives X', false): {skill_recovering_y_from_x.item():.3f}")
print("\nThe much higher skill in the first direction correctly identifies X as")
print("the driver.")
