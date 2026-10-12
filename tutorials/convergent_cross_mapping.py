"""Illustrates the CCM methodology of Sugihara, May, Ye, Hsieh, Deyle, Fogarty & Munch (2012):
cross-map skill versus library size for a synthetic unidirectionally
coupled pair (not the paper's exact map or parameter experiment).

X drives Y unidirectionally here (Y's update depends on X; X's own
update never depends on Y). Per convergent cross mapping's own logic --
counterintuitive on first encounter, see `torchnlts.causality.ccm`'s
module docstring -- embedding the *driven* variable and cross-mapping
to estimate the *driver* should improve as the library grows,
since the driven variable's own dynamics have the driver's signature
left in them. Directional asymmetry alone is not the convergence test.
"""

import torch

from torchnlts.causality import ccm_skill, cross_map_estimate

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

print(f"{'library':>7}  {'recover X from Y':>18}  {'recover Y from X':>18}")
# The API restricts neighbors to a leading prefix; all query points stay fixed.
for library_size in (50, 100, 200, 400, 800, num_points - offset):
    estimated_x = cross_map_estimate(
        y, x, embedding_dim, delay=delay, library_size=library_size, theiler_window=theiler_window
    )
    skill_recovering_x_from_y = ccm_skill(estimated_x, x[offset:])
    estimated_y = cross_map_estimate(
        x, y, embedding_dim, delay=delay, library_size=library_size, theiler_window=theiler_window
    )
    skill_recovering_y_from_x = ccm_skill(estimated_y, y[offset:])
    print(f"{library_size:>7}  {skill_recovering_x_from_y.item():>18.3f}  {skill_recovering_y_from_x.item():>18.3f}")

print("\nIncreasing skill recovering X from Y is consistent with the known X -> Y coupling.")
print("This single prefix-library sweep is illustrative, not a general proof of causation;")
print("shared forcing and synchronization can confound CCM interpretations.")
