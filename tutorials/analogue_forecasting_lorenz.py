"""Lorenz (1969) method of analogues on the Lorenz system's x-component.

Reference: Lorenz, E.N. (1969), "Atmospheric predictability as revealed by naturally occurring analogues",
J. Atmos. Sci. 26(4), 636-646.

To forecast from time t, find the stretches of the past that look most like the last CONTEXT samples and read
off what followed them. `analogues` ranks earlier windows by z-normalized Euclidean distance (only windows
whose continuation also ends before t are eligible), and returns each continuation rescaled to the context's
level and amplitude. The forecast is the continuation of the nearest analogue (k=1) or the mean over the k=5
nearest. Persistence repeats the last observed value.

Error is the RMSE over test origins, as a fraction of the series' standard deviation. It grows with the
horizon because nearby trajectories of the chaotic Lorenz system separate exponentially, and analogues beat
persistence at every horizon here, by the widest margin at short horizons.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.prediction import analogues

CONTEXT = 20
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
MAX_HORIZON = max(HORIZONS)
torch.manual_seed(0)

series = torch.as_tensor(Lorenz().make_trajectory(3000))[:, 0].to(torch.float64)
spread = float(series.std())
# The first half is history only; forecasts are made from origins in the second half.
origins = torch.arange(1500, series.shape[0] - MAX_HORIZON + 1, 4)

# One call at the longest horizon; shorter horizons use the first h steps of each continuation. Eligibility
# depends on the horizon (s + CONTEXT + horizon <= t), so this fixes the candidate set for all of them.
distances, starts, continuations = analogues(series, origins, CONTEXT, MAX_HORIZON, num_analogues=5)
print(f"analogues: distances {tuple(distances.shape)}, continuations {tuple(continuations.shape)}")
print(f"all analogues found: {bool((starts >= 0).all())}; mean nearest distance {float(distances[:, 0].mean()):.3f}")

truth = torch.stack([series[int(origin) : int(origin) + MAX_HORIZON] for origin in origins])
last = torch.stack([series[int(origin) - 1] for origin in origins])

print(f"\nRMSE / std(series) over {len(origins)} origins, context {CONTEXT}")
print(f"{'horizon':>8} {'persistence':>12} {'k=1':>8} {'k=5':>8}")
rows = []
for horizon in HORIZONS:
    target = truth[:, horizon - 1]
    persistence = (last - target).pow(2).mean().sqrt() / spread
    nearest = (continuations[:, 0, horizon - 1] - target).pow(2).mean().sqrt() / spread
    mean_five = (continuations[:, :, horizon - 1].mean(dim=1) - target).pow(2).mean().sqrt() / spread
    rows.append((float(persistence), float(nearest), float(mean_five)))
    print(f"{horizon:>8} {rows[-1][0]:>12.3f} {rows[-1][1]:>8.3f} {rows[-1][2]:>8.3f}")

assert rows[-1][1] > rows[0][1] and rows[-1][2] > rows[0][2], "analogue error should grow with horizon"
assert rows[0][1] < rows[0][0] and rows[0][2] < rows[0][0], "analogues should beat persistence at horizon 1"
print("\nanalogue error grows with horizon; analogues beat persistence at horizon 1")
