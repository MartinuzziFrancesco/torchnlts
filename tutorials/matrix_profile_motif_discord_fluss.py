"""Matrix profile motifs, discords and FLUSS segmentation.

Yeh et al. (2016), "Matrix Profile I" (ICDM): the matrix profile gives each
length-m subsequence's z-normalized distance to its nearest non-trivial neighbor.
Its minimum is the top motif pair, its maximum the top discord. Gharghabi et al.
(2017), "Matrix Profile VIII" (ICDM): FLUSS finds regime changes as the minimum of
the corrected arc curve built from the profile index.

Part 1: a noisy series with the same shape planted at two places has that pair as
its top motif; a sine with one stretch replaced by a square wave has that stretch
as its top discord. Part 2: a Lorenz series followed by a Rossler series (dysts)
has a regime boundary found at the splice.
"""

import math

import torch
from dysts.flows import Lorenz, Rossler

from torchnlts.matrixprofile import corrected_arc_curve, discords, matrix_profile, motifs, regime_boundaries

generator = torch.Generator().manual_seed(0)

# Part 1a: planted motif.
window = 30
series = torch.randn(600, generator=generator, dtype=torch.float64)
pattern = 4 * torch.sin(torch.linspace(0, 2 * math.pi, window, dtype=torch.float64))
series[100 : 100 + window] += pattern
series[420 : 420 + window] += pattern
profile, index = matrix_profile(series, window)
distances, pairs = motifs(profile, index, window, num_picks=2)
print("Planted motif at starts 100 and 420")
print(f"  top motif pair {sorted(pairs[0].tolist())}, distance {distances[0]:.3f}")
print(f"  second pair    {sorted(pairs[1].tolist())}, distance {distances[1]:.3f}")
assert all(abs(start - planted) <= 10 for start, planted in zip(sorted(pairs[0].tolist()), (100, 420)))
assert distances[0] < distances[1]

# Part 1b: planted discord.
time = torch.arange(600, dtype=torch.float64)
series = torch.sin(2 * math.pi * time / 30) + 0.05 * torch.randn(600, generator=generator, dtype=torch.float64)
series[300:330] = torch.sign(torch.sin(2 * math.pi * time[300:330] / 11))
profile, _ = matrix_profile(series, window)
values, starts = discords(profile, window, num_picks=2)
print("\nSquare-wave stretch replacing the sine at samples 300-329")
print(f"  top discord starts at {starts[0].item()} (profile {values[0]:.3f}); profile median {profile.median():.3f}")
assert 270 <= starts[0].item() <= 330

# Part 2: regime change from Lorenz to Rossler.
switch = 1500
lorenz = torch.as_tensor(Lorenz().make_trajectory(switch))[:, 0]
rossler = torch.as_tensor(Rossler().make_trajectory(switch))[:, 0]
standardize = lambda values: (values - values.mean()) / values.std()  # noqa: E731
series = torch.cat([standardize(lorenz), standardize(rossler)]).to(torch.float64)
window = 60
_, index = matrix_profile(series, window)
curve = corrected_arc_curve(index, window)
boundary = regime_boundaries(curve, window, num_regimes=2).item()
print(f"\nLorenz (x) for {switch} samples, then Rossler (x); window {window}")
print(f"  FLUSS boundary at {boundary} (true switch {switch}), CAC there {curve[boundary]:.3f}")
print(f"  CAC away from the switch (median over interior) {curve[window * 5 : -window * 5].median():.3f}")
assert abs(boundary - switch) <= 2 * window

print("\nThe planted pair is the top motif, the square wave the top discord, and the")
print("boundary lies within two windows of the splice. One series each, not a benchmark.")
