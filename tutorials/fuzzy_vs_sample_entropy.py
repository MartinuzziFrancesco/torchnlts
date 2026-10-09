"""Reproduces the core motivating claim of Chen, Wang, Xie & Yu (2007),
"Characterization of Surface EMG Signal Based on Fuzzy Entropy" (IEEE
Trans. Neural Syst. Rehabil. Eng. 15(2), 266-272): sample entropy's
hard threshold makes it fragile at small radius/short trajectories --
its long-template match count A(r) can hit exactly zero, making
SampEn undefined (`-log(0/B)` = `inf`, or `-log(0/0)` = `NaN` once
even B(r) collapses too) -- while fuzzy entropy's continuous
membership function `exp(-d^n/r)` degrades far more gracefully across
the same sweep: mathematically it's never exactly zero for a finite
distance, though in floating point it can still underflow to exactly
0.0 at a small enough radius (this run's sweep stops before that
point; it isn't a difference in kind from SampEn's own eventual
breakdown, just a much later one).

This uses i.i.d. uniform noise, the same class of signal the paper's
own Fig. 2 sweeps radius over, and a short trajectory (N=20) to make
the breakdown happen at a radius large enough to also show FuzzyEn's
values varying smoothly on both sides of where SampEn fails.
"""

import torch

from torchnlts.entropy import fuzzy_entropy, sample_entropy

torch.manual_seed(0)
trajectory = torch.rand(20, dtype=torch.float64)
radii = torch.tensor([0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=torch.float64)

sampen = sample_entropy(trajectory, radius=radii, embedding_dim=2)
fuzzyen = fuzzy_entropy(trajectory, radius=radii, embedding_dim=2)

print(f"{'radius':>8}  {'SampEn':>10}  {'FuzzyEn':>10}")
for radius, sampen_value, fuzzyen_value in zip(radii, sampen, fuzzyen):
    print(f"{radius.item():8.3f}  {sampen_value.item():10.4f}  {fuzzyen_value.item():10.4f}")

assert torch.isinf(sampen[-4:]).all(), "SampEn should diverge to inf at small enough radius (A(r)=0)"
assert torch.isfinite(fuzzyen).all(), "FuzzyEn should stay finite across the whole sweep"
assert (fuzzyen.diff() > 0).all(), "FuzzyEn should increase monotonically as radius shrinks"

print("\nAll checks passed: SampEn diverges to inf once the radius shrinks past the point where no")
print("(embedding_dim+1)-length template pair matches, while FuzzyEn keeps producing finite, smoothly")
print("increasing values across the same sweep, reproducing the paper's stated motivation for")
print("replacing the hard threshold with a continuous membership function.")
