"""Reproduces the motivation of Fadlallah, Chen, Keil & Principe (2013),
"Weighted-permutation entropy: A complexity measure for time series
incorporating amplitude information" (Phys. Rev. E 87, 022911), and of
Azami & Escudero (2016), "Amplitude-aware permutation entropy" (Comput.
Methods Programs Biomed. 128, 40-51): ordinary permutation entropy sees only
the rank order inside each window, so a large spike and a tiny fluctuation
with the same ordering count the same.

White noise with sparse large-amplitude bursts has almost the same ordinal
statistics as the noise alone, so permutation entropy barely changes. The
variance-weighted and amplitude-aware entropies give the high-variance
windows around each burst a large weight, so the burst-dominated pattern
distribution is visibly less uniform and the entropies drop.
"""

import torch

from torchnlts.entropy import permutation_entropy

torch.manual_seed(0)
length, embedding_dim = 2000, 4
noise = torch.randn(length, dtype=torch.float64)
bursty = noise.clone()
burst_positions = torch.arange(100, length - 10, 200)
for offset, sign in enumerate([1.0, -1.0, 1.0, -1.0]):  # each burst: a rising ramp of large values
    bursty[burst_positions + offset] += 40.0 * (offset + 1) * sign

print(f"{'measure':<22}{'noise':>10}{'with bursts':>14}")
results = {}
for name, kwargs in [
    ("permutation", {}),
    ("weighted (variance)", {"weighting": "variance"}),
    ("amplitude-aware", {"weighting": "amplitude"}),
]:
    pair = [permutation_entropy(signal, embedding_dim, normalize=True, **kwargs).item() for signal in (noise, bursty)]
    results[name] = pair
    print(f"{name:<22}{pair[0]:>10.4f}{pair[1]:>14.4f}")

plain_change = abs(results["permutation"][1] - results["permutation"][0])
assert plain_change < 0.03, "ordinary permutation entropy should barely register the bursts"
for name in ("weighted (variance)", "amplitude-aware"):
    drop = results[name][0] - results[name][1]
    assert drop > 0.1 and drop > 5 * plain_change, f"{name} should drop clearly when bursts are added"

print("\nAll checks passed: permutation entropy changes by less than 0.03 when sparse large bursts are added,")
print("while the variance-weighted and amplitude-aware entropies drop by more than 0.1 because the bursts")
print("dominate the weighted pattern distribution.")
