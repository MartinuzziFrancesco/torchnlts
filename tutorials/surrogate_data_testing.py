"""Reproduces the surrogate data testing methodology of Theiler, Eubank,
Longtin, Galdrikian & Farmer (1992), using Schreiber & Schmitz (1996)'s
IAAFT surrogates (the most refined of this library's three surrogate
methods -- see `torchnlts.surrogate.phase`/`aaft` for the simpler
methods IAAFT iteratively improves on).

IAAFT preserves the amplitude distribution exactly and the power spectrum
approximately. Its null is a stationary Gaussian linear process observed
through a static monotonic transformation, not just a Gaussian process.
Amplitude distribution is not itself a linear temporal property, and
IAAFT does not guarantee removal of every nonlinear dependence.

Use a preselected lower-tail permutation-entropy statistic: unusually low
entropy is the alternative. Report its Monte Carlo rank p-value, whose
minimum with 20 surrogates is 1/21, rather than assuming Gaussian tails
from a z-score. This is an approximate surrogate test: spectral mismatch,
nonstationarity, and finite-record effects can also cause rejection;
rejection alone does not establish deterministic chaos.
"""

import torch

from torchnlts.entropy.permutation import permutation_entropy
from torchnlts.surrogate.iaaft import iaaft_surrogate
from torchnlts.systems import trajectory

torch.manual_seed(0)
signal = trajectory("Lorenz", 1000)[:, 0]
signal = (signal - signal.mean()) / signal.std()

real_statistic = permutation_entropy(signal, embedding_dim=4, delay=1, normalize=True)

num_surrogates = 20
surrogate_statistics = []
spectral_errors = []
target_magnitude = torch.fft.rfft(signal).abs()
for seed in range(num_surrogates):
    surrogate = iaaft_surrogate(signal, max_iterations=50, generator=torch.Generator().manual_seed(seed))
    surrogate_statistics.append(permutation_entropy(surrogate, embedding_dim=4, delay=1, normalize=True))
    spectral_errors.append((torch.fft.rfft(surrogate).abs() - target_magnitude).norm() / target_magnitude.norm())
surrogate_statistics = torch.stack(surrogate_statistics)
spectral_errors = torch.stack(spectral_errors)

# Count ties in the rejection tail conservatively; include the observed sample.
p_value = (1 + (surrogate_statistics <= real_statistic).sum().item()) / (num_surrogates + 1)

print(f"real signal permutation entropy:        {real_statistic.item():.4f}")
print(f"surrogate permutation entropy (mean/std): {surrogate_statistics.mean().item():.4f} / {surrogate_statistics.std().item():.4f}")
print(f"lower-tail rank p-value: {p_value:.4f} (minimum possible: {1 / (num_surrogates + 1):.4f})")
print(f"relative Fourier-magnitude error (min/max): {spectral_errors.min().item():.4f} / {spectral_errors.max().item():.4f}")
print("\nNominal 5% rank-test decision:", "reject" if p_value <= 0.05 else "do not reject")
print("Interpret against the stationary, statically transformed Gaussian-linear null.")
print("IAAFT is approximate: check spectral mismatch and stationarity before attributing")
print("a rejection to nonlinear dynamics; this is not a proof of deterministic chaos.")
