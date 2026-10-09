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
entropy is the alternative. Report its Monte Carlo rank p-value
(``surrogate_test``, ties counted conservatively, the observed sample
included), whose minimum with 20 surrogates is 1/21, rather than assuming Gaussian tails
from a z-score. This is an approximate surrogate test: spectral mismatch,
nonstationarity, and finite-record effects can also cause rejection;
rejection alone does not establish deterministic chaos.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.entropy import permutation_entropy
from torchnlts.surrogate import iaaft_surrogate, surrogate_test

torch.manual_seed(0)
signal = torch.as_tensor(Lorenz().make_trajectory(1000))[:, 0]
signal = (signal - signal.mean()) / signal.std()


def entropy(data):
    return permutation_entropy(data, embedding_dim=4, delay=1, normalize=True)


num_surrogates = 20
result = surrogate_test(
    signal, entropy, iaaft_surrogate, num_surrogates, tail="left", generator=torch.Generator().manual_seed(0), max_iterations=50
)
real_statistic = result.statistic
surrogate_statistics = result.surrogate_statistics
p_value = result.p_value.item()

# Regenerate the tested ensemble: same generator seed and batch shape as surrogate_test used.
ensemble = iaaft_surrogate(
    signal.view(1, -1, 1).repeat(num_surrogates, 1, 1), max_iterations=50, generator=torch.Generator().manual_seed(0)
)
assert torch.equal(entropy(ensemble), surrogate_statistics)
target_magnitude = torch.fft.rfft(signal).abs()
spectral_errors = (torch.fft.rfft(ensemble[..., 0], dim=-1).abs() - target_magnitude).norm(dim=-1) / target_magnitude.norm()

print(f"real signal permutation entropy:        {real_statistic.item():.4f}")
print(f"surrogate permutation entropy (mean/std): {surrogate_statistics.mean().item():.4f} / {surrogate_statistics.std().item():.4f}")
print(f"lower-tail rank p-value: {p_value:.4f} (minimum possible: {1 / (num_surrogates + 1):.4f})")
print(f"relative Fourier-magnitude error (min/max): {spectral_errors.min().item():.4f} / {spectral_errors.max().item():.4f}")
print("\nNominal 5% rank-test decision:", "reject" if p_value <= 0.05 else "do not reject")
print("Interpret against the stationary, statically transformed Gaussian-linear null.")
print("IAAFT is approximate: check spectral mismatch and stationarity before attributing")
print("a rejection to nonlinear dynamics; this is not a proof of deterministic chaos.")
