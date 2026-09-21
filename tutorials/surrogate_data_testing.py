"""Reproduces the surrogate data testing methodology of Theiler, Eubank,
Longtin, Galdrikian & Farmer (1992), using Schreiber & Schmitz (1996)'s
IAAFT surrogates (the most refined of this library's three surrogate
methods -- see `torchnlts.surrogate.phase`/`aaft` for the simpler
methods IAAFT iteratively improves on).

The idea: generate many surrogates that share the real signal's linear
properties (amplitude distribution, approximate power spectrum) but
have randomized phase relationships, destroying any nonlinear
structure. If a nonlinear statistic computed on the real signal falls
far outside the range of that same statistic computed on the
surrogates, the real signal has genuine nonlinear structure the
surrogates can't reproduce -- evidence against the null hypothesis that
it came from a linear (Gaussian) process.

Permutation entropy is used as the nonlinear statistic here: the Lorenz
attractor's x-coordinate has a much lower ordinal-pattern entropy than
its own IAAFT surrogates, since chaotic determinism constrains which
ordinal patterns can occur next in a way linear correlation alone does
not.
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
surrogate_statistics = torch.tensor(
    [
        permutation_entropy(
            iaaft_surrogate(signal, max_iterations=50, generator=torch.Generator().manual_seed(seed)),
            embedding_dim=4,
            delay=1,
            normalize=True,
        ).item()
        for seed in range(num_surrogates)
    ]
)

z_score = (real_statistic - surrogate_statistics.mean()) / surrogate_statistics.std()

print(f"real signal permutation entropy:        {real_statistic.item():.4f}")
print(f"surrogate permutation entropy (mean/std): {surrogate_statistics.mean().item():.4f} / {surrogate_statistics.std().item():.4f}")
print(f"z-score of real signal vs. surrogate distribution: {z_score.item():.1f}")
print("\nA z-score this large (|z| >> 3) rejects the linear-process null")
print("hypothesis: the Lorenz signal's low ordinal-pattern entropy is not")
print("something its amplitude distribution and power spectrum alone explain.")
