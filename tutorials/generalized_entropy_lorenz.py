"""Reproduces the defining property of Rényi (1961) and Tsallis (1988)
generalized entropy: both are one-parameter families that reduce
exactly to Shannon entropy at their order parameter equal to 1, and
Rényi's own theorem that its entropy is non-increasing in that order
-- checked here not on a synthetic probability vector, but on the
*real* ordinal-pattern distribution the Lorenz system's x-coordinate
produces, reusing `torchnlts.entropy.ordinal_pattern_probabilities`
directly rather than recomputing it. This is the "pluggable probability
estimator" idea the gap survey flagged: `torchnlts.entropy.generalized`
knows nothing about ordinal patterns, delay embedding, or the Lorenz
system -- it just takes whatever probability distribution it's handed.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.entropy import ordinal_pattern_probabilities, permutation_entropy, renyi_entropy, tsallis_entropy

state = torch.as_tensor(Lorenz().make_trajectory(3000))[:, 0].to(torch.float64)
probability = ordinal_pattern_probabilities(state, embedding_dim=4, delay=1)

shannon = permutation_entropy(state, embedding_dim=4, delay=1)
renyi_at_1 = renyi_entropy(probability, alpha=1.0)
tsallis_at_1 = tsallis_entropy(probability, q=1.0)

print(f"permutation_entropy (Shannon):  {shannon.item():.6f}")
print(f"renyi_entropy(alpha=1):         {renyi_at_1.item():.6f}")
print(f"tsallis_entropy(q=1):           {tsallis_at_1.item():.6f}")
assert torch.equal(renyi_at_1, shannon), "Renyi at alpha=1 must be exactly permutation_entropy's own Shannon value"
assert torch.equal(tsallis_at_1, shannon), "Tsallis at q=1 must be exactly permutation_entropy's own Shannon value"

alphas = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 4.0, 10.0, float("inf")]
curve = [renyi_entropy(probability, alpha).item() for alpha in alphas]
print(f"\n{'alpha':>7}  {'Renyi entropy':>13}")
for alpha, value in zip(alphas, curve):
    print(f"{alpha:>7}  {value:>13.6f}")

assert all(curve[i] >= curve[i + 1] - 1e-9 for i in range(len(curve) - 1)), (
    "Renyi entropy must be non-increasing in alpha (Renyi 1961)"
)

print("\nAll checks passed: both generalized entropies recover permutation_entropy's own Shannon value")
print("exactly at order 1 (not just approximately), on the Lorenz x-coordinate's real ordinal-pattern")
print("distribution -- and Renyi entropy is non-increasing in its order across the full sweep, exactly")
print("as Renyi's own theorem requires, not just at the one alpha=1 case where it has to agree with Shannon.")
