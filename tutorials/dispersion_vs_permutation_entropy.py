"""Reproduces the core motivating claim of Rostaghi & Azami (2016),
"Dispersion Entropy: A Measure for Time-Series Analysis" (as restated,
citing the same paper, in Azami, Rostaghi, Abasolo & Escudero 2017,
arXiv:1606.01379): permutation entropy (Bandt & Pompe 2002) only looks
at the *order* of nearby values, discarding their actual amplitudes --
"some information about the amplitude values is ignored". Dispersion
entropy's whole point is to keep that information.

This is provable, not just empirically likely, in exact real-number
arithmetic: permutation entropy's ordinal patterns are the rank order
of each embedded window's values, and any strictly increasing
transform of a trajectory preserves every pairwise rank order exactly
-- so permutation entropy of x and of f(x), for any strictly
increasing f, must be *identical*. In floating-point arithmetic this
needs the *computed* f(x) to still preserve every pairwise order too
(a pathological or overflow-prone f could in principle introduce a
tie or a swap that never exists mathematically) -- true for the
specific transform checked here, f(x) = exp(x / std(x)), confirmed to
full floating-point precision (`torch.equal`, not just `isclose`), not
assumed in general. Dispersion entropy has no such invariance (its
Gaussian-CDF classification depends on the trajectory's actual
distribution of values, which f distorts even though it never swaps
any two values' order).
"""

import torch

from torchnlts.entropy import dispersion_entropy, permutation_entropy
from torchnlts.systems import trajectory

state = trajectory("Lorenz", 2000)[:, 0].to(torch.float64)
transformed = torch.exp(state / state.std())  # strictly increasing, so ranks are untouched

permutation_state = permutation_entropy(state, embedding_dim=3)
permutation_transformed = permutation_entropy(transformed, embedding_dim=3)
dispersion_state = dispersion_entropy(state, num_classes=6, embedding_dim=3)
dispersion_transformed = dispersion_entropy(transformed, num_classes=6, embedding_dim=3)

print(f"permutation entropy: original={permutation_state.item():.6f}, exp-transformed={permutation_transformed.item():.6f}")
print(f"dispersion entropy:  original={dispersion_state.item():.4f}, exp-transformed={dispersion_transformed.item():.4f}")

assert torch.equal(permutation_state, permutation_transformed), "permutation entropy must be exactly invariant under a strictly increasing transform"
assert abs(dispersion_state.item() - dispersion_transformed.item()) > 0.1, "dispersion entropy should visibly change under the same transform"

print("\nAll checks passed: permutation entropy is bit-for-bit unchanged by a transform that alters every")
print("value but no pairwise order, while dispersion entropy -- sensitive to the actual value distribution,")
print("not just rank order -- shifts by a large margin, reproducing the paper's stated motivation for it.")
