"""Reproduces Bandt & Pompe (2002) and Richman & Moorman (2000):
complexity/entropy measures that are low for *any* deterministic signal
(periodic or chaotic) and high for stochastic noise -- the same
"deterministic vs. random" distinction as
:mod:`recurrence_rqa.py`'s determinism, from a completely different
angle (ordinal-pattern statistics and embedded template matching,
rather than recurrence structure).

Permutation entropy is normalized to [0, 1] (1 = every ordinal pattern
equally likely, the maximum-entropy case); sample entropy is unbounded
but follows the same ordering.
"""

import torch

from torchnlts.entropy.permutation import permutation_entropy
from torchnlts.entropy.sample import sample_entropy
from torchnlts.systems import trajectory

torch.manual_seed(0)
t = torch.linspace(0, 100, 1000)
periodic = torch.sin(t)
noise = torch.randn(1000)
chaotic = trajectory("Lorenz", 1000)[:, 0]

signals = {"periodic (sine)": periodic, "chaotic (Lorenz x)": chaotic, "white noise": noise}
signals = {name: (x - x.mean()) / x.std() for name, x in signals.items()}  # comparable scale for sample_entropy's radius

print(f"{'signal':>20}  {'PermEn':>7}  {'SampEn':>7}")
for name, signal in signals.items():
    perm_en = permutation_entropy(signal, embedding_dim=4, delay=1, normalize=True)
    sample_en = sample_entropy(signal, radius=0.2, embedding_dim=2, delay=1)
    print(f"{name:>20}  {perm_en.item():>7.3f}  {sample_en.item():>7.3f}")

print("\nBoth deterministic signals (periodic, chaotic) score far lower on both")
print("measures than white noise, even though one of them is chaotic --")
print("'low-dimensional dynamics', not 'predictable moment to moment', is")
print("what these measures actually pick up on.")
