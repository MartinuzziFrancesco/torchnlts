"""Reproduces Bandt & Pompe (2002) and Richman & Moorman (2000):
ordinal-pattern and template-matching entropy measures, illustrated on
a sine wave, Lorenz x, and white noise. At these sampling intervals and
parameters, the smooth deterministic signals have lower entropy than
white noise. This ordering is not universal: chaotic signals can have
high entropy, and correlated stochastic signals can have low entropy.

Permutation entropy is normalized to [0, 1] (1 = every ordinal pattern
equally likely, the maximum-entropy case); sample entropy is unbounded
and depends on the template length, delay, and matching radius.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.entropy import permutation_entropy, sample_entropy

torch.manual_seed(0)
t = torch.linspace(0, 100, 1000)
periodic = torch.sin(t)
noise = torch.randn(1000)
chaotic = torch.as_tensor(Lorenz().make_trajectory(1000))[:, 0]

signals = {"periodic (sine)": periodic, "chaotic (Lorenz x)": chaotic, "white noise": noise}
signals = {
    name: (x - x.mean()) / x.std() for name, x in signals.items()
}  # comparable scale for sample_entropy's radius

print(f"{'signal':>20}  {'PermEn':>7}  {'SampEn':>7}")
for name, signal in signals.items():
    perm_en = permutation_entropy(signal, embedding_dim=4, delay=1, normalize=True)
    sample_en = sample_entropy(signal, radius=0.2, embedding_dim=2, delay=1)
    print(f"{name:>20}  {perm_en.item():>7.3f}  {sample_en.item():>7.3f}")

print("\nAt these sampling intervals and parameters, sine and Lorenz x have")
print("lower entropy than white noise. This measures temporal regularity;")
print("neither statistic alone establishes determinism or low dimensionality.")
