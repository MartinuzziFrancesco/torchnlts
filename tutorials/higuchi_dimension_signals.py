"""Reproduces Higuchi (1988): the fractal dimension D_H of a scalar
signal's own graph, illustrated on a sine wave, Lorenz x, and white
noise. [1, 2] is D_H's theoretical range -- 1 for a smooth curve, 2 for
a maximally rough (space-filling) one -- not a guarantee for every
finite-sample estimate at an arbitrary stride sweep (this module's own
hand-derived test example, at a short, arbitrarily chosen 2-stride
sweep, gives 3.263, well outside it); the sweep and sample sizes below
are chosen specifically to keep these three signals' estimates inside
it, not because the estimator is bounded there unconditionally. And,
unlike the entropy measures compared in `entropy_measures.py`, this
needs no delay embedding at all: it's a direct roughness statistic on
the raw series.

Unlike most of this subpackage's other recent additions (box-counting,
ordinal/dispersion patterns, Lempel-Ziv parsing), Higuchi's curve
length is built from ordinary subtraction and absolute value, no
threshold or discrete search -- differentiable (piecewise) almost
everywhere.
"""

import torch

from torchnlts.dimension.higuchi import higuchi_dimension, higuchi_length
from torchnlts.systems import trajectory

torch.manual_seed(0)
t = torch.linspace(0, 100, 2000, dtype=torch.float64)
periodic = torch.sin(t)
noise = torch.randn(2000, dtype=torch.float64)
chaotic = trajectory("Lorenz", 2000)[:, 0].to(torch.float64)

signals = {"periodic (sine)": periodic, "chaotic (Lorenz x)": chaotic, "white noise": noise}
signals = {name: (x - x.mean()) / x.std() for name, x in signals.items()}  # comparable scale

ks = list(range(1, 21))
k_tensor = torch.tensor(ks, dtype=torch.float64)

print(f"{'signal':>20}  {'D_H':>6}")
dimensions = {}
for name, signal in signals.items():
    length = higuchi_length(signal, ks)
    dimension = higuchi_dimension(k_tensor, length)
    dimensions[name] = dimension
    print(f"{name:>20}  {dimension.item():>6.3f}")

assert 0.8 < dimensions["periodic (sine)"] < 1.2, "a smooth periodic signal should be close to D_H=1"
assert 1.8 < dimensions["white noise"] < 2.2, "white noise should be close to the maximally rough D_H=2"
assert dimensions["periodic (sine)"] < dimensions["chaotic (Lorenz x)"] < dimensions["white noise"]

print("\nAll checks passed: at this sampling and stride sweep, D_H orders the three signals as expected --")
print("close to 1 for the smooth periodic signal, close to 2 for white noise, and in between for the")
print("chaotic Lorenz x-coordinate's own graph -- without any delay embedding, unlike permutation/sample")
print("entropy's own comparison on these same signals. This ordering isn't a universal property of")
print("chaotic signals in general, any more than entropy_measures.py's own analogous ordering is.")
