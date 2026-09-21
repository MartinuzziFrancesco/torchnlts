"""Reproduces Marwan, Romano, Thiel & Kurths (2007): recurrence
quantification analysis distinguishes deterministic signals (periodic
or chaotic) from stochastic ones, using only their recurrence
structure -- no model of the underlying dynamics needed.

Determinism (DET, the fraction of recurrent points that form diagonal
lines of length >= 2) is compared across three signals, each
thresholded to the *same* recurrence rate (10%) so the comparison isn't
confounded by one signal simply having a looser threshold: a periodic
sine wave, the Lorenz system's chaotic x-coordinate, and white noise.
"""

import torch

from torchnlts.recurrence.matrix import hard_recurrence_matrix
from torchnlts.recurrence.rqa import determinism, recurrence_rate
from torchnlts.systems import trajectory

torch.manual_seed(0)
t = torch.linspace(0, 100, 1000)
periodic = torch.sin(t)
noise = torch.randn(1000)
chaotic = trajectory("Lorenz", 1000)[:, 0]
chaotic = (chaotic - chaotic.mean()) / chaotic.std()

target_rate = 0.1
theiler_window = 1


def epsilon_for_target_rate(x: torch.Tensor) -> float:
    """Binary-search the recurrence threshold hitting `target_rate`."""
    low, high = 1e-3, 5.0
    for _ in range(30):
        mid = (low + high) / 2
        rate = recurrence_rate(hard_recurrence_matrix(x, mid), theiler_window)
        low, high = (mid, high) if rate < target_rate else (low, mid)
    return mid


print(f"{'signal':>20}  {'epsilon':>8}  {'RR':>6}  {'DET':>6}")
for name, signal in [("periodic (sine)", periodic), ("chaotic (Lorenz x)", chaotic), ("white noise", noise)]:
    epsilon = epsilon_for_target_rate(signal)
    recurrence = hard_recurrence_matrix(signal, epsilon)
    rate = recurrence_rate(recurrence, theiler_window)
    det = determinism(recurrence, lmin=2, theiler_window=theiler_window)
    print(f"{name:>20}  {epsilon:>8.4f}  {rate.item():>6.3f}  {det.item():>6.3f}")

print("\nAt matched recurrence rate, both deterministic signals (periodic and")
print("chaotic) have far higher determinism than white noise -- DET separates")
print("'governed by some dynamics' from 'no dynamics at all', regardless of")
print("whether that underlying dynamics is periodic or chaotic.")
