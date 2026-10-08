"""Reproduces Marwan, Romano, Thiel & Kurths (2007): recurrence
quantification analysis compares recurrence structure in a sine wave,
Lorenz x, and white noise. This is a scalar-observable comparison,
without delay reconstruction; DET is not a general test of determinism.

Determinism (DET, the fraction of recurrent points that form diagonal
lines of length >= 2) is compared across three signals, each
thresholded to the *same* recurrence rate (10% of the full matrix; the printed
RR excludes the Theiler window, so it is slightly lower) so the comparison isn't
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


print(f"{'signal':>20}  {'RR':>6}  {'DET':>6}")
for name, signal in [("periodic (sine)", periodic), ("chaotic (Lorenz x)", chaotic), ("white noise", noise)]:
    recurrence = hard_recurrence_matrix(signal, recurrence_rate=target_rate)
    rate = recurrence_rate(recurrence, theiler_window)
    det = determinism(recurrence, lmin=2, theiler_window=theiler_window)
    print(f"{name:>20}  {rate.item():>6.3f}  {det.item():>6.3f}")

print("\nAt matched recurrence rate, sine and Lorenz x have higher DET than")
print("white noise here. DET measures diagonal recurrence structure;")
print("sampling and serial correlation matter, and stochastic signals can")
print("also have high DET. This comparison alone does not establish determinism.")
