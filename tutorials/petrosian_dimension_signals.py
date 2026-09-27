"""Petrosian fractal dimension illustrated on periodic, chaotic, and noise
signals. Unlike Higuchi (which computes roughness via curve length and
averaging over strides), Petrosian counts sign changes in the first
difference: a single-pass, deterministic estimate with no sweep
parameter.

Petrosian's dimension ranges theoretically from 1 (no sign changes: a
monotonic series) to values greater than 1 (more sign changes: higher
frequency content or roughness). This tutorial verifies that on three
clearly-contrasting signals -- a smooth periodic wave, chaotic dynamics,
and white noise -- the ordering is consistent: smoother signals have
lower estimates than noisier ones.

This ordering depends on the sampling: D_P only counts reversals of the
first difference, so a periodic signal sampled near its Nyquist rate
(e.g. cos(pi * t) at integer t, which reverses every step) scores
higher than white noise. It also does not distinguish chaos from noise
in general; Lorenz x lands in between here because of this sampling.

Reference: Petrosian, A. (1995), "Kolmogorov complexity of finite
sequences and recognition of different preictal EEG patterns", Proc.
8th IEEE Symposium on Computer-Based Medical Systems, 212-217.
"""

import torch

from torchnlts.dimension.petrosian import petrosian_dimension
from torchnlts.systems import trajectory

torch.manual_seed(0)
time_points = torch.linspace(0, 100, 2000, dtype=torch.float64)
periodic = torch.sin(time_points)
noise = torch.randn(2000, dtype=torch.float64)
chaotic = trajectory("Lorenz", 2000)[:, 0].to(torch.float64)

signals = {"periodic (sine)": periodic, "chaotic (Lorenz x)": chaotic, "white noise": noise}
signals = {name: (signal - signal.mean()) / signal.std() for name, signal in signals.items()}  # comparable scale

print(f"{'signal':>20}  {'D_P':>6}")
dimensions = {}
for name, signal in signals.items():
    dimension = petrosian_dimension(signal)
    dimensions[name] = dimension
    print(f"{name:>20}  {dimension.item():>6.4f}")

assert dimensions["periodic (sine)"] < 1.005, "smooth periodic signal should have very low D_P"
assert dimensions["white noise"] > 1.03, "white noise should have noticeably higher D_P"
assert (
    dimensions["periodic (sine)"]
    < dimensions["chaotic (Lorenz x)"]
    < dimensions["white noise"]
), "Petrosian should order the three signals: periodic < chaotic < noise"

print("\nAll checks passed: Petrosian orders the three signals as expected --")
print("lowest for the smooth periodic sine wave, highest for white noise, and in between")
print("for the chaotic Lorenz x-coordinate, at this sampling rate. D_P is derived from the")
print("count of derivative sign reversals, so the ordering changes with the sampling.")
