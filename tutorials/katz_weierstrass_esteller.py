"""Katz (1988) fractal dimension on Weierstrass cosine functions, following
the comparison in Esteller et al. (2001), "A comparison of waveform
fractal dimension algorithms", IEEE Trans. Circuits Syst. I 48(2),
177-183, Fig. 2.

Test signal (their eq. 8, gamma=5, M=26):

    W_H(t) = sum_{i=0}^{M} gamma**(-i*H) * cos(2*pi*gamma**i*t)

The infinite series has graph dimension 2 - H; the truncated, sampled
series approximates it at the sampling resolution. Samples are taken at
t = j / N with N a power of two, and each phase gamma**i * j is reduced
mod N with integer arithmetic, so no cosine is evaluated at a huge
argument and the high-order terms do not alias onto one frequency.

Esteller et al. report that Katz's estimate increases with the true
dimension but is biased low and compressed toward 1, with a wider range
for shorter windows (about 1-1.3 below 250 points, 1-1.2 above 750),
while Higuchi's tracks the true value. This script reproduces those
qualitative findings; the magnitude of Katz's range does not match
(about 1.0-1.08 here). Katz's planar convention depends on the
amplitude scale relative to the unit index spacing, and the paper does
not give its signal scaling, so the exact numbers are not reproducible.
"""

import math

import torch

from torchnlts.dimension import higuchi_dimension, higuchi_length, katz_dimension


def weierstrass_cosine(hurst: float, num_points: int, gamma: int = 5, num_terms: int = 26) -> torch.Tensor:
    """W_H sampled at t = j / num_points, phases reduced exactly mod num_points."""
    sample_index = torch.arange(num_points, dtype=torch.int64)
    signal = torch.zeros(num_points, dtype=torch.float64)
    for order in range(num_terms + 1):
        phase = (pow(gamma, order, num_points) * sample_index) % num_points
        signal += gamma ** (-order * hurst) * torch.cos(2 * math.pi * phase.to(torch.float64) / num_points)
    return signal


# Sanity check: a straight line has D = 1 under Katz's definition.
line = torch.arange(0.0, 1000.0, dtype=torch.float64) * 2.5 + 1.0
assert torch.isclose(katz_dimension(line), torch.tensor(1.0, dtype=torch.float64))

true_dimensions = [1.1, 1.3, 1.5, 1.7, 1.9]
strides = torch.arange(1, 11, dtype=torch.float64)
katz_range = {}
for num_points in (256, 1024):
    print(f"\nN = {num_points} points")
    print(f"{'true D':>8}  {'Katz':>8}  {'Higuchi':>8}")
    katz_estimates = []
    for true_dimension in true_dimensions:
        signal = weierstrass_cosine(2.0 - true_dimension, num_points)
        katz_estimate = katz_dimension(signal).item()
        higuchi_estimate = higuchi_dimension(strides, higuchi_length(signal, strides)).item()
        katz_estimates.append(katz_estimate)
        print(f"{true_dimension:>8.2f}  {katz_estimate:>8.4f}  {higuchi_estimate:>8.3f}")
        assert abs(higuchi_estimate - true_dimension) < 0.1, "Higuchi should track the true dimension"
        assert katz_estimate < true_dimension, "Katz should be biased low"
    assert all(later > earlier for earlier, later in zip(katz_estimates, katz_estimates[1:])), (
        "Katz should increase with the true dimension"
    )
    katz_range[num_points] = katz_estimates[-1] - katz_estimates[0]

assert katz_range[256] > katz_range[1024], "Katz's range should be wider for the shorter window"

print("\nKatz increases with the true dimension but stays compressed near 1, with a wider range")
print("at N=256 than at N=1024; Higuchi stays within 0.1 of the true value. Both match Esteller")
print("et al. qualitatively; Katz's range here is narrower than their reported 1-1.2/1-1.3.")
