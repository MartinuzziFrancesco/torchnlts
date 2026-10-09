"""Reproduces the DFA scaling law of Peng et al. (1994), "Mosaic
organization of DNA nucleotides", Phys. Rev. E 49(2), 1685, and its
extension in Kantelhardt et al. (2001), Physica A 295, 441: F(n) ~
n**alpha, with alpha distinguishing uncorrelated noise (alpha=0.5) from
long-range-correlated (1/f, "pink", alpha=1.0) and non-stationary
(Brownian motion, alpha=1.5) signals -- the three canonical reference
points for the DFA exponent quoted throughout that literature.

Brownian motion is the cumulative sum of white noise; asymptotically,
integrating uncorrelated noise once raises alpha by 1. All three
reference values are asymptotic: one finite realization of each signal
(8000 samples, as below) lands close to them, not exactly on them, and
the white-to-Brownian difference comes out near, not exactly at, 1.
"""

import torch

from torchnlts.fluctuations import dfa_exponent, dfa_fluctuation

torch.manual_seed(0)
num_points = 8000


def pink_noise(num_points: int, dtype=torch.float64) -> torch.Tensor:
    """1/f-power-spectrum noise via spectral filtering of white noise."""
    white = torch.randn(num_points, dtype=dtype)
    spectrum = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(num_points).to(dtype).clone()
    freqs[0] = freqs[1]  # avoid a division by zero at DC
    pink = torch.fft.irfft(spectrum / torch.sqrt(freqs), n=num_points)
    return (pink - pink.mean()) / pink.std()


white_noise = torch.randn(num_points, dtype=torch.float64)
brownian_motion = torch.cumsum(white_noise, dim=0)
pink = pink_noise(num_points)

signals = {"white noise": white_noise, "1/f (pink) noise": pink, "Brownian motion": brownian_motion}
literature_alpha = {"white noise": 0.5, "1/f (pink) noise": 1.0, "Brownian motion": 1.5}

# Log-spaced window sizes, same convention as antropy/nolds: from 10 to
# 10% of the series length, kept well above order + 1 = 2.
window_sizes = sorted(
    set(int(value) for value in torch.logspace(1, torch.log10(torch.tensor(num_points / 10.0)), 20).tolist())
)
window_size_tensor = torch.tensor(window_sizes, dtype=torch.float64)

print(f"{'signal':>20}  {'alpha (DFA)':>12}  {'literature':>10}")
estimated_alpha = {}
for name, signal in signals.items():
    fluctuation = dfa_fluctuation(signal, window_sizes)
    alpha = dfa_exponent(window_size_tensor, fluctuation)
    estimated_alpha[name] = alpha.item()
    print(f"{name:>20}  {alpha.item():>12.3f}  {literature_alpha[name]:>10.1f}")

assert abs(estimated_alpha["white noise"] - 0.5) < 0.05, "white noise should give alpha close to 0.5"
assert abs(estimated_alpha["Brownian motion"] - 1.5) < 0.05, "Brownian motion should give alpha close to 1.5"
# Pink noise's estimate converges slowest of the three, so its band is wider.
assert abs(estimated_alpha["1/f (pink) noise"] - 1.0) < 0.15, "1/f noise should give alpha reasonably close to 1.0"

print("\nAll checks passed: one realization of each signal lands near the asymptotic values")
print("0.5 (white), 1.0 (1/f) and 1.5 (Brownian) used in the DFA literature.")
