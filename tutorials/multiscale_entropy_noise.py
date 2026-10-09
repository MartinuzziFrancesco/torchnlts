"""Reproduces the central qualitative claim of Costa, Goldberger & Peng
(2002), "Multiscale entropy analysis of complex physiologic time
series" (Phys. Rev. Lett. 89, 068102), Fig. 1: a single-scale entropy
can misleadingly rank pure white noise as "more complex" than a signal
with genuine long-range structure, simply because randomness maximizes
short-range irregularity -- and multiscale entropy's whole point is
that sweeping the coarse-graining scale reveals the ranking is
backwards.

White noise and 1/f ("pink") noise, both normalized to zero mean/unit
variance so `radius=0.15` (Costa et al.'s usual choice, relative to the
signal's own std) is directly comparable between them. At scale 1
(plain sample entropy), white noise's short-range unpredictability
gives it higher entropy. As the coarse-graining scale grows, averaging
increasingly resembles a constant for pure randomness (nothing but
noise to average away), so white noise's entropy trends downward
overall (not perfectly monotonically -- finite-sample noise gives it
the occasional small uptick, see the printed curve); pink noise's
correlated structure survives coarse-graining much better, so its
entropy stays roughly flat. The two curves cross, matching the shape
of Costa et al.'s own Fig. 1 (they use 3x10^4 points and scales up to
20; this uses 3000 points and scales up to 15, a smaller but
qualitatively matching setup).
"""

import torch

from torchnlts.entropy import multiscale_entropy

torch.manual_seed(0)
num_points = 3000
scales = list(range(1, 16))
radius = 0.15


def pink_noise(num_points: int, dtype=torch.float64) -> torch.Tensor:
    """1/f-power-spectrum noise via spectral filtering of white noise."""
    white = torch.randn(num_points, dtype=dtype)
    spectrum = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(num_points).to(dtype).clone()
    freqs[0] = freqs[1]  # avoid a division by zero at DC
    pink = torch.fft.irfft(spectrum / torch.sqrt(freqs), n=num_points)
    return (pink - pink.mean()) / pink.std()


white = torch.randn(num_points, dtype=torch.float64)
white = (white - white.mean()) / white.std()
pink = pink_noise(num_points)

white_curve = multiscale_entropy(white, scales=scales, radius=radius, embedding_dim=2)
pink_curve = multiscale_entropy(pink, scales=scales, radius=radius, embedding_dim=2)

print(f"{'scale':>5}  {'white SampEn':>13}  {'1/f SampEn':>11}")
for scale, white_entropy, pink_entropy in zip(scales, white_curve.tolist(), pink_curve.tolist()):
    print(f"{scale:>5}  {white_entropy:>13.4f}  {pink_entropy:>11.4f}")

print(f"\nscale=1:  white={white_curve[0]:.4f} > 1/f={pink_curve[0]:.4f}  (single-scale entropy ranks white noise 'more complex')")
print(f"scale={scales[-1]}: white={white_curve[-1]:.4f} < 1/f={pink_curve[-1]:.4f}  (the ranking has reversed)")

assert white_curve[0] > pink_curve[0], "at scale 1, white noise should read as more entropic than 1/f noise"
assert white_curve[-1] < pink_curve[-1], "by the largest scale, that ranking should have reversed"
print("\nAll checks passed: the scale-1 ranking (white 'more complex' than 1/f) is exactly backwards")
print("by the largest scale swept here -- reproducing why Costa et al. argue a single entropy value")
print("can't stand in for a system's complexity across scales.")
