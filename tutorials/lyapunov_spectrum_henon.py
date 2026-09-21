"""Reproduces Benettin, Galgani, Giorgilli & Strelcyn (1980) and Geist,
Parlitz & Lauterborn (1990): the full Lyapunov spectrum of a
differentiable map, via the Benettin/QR ("H2") method -- not from
observed data (see `lyapunov_from_data.py`) but by differentiating the
map itself with `torch.func`.

The Hénon map (a=1.4, b=0.3) is the standard benchmark here: its two
Lyapunov exponents are well known from the literature (approximately
0.42 and -1.62), one positive (chaotic stretching) and one negative
(contraction -- the Hénon map has zero net area growth since its
Jacobian determinant is the constant -b).
"""

import torch

from torchnlts.lyapunov.spectrum import lyapunov_spectrum


def henon_step(x, a=1.4, b=0.3):
    return torch.stack([1.0 - a * x[0] ** 2 + x[1], b * x[0]])


initial_state = torch.zeros(2, dtype=torch.float64)
spectrum = lyapunov_spectrum(henon_step, initial_state, num_steps=10000)

print(f"Lyapunov spectrum: {spectrum.tolist()}")
print("literature values are approximately [0.42, -1.62]")
print(f"sum of exponents: {spectrum.sum().item():.4f} (should be near log(b) = log(0.3) = {torch.tensor(0.3).log().item():.4f},")
print("the map's constant Jacobian determinant -- the sum of Lyapunov exponents")
print("is the average log-volume growth rate, independent of the QR method's")
print("individual-exponent estimates)")
