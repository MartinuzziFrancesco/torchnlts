"""Gottwald-Melbourne 0-1 test for chaos on the logistic map, following
Gottwald & Melbourne (2009), "On the implementation of the 0-1 test for
chaos", SIAM J. Appl. Dyn. Syst. 8(1), 129-145, Figs. 4 and 7.

The logistic map x -> r x (1 - x) is run for N = 2000 points per
parameter, as in their Fig. 4, and `torchnlts.chaos.zero_one` computes K as
the median of K_c over 100 equally spaced frequencies in (pi/5, 4*pi/5),
the range they recommend to avoid resonances near 0 and pi. K is compared
with the sign of the largest Lyapunov exponent, the average of
log|r (1 - 2 x)| along the orbit. The paper reports K close to 0 in the
periodic windows (e.g. r = 3.55) and close to 1 in the chaotic regime.

Their weakly chaotic example (r just above the edge of chaos, 3.5699...)
needs more data than N = 2000 to reach K = 1 (they report K = 0.027 at
N = 2000, rising with N); this script shows the same slow convergence.
"""

import math

import torch

from torchnlts.chaos.zero_one import zero_one_test


def logistic_orbit(parameter: float, num_points: int, transient: int = 1000) -> torch.Tensor:
    """Orbit of the logistic map from x0 = 0.4, discarding a transient."""
    value, orbit = 0.4, []
    for step in range(transient + num_points):
        value = parameter * value * (1 - value)
        if step >= transient:
            orbit.append(value)
    return torch.tensor(orbit, dtype=torch.float64)


def lyapunov_exponent(parameter: float, orbit: torch.Tensor) -> float:
    return torch.log((parameter * (1 - 2 * orbit)).abs()).mean().item()


frequencies = torch.linspace(math.pi / 5, 4 * math.pi / 5, 100, dtype=torch.float64)

print(f"{'r':>6}  {'K':>7}  {'lyapunov':>9}  regime")
cases = [(3.2, "period 2"), (3.55, "period 8"), (3.835, "period 3"), (3.6, "chaotic, 2 bands"), (3.7, "chaotic"), (3.9, "chaotic"), (3.97, "chaotic"), (4.0, "chaotic")]
for parameter, regime in cases:
    orbit = logistic_orbit(parameter, 2000)
    statistic = zero_one_test(orbit, frequencies).item()
    exponent = lyapunov_exponent(parameter, orbit)
    print(f"{parameter:>6.3f}  {statistic:>7.4f}  {exponent:>9.4f}  {regime}")
    if exponent < -0.01:
        assert abs(statistic) < 0.1, "K should be near 0 in a periodic window"
    else:
        assert statistic > 0.9, "K should be near 1 for chaos"

# K over a grid of r against the sign of the Lyapunov exponent: the two
# classifications (K > 0.5, lambda > 0) agree except very close to
# bifurcation points and the edge of chaos, where finite N blurs K.
grid = torch.arange(3.5, 4.0 + 1e-9, 0.01, dtype=torch.float64).tolist()
agree = 0
for parameter in grid:
    orbit = logistic_orbit(parameter, 2000)
    agree += int((zero_one_test(orbit, frequencies).item() > 0.5) == (lyapunov_exponent(parameter, orbit) > 0))
fraction = agree / len(grid)
print(f"\nK > 0.5 agrees with the sign of the Lyapunov exponent at {agree}/{len(grid)} values of r in [3.5, 4]")
assert fraction > 0.9

# Weak chaos: K grows with N just above the edge of chaos (r = 3.569945672 + 0.001).
weak = 3.569945672 + 0.001
sizes = (2000, 20000)
weak_statistics = [zero_one_test(logistic_orbit(weak, size), frequencies, max_lag=size // 10).item() for size in sizes]
print("\nweak chaos at r = 3.5710:")
for size, statistic in zip(sizes, weak_statistics):
    print(f"  N = {size:>6}: K = {statistic:.4f}")
assert weak_statistics[1] > weak_statistics[0]

print("\nK is near 0 in the periodic windows and near 1 in the chaotic regime, tracking the sign of")
print("the Lyapunov exponent as in Gottwald & Melbourne (2009); weak chaos needs a longer series.")
