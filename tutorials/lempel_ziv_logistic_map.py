"""Illustrates Lempel-Ziv complexity (Lempel & Ziv 1976; Kaspar &
Schuster 1987's practical algorithm) against the logistic map's
period-doubling route to chaos, the same benchmark system
`tutorials/lyapunov_from_data.py` uses for the Lyapunov exponent --
not the paper's own exact experiment (Lempel & Ziv 1976 doesn't study
the logistic map), and not tied to a specific reproduction target from
the biomedical-signal-analysis literature that also applies this
measure to periodic/chaotic/random signals -- a standard illustration
of the measure's behavior, not a numeric result copied from a paper.

Rather than trust a memorized claim about which `r` values are
periodic (an earlier draft of this script got one wrong -- r=3.6 turns
out to already be chaotic, past the logistic map's Feigenbaum
accumulation point at r=3.569945..., not periodic as first assumed),
each swept r's regime is *derived* here from the sign of its own exact
Lyapunov exponent (the logistic map's derivative is known in closed
form, f'(x) = r(1 - 2x), so this needs no data-driven estimator).
r=3.83 is included specifically because it sits *inside* the
nominally-chaotic band yet has a negative Lyapunov exponent -- the
well-known period-3 window -- to check whether Lempel-Ziv complexity
tracks genuine dynamical regularity there too, not just "low r".
"""

import numpy as np
import torch
from dysts.maps import Logistic

from torchnlts.entropy import lempel_ziv_complexity

torch.manual_seed(0)
num_points, warmup = 2000, 200


def logistic_trajectory(r: float) -> torch.Tensor:
    system = Logistic(parameters={"r": r})
    trajectory = torch.as_tensor(system.make_trajectory(num_points + warmup, init_cond=np.array([0.4])), dtype=torch.float64)
    return trajectory[warmup:]


def logistic_lyapunov_exponent(r: float, trajectory: torch.Tensor) -> float:
    """Exact Lyapunov exponent from the map's own closed-form derivative
    f'(x) = r(1 - 2x), averaged along the trajectory -- independent of
    any data-driven estimator, used here purely to label each r's
    regime (periodic if negative, chaotic if positive), not as a demo
    of `torchnlts.lyapunov` itself (see `lyapunov_from_data.py` for that).
    """
    derivative = (r * (1 - 2 * trajectory)).abs()
    derivative = derivative[derivative > 0]  # a trajectory point at exactly 0 or 1 would give log(0)
    return torch.log(derivative).mean().item()


r_values = [3.2, 3.5, 3.6, 3.7, 3.83, 3.9, 3.99, 4.0]

print(f"{'r':>6}  {'Lyapunov':>9}  {'regime':>10}  {'normalized LZ':>14}")
periodic_complexities = []
chaotic_complexities = {}
for r in r_values:
    trajectory = logistic_trajectory(r)
    lyapunov = logistic_lyapunov_exponent(r, trajectory)
    complexity = lempel_ziv_complexity(trajectory, normalize=True).item()
    regime = "periodic" if lyapunov < 0 else "chaotic"
    print(f"{r:>6.2f}  {lyapunov:>9.4f}  {regime:>10}  {complexity:>14.4f}")
    if lyapunov < 0:
        periodic_complexities.append(complexity)
    else:
        chaotic_complexities[r] = complexity

random_noise = torch.rand(num_points, dtype=torch.float64)  # same [0, 1] range as the logistic map
random_complexity = lempel_ziv_complexity(random_noise, normalize=True).item()
print(f"{'--':>6}  {'--':>9}  {'random':>10}  {random_complexity:>14.4f}")

max_periodic = max(periodic_complexities)
full_chaos = chaotic_complexities[4.0]
print(f"\nmax over regimes with a negative Lyapunov exponent: {max_periodic:.4f}")
print(f"r=4.0 (most positive Lyapunov exponent here):        {full_chaos:.4f}")
print(f"random noise:                                        {random_complexity:.4f}")

assert full_chaos > 5 * max_periodic, "the r with the most positive Lyapunov exponent should read far more complex than any negative-exponent regime"
assert abs(full_chaos - random_complexity) < 0.2, "r=4.0 should read nearly as complex as pure randomness, for this trajectory length"
print("\nAll checks passed: every r with a negative Lyapunov exponent (including r=3.83's period-3 window,")
print("embedded inside the nominally chaotic band) reads as far less complex than r=4.0, whose own")
print("normalized complexity lands close to pure random noise's -- for the specific trajectory length,")
print("radius, and r values used here, not a general claim about chaos-vs-noise indistinguishability.")
print(f"\nNote r=3.60 above: its Lyapunov exponent is positive (genuinely chaotic), yet its normalized")
print(f"complexity ({chaotic_complexities[3.6]:.4f}) still reads as minimal -- mean-threshold binarization")
print("happens to collapse this particular trajectory into a period-2-looking symbol sequence, a real")
print("information loss binarization can cause, not a Lempel-Ziv complexity failure. Left in deliberately,")
print("not filtered out, as an honest limit of this measure's usual continuous-signal preprocessing step.")
