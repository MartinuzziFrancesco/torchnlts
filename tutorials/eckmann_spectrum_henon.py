"""Reproduces Eckmann, Kamphorst, Ruelle & Ciliberto (1986): the Lyapunov
spectrum of the Henon map (a=1.4, b=0.3) estimated from its x-series alone
and compared with the known exponents (about 0.42 and -1.62) and with
`lyapunov_spectrum` applied to the map itself.

The embedding dimension d_E and the matrix dimension d_M (the number of
exponents estimated) are separate choices. With d_M larger than the
system's dimension the extra exponents are spurious; the paper's
guidance is to pick d_M just above the number of positive exponents.
When d_E - 1 = (d_M - 1) * m with m > 1 (compressed embedding, e.g.
d_E=5, d_M=3), each fitted map advances m steps and the product follows
the orbit at stride m, as in the paper.
"""

import math

import torch

from torchnlts.lyapunov import eckmann_spectrum, lyapunov_spectrum


def henon_step(state, quadratic_coefficient=1.4, contraction=0.3):
    return torch.stack([1.0 - quadratic_coefficient * state[0] ** 2 + state[1], contraction * state[0]])


state = torch.tensor([0.1, 0.0], dtype=torch.float64)
values = []
for step in range(5200):
    state = henon_step(state)
    if step >= 200:
        values.append(state[0])
series = torch.stack(values)

reference = lyapunov_spectrum(
    henon_step, torch.tensor([0.1, 0.0], dtype=torch.float64), num_steps=20000, transient_steps=200
)
print(f"lyapunov_spectrum on the map: {reference.tolist()}")
print(f"sum of exponents: {reference.sum().item():.4f} (log 0.3 = {math.log(0.3):.4f})\n")

print(f"{'embedding_dim':>13} {'matrix_dim':>10} {'neighbors':>9}   exponents (per step)")
settings = [(2, 2, None), (2, 2, 12), (3, 3, None), (4, 4, None), (5, 3, None), (7, 4, None)]
results = {}
for embedding_dim, matrix_dim, num_neighbors in settings:
    exponents = eckmann_spectrum(series, embedding_dim, matrix_dim, num_neighbors=num_neighbors)
    results[(embedding_dim, matrix_dim, num_neighbors)] = exponents
    label = "default" if num_neighbors is None else num_neighbors
    print(
        f"{embedding_dim:>13} {matrix_dim:>10} {label!s:>9}   "
        + "  ".join(f"{value:7.3f}" for value in exponents.tolist())
    )

# Matrix dimension equal to the system's: both exponents are recovered
# (the contracting one with the local linear fit's known bias).
exact = results[(2, 2, None)]
assert abs(exact[0] - reference[0]) < 0.03
assert abs(exact[1] - reference[1]) < 0.15
assert abs(exact.sum() - math.log(0.3)) < 0.15

# Larger matrix dimension: the true exponents are still among the results,
# alongside spurious ones.
for key in [(3, 3, None), (4, 4, None), (5, 3, None)]:
    estimate = results[key]
    for true_value in reference.tolist():
        assert (estimate - true_value).abs().min() < 0.1
assert results[(4, 4, None)][0] > reference[0] + 0.2  # a spurious positive exponent appears
print("\nWith matrix_dim above the system's dimension, extra (spurious) exponents appear; the true ones remain.")
