"""Reproduces Russell, Hanson & Ott (1980)'s box-counting estimate of
the generalized dimension spectrum D_q on a textbook strongly
multifractal measure: a p-weighted binary cascade (recursively split
[0, 1) into two halves with unequal probability p/1-p at each of
``depth`` levels). D0 (box-counting/"capacity" dimension), D1
(information dimension), and D2 (Hentschel & Procaccia 1983's own
order-2 identification with correlation dimension) all fall out of
the same `torchnlts.dimension.generalized_dimension` call
at different `order` values. The cascade's D_q spectrum is checked to
be non-increasing across the order sweep -- a *finite-sample*
regression check consistent with Rényi's theorem about the true,
limiting D_q, not a proof that this specific numerical estimate
satisfies it in general.

Also applies the same method to the Lorenz attractor and compares its
box-counting D2 against `torchnlts.dimension.correlation`'s existing
*pairwise-distance*-based D2 (Grassberger & Procaccia 1983) -- both
target the same theoretical D2 (Hentschel & Procaccia's identification
above), but as genuinely different finite-sample *estimators* of it,
and box-counting is well known to need far more data than the
correlation sum to converge to the same accuracy for a
higher-dimensional attractor (the historical reason Grassberger &
Procaccia introduced the correlation sum in the first place). This
comparison is illustrative, not a tight quantitative match: at a
practical sample size the box-counting estimate undershoots the
correlation-sum one and even distorts D_q's ordering across an order
sweep (not run here, since it isn't this tutorial's own claim) --
observed directly during this module's development, not asserted by
this tutorial itself, which only prints both values side by side.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.dimension import box_counting_entropy, correlation_dimension, correlation_sum, generalized_dimension

# Part 1: a p-weighted binary cascade -- D_q spectrum and monotonicity.
generator = torch.Generator().manual_seed(0)
depth = 20
probability_left = 0.8
bits = torch.bernoulli(torch.full((20000, depth), 1 - probability_left, dtype=torch.float64), generator=generator)
weights = 0.5 ** torch.arange(1, depth + 1, dtype=torch.float64)
cascade = (bits * weights).sum(dim=-1)

box_sizes = torch.tensor([0.02, 0.01, 0.005, 0.002, 0.001], dtype=torch.float64)
orders = [0.0, 0.5, 1.0, 2.0, 4.0]
dimensions = []
print(f"{'order':>6}  {'D_q':>8}")
for order in orders:
    entropy = box_counting_entropy(cascade, box_size=box_sizes, order=order)
    dimension = generalized_dimension(box_sizes, entropy)
    dimensions.append(dimension)
    print(f"{order:>6}  {dimension.item():>8.4f}")

assert all(dimensions[i] >= dimensions[i + 1] - 1e-6 for i in range(len(dimensions) - 1)), (
    "D_q must be non-increasing in order (Renyi's theorem, via Russell, Hanson & Ott 1980)"
)
# Non-increasing alone would trivially pass for a constant (order-independent)
# estimator too; this cascade's own D_q genuinely varies with order (unlike
# a uniform interval's, whose D_q is close to 1 for every order), so this
# also rules that degenerate case out.
assert dimensions[0] - dimensions[-1] > 0.1, (
    "D_q should vary substantially with order for this strongly multifractal cascade"
)

# Part 2: the Lorenz attractor -- box-counting D2 vs the library's
# existing pairwise-distance D2, an illustrative (not tight) comparison.
state = torch.as_tensor(Lorenz().make_trajectory(5000)).to(torch.float64)
state = (state - state.mean(dim=0)) / state.std(dim=0)

box_size_lorenz = torch.logspace(-1.5, 0.0, 12, dtype=torch.float64)
box_counting_d2 = generalized_dimension(box_size_lorenz, box_counting_entropy(state, box_size_lorenz, order=2.0))

radius_lorenz = torch.logspace(-1.5, -0.3, 12, dtype=torch.float64)
correlation_sum_d2 = correlation_dimension(radius_lorenz, correlation_sum(state, radius_lorenz, theiler_window=10))

print("\nLorenz attractor, N=5000 points:")
print(f"  box-counting D2:      {box_counting_d2.item():.3f}")
print(f"  correlation-sum D2:   {correlation_sum_d2.item():.3f}  (literature value commonly cited around 2.05)")
print("  (box-counting undershoots at this sample size -- it needs far more data than the correlation sum")
print("   to resolve a 3D attractor's fine structure, the same reason Grassberger & Procaccia (1983)")
print("   introduced the correlation sum as a more data-efficient alternative to box-counting.)")

print("\nAll checks passed: the binary cascade's D_q spectrum is non-increasing and substantially varies")
print("across the order sweep, consistent with Renyi's theorem about the true D_q, and D0/D1/D2 fall out")
print("of one method as special cases of the same generalized_dimension call at different orders.")
