"""Vertical-line RQA on the logistic map: Marwan, Romano, Thiel & Kurths
(2007), Sec. 3.5.3, eqs. 62-64 (laminarity, trapping time); cf. Marwan et al. (2002).

Laminarity (LAM) is the fraction of recurrent points forming vertical lines
of length >= 2, i.e. states that stay in the same neighborhood for
consecutive steps. Sweeping r in x -> r x (1 - x) (batched over r), LAM is
zero on periodic orbits that jump between distinct points (period-4 and
-8 below r ~ 3.57, the period-5 window near 3.74 and the period-3 window at
r ~ 3.84), where DET is high (1.0 at r = 3.84). LAM is nonzero only in the
chaotic regime and peaks just after the band merging near r ~ 3.678, a
chaos-chaos transition that DET barely registers. The recurrence time
entropy (RTE, Little et al. 2007) is reported alongside.
"""

import torch

from torchnlts.recurrence import (
    determinism,
    hard_recurrence_matrix,
    laminarity,
    recurrence_rate,
    rt_entropy,
    trapping_time,
)

num_points = 1000
transient = 500
epsilon = 0.1
theiler_window = 1

parameters = torch.linspace(3.5, 4.0, 26, dtype=torch.float64)
state = torch.full((len(parameters),), 0.4, dtype=torch.float64)
orbit = []
for step in range(transient + num_points):
    state = parameters * state * (1 - state)
    if step >= transient:
        orbit.append(state.clone())
orbit = torch.stack(orbit, dim=1).unsqueeze(-1)  # [B, T, 1]

recurrence = hard_recurrence_matrix(orbit, epsilon=epsilon)
rate = recurrence_rate(recurrence, theiler_window=theiler_window)
det = determinism(recurrence, lmin=2, theiler_window=theiler_window)
lam = laminarity(recurrence, lmin=2, theiler_window=theiler_window)
tt = trapping_time(recurrence, lmin=2, theiler_window=theiler_window)
rte = rt_entropy(recurrence, theiler_window=theiler_window)

print(f"{'r':>7}  {'RR':>6}  {'DET':>6}  {'LAM':>6}  {'TT':>6}  {'RTE':>6}")
for index, value in enumerate(parameters.tolist()):
    print(
        f"{value:>7.3f}  {rate[index]:>6.3f}  {det[index]:>6.3f}  {lam[index]:>6.3f}  {tt[index]:>6.3f}  {rte[index]:>6.3f}"
    )

print("\nLAM is 0 on periodic orbits (DET stays high there) and peaks just")
print("after the band merging near r = 3.678. RR also changes with r at a")
print("fixed epsilon, and this is one run of the map, not a statistical test.")
