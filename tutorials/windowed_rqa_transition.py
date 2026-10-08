"""Windowed RQA on a logistic map whose parameter jumps from periodic to chaotic.

A sliding window along the main diagonal of the recurrence matrix gives RQA
measures as a function of time (Marwan et al. (2007), Sec. 3.5; cf.
RecurrenceAnalysis.jl's ``windowed``). The orbit below follows r = 3.5
(period 4) for 500 steps, then r = 3.9 (chaotic). Windowed DET and DIV = 1 / Lmax
change at the switch (index 500); windows straddling it mix both regimes.
"""

import torch

from torchnlts.recurrence.matrix import hard_recurrence_matrix
from torchnlts.recurrence.rqa import determinism, divergence, windowed_rqa

num_points = 1000
switch = 500
window_size = 100
step = 50
epsilon = 0.02

state = torch.tensor(0.4, dtype=torch.float64)
orbit = []
for index in range(300 + num_points):  # 300 discarded transient steps
    state = (3.5 if index < 300 + switch else 3.9) * state * (1 - state)
    if index >= 300:
        orbit.append(state)
orbit = torch.stack(orbit).unsqueeze(-1)

recurrence = hard_recurrence_matrix(orbit, epsilon=epsilon)
det = windowed_rqa(recurrence, determinism, window_size, step, theiler_window=1)
div = windowed_rqa(recurrence, divergence, window_size, step, theiler_window=1)

print(f"{'start':>5}  {'DET':>6}  {'DIV':>6}")
for index in range(len(det)):
    print(f"{index * step:>5}  {det[index]:>6.3f}  {div[index]:>6.3f}")

print("\nWindows starting before 400 lie in the periodic regime, those from 500 in the")
print("chaotic one. One orbit and one epsilon, not a statistical comparison.")
