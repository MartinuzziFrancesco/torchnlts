"""Reproduces the Fraser & Swinney (1986) recipe: choosing a delay embedding's
delay from the first minimum of the self mutual information of a scalar
observable, compared with autocorrelation criteria.

Applies it to the x-coordinate of the Lorenz system. The autocorrelation
zero crossing and the first-minimum methods pick much larger delays than the
mutual-information minimum here: the autocorrelation only measures linear
dependence, and for Lorenz x it stays positive over the search range.
The chosen delay (in time units, using the sampling step of the generated
trajectory) is then used for a three-dimensional delay embedding.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.convert import as_tensor
from torchnlts.embedding import auto_mutual_information, delay_embedding, estimate_delay

times, states = Lorenz().make_trajectory(3000, return_times=True)
sampling_step = float(times[1] - times[0])
observable = as_tensor(states[:, 0])

max_delay = 60
information = auto_mutual_information(observable, max_delay)
print(f"sampling step: {sampling_step:.4f} time units")
print(f"{'delay':>5}  {'AMI (nats)':>10}")
for delay in range(0, 30, 3):
    print(f"{delay:>5}  {information[delay].item():>10.3f}")

print(f"\n{'method':>8}  {'delay':>5}  {'time units':>10}")
chosen = {}
for method in ("mi_min", "ac_1/e", "ac_min", "ac_zero"):
    chosen[method] = estimate_delay(observable, max_delay, method).item()
    print(f"{method:>8}  {chosen[method]:>5}  {chosen[method] * sampling_step:>10.3f}")
print("\nac_zero returning max_delay means no zero crossing was found in the searched range.")

embedding = delay_embedding(observable, 3, chosen["mi_min"])
spread = embedding.std(dim=0)
print(
    f"\nembedding with the mutual-information delay: shape {tuple(embedding.shape)}, per-axis std {[round(value, 2) for value in spread.tolist()]}"
)
print(f"correlation between first two embedding axes: {torch.corrcoef(embedding[:, :2].T)[0, 1].item():.3f}")
