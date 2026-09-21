"""Reproduces Kennel, Brown & Abarbanel (1992) and Cao (1997): choosing
a delay embedding's dimension from a scalar observable alone, with no
knowledge of the true state space.

Embeds only the x-coordinate of the Lorenz system at increasing
embedding dimensions. False nearest neighbors (Kennel et al.) drops to
(near) zero when enough delay coordinates unfold the attractor: points
that only *looked* close because the embedding was too low-dimensional stop
appearing as neighbors. Cao's E1 (a ratio-based alternative avoiding
FNN's distance-tolerance hyperparameters) saturates around the same
dimension. This is an embedding-dimension diagnostic, not an estimate
of the attractor's fractal dimension (about 2.05 for Lorenz).
"""

from torchnlts.embedding.cao import cao_e1
from torchnlts.embedding.fnn import hard_fnn
from torchnlts.systems import trajectory

full_state = trajectory("Lorenz", 3000)
observable = full_state[:, 0]

delay, theiler_window = 8, 8

print(f"{'dim':>3}  {'FNN fraction':>13}  {'Cao E1':>8}")
for embedding_dim in range(1, 7):
    fnn = hard_fnn(observable, embedding_dim, delay, theiler_window)
    e1 = cao_e1(observable, embedding_dim, delay, theiler_window)
    print(f"{embedding_dim:>3}  {fnn.item():>13.4f}  {e1.item():>8.4f}")

print("\nLook for FNN approaching zero and Cao E1 leveling off near 1.")
print("E1(d) = E(d+1)/E(d); its largest jump is not the saturation criterion.")
print("These diagnose an adequate embedding, not the attractor's fractal dimension.")
