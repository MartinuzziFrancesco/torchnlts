"""Reproduces Takens (1981): a scalar observable's delay embedding is
diffeomorphic to the original attractor.

Embeds only the x-coordinate of the Lorenz system and checks that each
point's nearest neighbor in the 3D reconstruction matches its nearest
neighbor in the true 3D state space far more often than chance.
"""

import torch
from dysts.flows import Lorenz

from torchnlts.embedding import delay_embedding
from torchnlts.neighbors import knn_search

full_state = torch.as_tensor(Lorenz().make_trajectory(2000))
observable = full_state[:, 0]

embedding_dim, delay, theiler_window = 3, 5, 5
reconstruction = delay_embedding(observable, embedding_dim, delay)
full_state = full_state[: reconstruction.shape[0]]

_, true_neighbor = knn_search(full_state, num_neighbors=1, theiler_window=theiler_window)
_, reconstructed_neighbor = knn_search(reconstruction, num_neighbors=1, theiler_window=theiler_window)

agreement = (true_neighbor[:, 0] == reconstructed_neighbor[:, 0]).float().mean().item()
chance = 1 / (full_state.shape[0] - 2 * theiler_window - 1)

print(f"nearest-neighbor agreement (true 3D state vs. delay reconstruction): {agreement:.3f}")
print(f"agreement expected by chance: {chance:.5f}")
