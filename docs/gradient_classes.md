# Gradient-class table

Living record of the gradient-class label for every public torchnlts function. Updated in the same commit
that adds or changes the function.

| Function | Class | Notes |
|---|---|---|
| `torchnlts.neighbors.distance.pairwise_distance` | exact | Ordinary autodiff via `torch.cdist`. Non-differentiable at zero distance (coincident points); gradcheck in `tests/test_distance.py` avoids that singularity by using two disjoint point sets. |
