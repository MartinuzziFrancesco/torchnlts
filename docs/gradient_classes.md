# Gradient-class table

Living record of the gradient-class label for every public torchnlts function. Updated in the same commit
that adds or changes the function.

| Function | Class | Notes |
|---|---|---|
| `torchnlts.neighbors.distance.pairwise_distance` | exact | Ordinary autodiff via `torch.cdist`. Non-differentiable at zero distance (coincident points); gradcheck in `tests/test_distance.py` avoids that singularity by using two disjoint point sets. |
| `torchnlts.neighbors.theiler.theiler_window_mask` | exact | Boolean, index-only; no gradient to speak of. |
| `torchnlts.neighbors.theiler.apply_theiler_window` | exact | Mask depends only on time indices, never on data, so no data-dependent boundary; gradcheck in `tests/test_theiler.py`. |
