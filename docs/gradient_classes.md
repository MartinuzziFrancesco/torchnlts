# Gradient-class table

Living record of the gradient-class label for every public torchnlts function. Updated in the same commit
that adds or changes the function.

| Function | Class | Notes |
|---|---|---|
| `torchnlts.neighbors.distance.pairwise_distance` | exact | Ordinary autodiff via `torch.cdist`. Non-differentiable at zero distance (coincident points); gradcheck in `tests/test_distance.py` avoids that singularity by using two disjoint point sets. |
| `torchnlts.neighbors.theiler.theiler_window_mask` | exact | Boolean, index-only; no gradient to speak of. |
| `torchnlts.neighbors.theiler.apply_theiler_window` | exact | Mask depends only on time indices, never on data, so no data-dependent boundary; gradcheck in `tests/test_theiler.py`. |
| `torchnlts.neighbors.knn.knn_search` | piecewise | `torch.topk` gradients route correctly to selected distances, but which points are selected is discrete: a tie at the k-th/(k+1)-th neighbor boundary is a non-differentiable jump. Gradcheck in `tests/test_knn.py` uses well-separated points to avoid that boundary. |
| `torchnlts.neighbors.radius.radius_neighbors` | none | Hard Heaviside inclusion test (bool output, no gradient). Differentiable relaxation belongs in a separately-named function built on the future `TrajectoryKernel`, not a hidden hard/soft swap. |
| `torchnlts.neighbors.kernels.trajectory_kernel` | smooth | Smooth everywhere at any realistic scale; gradcheck passes. Known ceiling (documented, not fixed): distance and temperature both within ~1 order of magnitude of float32's representable floor (~1.2e-38) can give an infinite temperature gradient, since `temperature**2` underflows to exactly 0 independent of formulation. float64 is unaffected (verified); no real trajectory data lives at that scale. |
| `torchnlts.embedding.delay.delay_embedding` | exact | Pure indexing/reshape; every output entry is exactly one input entry, no boundary. Gradcheck passes. |
