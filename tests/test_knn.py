"""Tests for torchnlts.neighbors.knn.knn_search.

Dual ground truth:
  1. Paper: Kantz & Schreiber (2004), Ch. 3 -- k nearest phase-space
     neighbors excluding a Theiler window; hand-derived on a 1D point
     set small enough to enumerate by hand.
  2. Julia: Neighborhood.jl's `bulksearch(tree, pts, NeighborNumber(k),
     Theiler(w))`, the exact routine DynamicalSystemsBase/ChaosTools use
     for phase-space neighbor search.
"""

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.neighbors.knn import knn_search


def test_knn_search_hand_derived():
    # Points on a line: 0, 1, 2, 3.5, 10, 10.2, 20. Theiler window w=1.
    x = torch.tensor([0.0, 1.0, 2.0, 3.5, 10.0, 10.2, 20.0])
    dist, idx = knn_search(x, num_neighbors=2, theiler_window=1)
    # Point 0: index 1 is excluded by the Theiler window; neighbors are index 2 (d=2.0), then 3 (d=3.5).
    assert idx[0].tolist() == [2, 3]
    assert torch.allclose(dist[0], torch.tensor([2.0, 3.5]))
    # Point 4: index 5 (d=0.2) is excluded by the Theiler window; neighbors are index 2 (d=8.0), then 1 (d=9.0).
    assert idx[4].tolist() == [2, 1]
    assert torch.allclose(dist[4], torch.tensor([8.0, 9.0]))


def test_knn_search_k_exceeds_available_points():
    x = torch.tensor([0.0, 1.0, 2.0])
    dist, idx = knn_search(x, num_neighbors=4, theiler_window=0)
    assert dist.shape == (3, 4)
    # theiler=0 excludes only self, leaving 2 real candidates; the 4th requested neighbor pads with inf / -1.
    assert torch.isinf(dist[:, 3]).all()
    assert (idx[:, 3] == -1).all()
    assert torch.isfinite(dist[:, :2]).all()


def test_knn_search_theiler_exhausts_candidates():
    # w=1 excludes both other points for the middle index, leaving zero valid candidates.
    x = torch.tensor([0.0, 1.0, 2.0])
    dist, idx = knn_search(x, num_neighbors=1, theiler_window=1)
    assert torch.isinf(dist[1]).all()
    assert (idx[1] == -1).all()


def test_knn_search_shapes():
    x = torch.randn(2, 20, 3)
    dist, idx = knn_search(x, num_neighbors=4, theiler_window=1)
    assert dist.shape == (2, 20, 4)
    assert idx.shape == (2, 20, 4)


def test_knn_search_sorted_ascending():
    x = torch.randn(30)
    dist, _ = knn_search(x, num_neighbors=5, theiler_window=0)
    assert torch.all(dist[:, :-1] <= dist[:, 1:])


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, "2", None, float("nan")])
def test_knn_search_rejects_invalid_num_neighbors(bad):
    with pytest.raises(ValueError, match="num_neighbors"):
        knn_search(torch.randn(10), bad)


def test_knn_search_accepts_numpy_integer_num_neighbors():
    trajectory = torch.randn(12, 2)
    distances, indices = knn_search(trajectory, 3)
    numpy_distances, numpy_indices = knn_search(trajectory, numpy.int64(3))
    assert torch.equal(distances, numpy_distances) and torch.equal(indices, numpy_indices)


def test_knn_search_gradcheck():
    # Well-separated points avoid ties at the k-th/(k+1)-th boundary (the non-differentiable point).
    torch.manual_seed(0)
    x = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)

    def f(x):
        dist, _ = knn_search(x, num_neighbors=2, theiler_window=1)
        return dist

    assert torch.autograd.gradcheck(f, (x,))


def test_knn_search_matches_neighborhood_jl():
    torch.manual_seed(1)
    x = torch.randn(15, 2, dtype=torch.float64)
    k, w = 3, 1
    dist, idx = knn_search(x, num_neighbors=k, theiler_window=w)

    rows = ";".join(" ".join(f"{v:.17g}" for v in row) for row in x.numpy())
    julia_code = f"""
    using Neighborhood, StaticArrays
    X = [{rows}]
    pts = [SVector{{2}}(X[i, 1], X[i, 2]) for i in 1:size(X, 1)]
    tree = KDTree(pts, Euclidean())
    idxs, dists = bulksearch(tree, pts, NeighborNumber({k}), Theiler({w}))
    for i in eachindex(pts)
        println(join(idxs[i] .- 1, ","), "|", join(dists[i], ","))
    end
    """
    lines = run_julia(julia_code).strip().splitlines()
    assert len(lines) == len(x)
    for i, line in enumerate(lines):
        idx_part, dist_part = line.split("|")
        julia_idx = [int(v) for v in idx_part.split(",")]
        julia_dist = [float(v) for v in dist_part.split(",")]
        # Compare neighbor sets (tie order may differ) and sorted distances.
        assert set(julia_idx) == set(idx[i].tolist()), (i, julia_idx, idx[i].tolist())
        assert torch.allclose(torch.tensor(julia_dist, dtype=torch.float64), dist[i], atol=1e-10)
