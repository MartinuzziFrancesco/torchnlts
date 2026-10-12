"""Tests for torchnlts.neighbors.radius.radius_neighbors.

Dual ground truth:
  1. Paper: Grassberger & Procaccia (1983), eq. 1 -- Heaviside inclusion
     test ``Theta(r - |X_i - X_j|)``, inclusive at ``d == r``;
     hand-derived on a small point set.
  2. Julia: Neighborhood.jl's `bulksearch(tree, pts, WithinRange(r),
     Theiler(w))`, confirmed by inspection to also be inclusive at
     ``d == r``.
"""

import torch

from tests.conftest import run_julia
from torchnlts.neighbors.radius import radius_neighbors


def test_radius_neighbors_hand_derived():
    x = torch.tensor([0.0, 1.0, 2.0, 3.5, 10.0, 10.2, 20.0])
    mask = radius_neighbors(x, radius=2.0, theiler_window=1)
    # Point 0 (0.0): only index 2 (2.0) is within r=2.0 and outside the
    # Theiler window; boundary is inclusive.
    assert mask[0].nonzero().flatten().tolist() == [2]
    # Point 2 (2.0): symmetric, only index 0.
    assert mask[2].nonzero().flatten().tolist() == [0]
    # Point 1 (1.0): nearest non-excluded candidate is 2.5 away, no hits.
    assert mask[1].sum() == 0


def test_radius_neighbors_infinite_radius_still_excludes_theiler():
    # Regression: inf <= inf is True, so r=inf must not rely on an inf-filled distance matrix to exclude Theiler pairs.
    x = torch.tensor([0.0, 1.0, 2.0])
    mask = radius_neighbors(x, radius=float("inf"), theiler_window=1)
    assert mask[1].sum() == 0
    assert not mask[0, 0] and not mask[0, 1]
    assert mask[0, 2]


def test_radius_neighbors_shapes_and_symmetry():
    x = torch.randn(2, 12, 3)
    mask = radius_neighbors(x, radius=1.0, theiler_window=0)
    assert mask.shape == (2, 12, 12)
    assert torch.equal(mask, mask.transpose(-1, -2))


def test_radius_neighbors_matches_neighborhood_jl():
    torch.manual_seed(2)
    x = torch.randn(15, 2, dtype=torch.float64)
    r, w = 1.2, 1
    mask = radius_neighbors(x, radius=r, theiler_window=w)

    rows = ";".join(" ".join(f"{v:.17g}" for v in row) for row in x.numpy())
    julia_code = f"""
    using Neighborhood, StaticArrays
    X = [{rows}]
    pts = [SVector{{2}}(X[i, 1], X[i, 2]) for i in 1:size(X, 1)]
    tree = KDTree(pts, Euclidean())
    idxs, _ = bulksearch(tree, pts, WithinRange({r}), Theiler({w}))
    for i in eachindex(pts)
        println(join(sort(idxs[i] .- 1), ","))
    end
    """
    lines = run_julia(julia_code).splitlines()
    assert len(lines) == len(x)  # a point without neighbors prints an empty line, which strip() would drop
    for i, line in enumerate(lines):
        julia_neighbors = sorted(int(v) for v in line.split(",")) if line else []
        torch_neighbors = sorted(mask[i].nonzero().flatten().tolist())
        assert julia_neighbors == torch_neighbors, (i, julia_neighbors, torch_neighbors)
