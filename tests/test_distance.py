"""Tests for torchnlts.neighbors.distance.pairwise_distance.

Dual ground truth:
  1. Paper: Grassberger & Procaccia (1983) use pairwise Euclidean
     distance ||X_i - X_j|| between embedded vectors; validated here
     against a hand-derived analytic case (unit square / 3-4-5 triangle)
     computable from the definition directly, with no library involved.
  2. Julia: Distances.jl's `pairwise(Euclidean(), X)`, the pairwise-
     distance routine DynamicalSystems.jl/ChaosTools.jl/DelayEmbeddings.jl
     use internally (e.g. for correlation-sum and neighbor searches).
     Compared numerically at float64 with rtol=1e-10 (same algorithm,
     same precision, so we expect near machine-epsilon agreement rather
     than a loose tolerance).
"""

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.neighbors.distance import pairwise_distance


def test_pairwise_distance_hand_derived():
    # 3-4-5 right triangle in 2D: analytic Euclidean distances are exact.
    x = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    d = pairwise_distance(x)
    expected = torch.tensor(
        [
            [0.0, 3.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 5.0, 0.0],
        ]
    )
    assert torch.allclose(d, expected, atol=1e-6)


def test_pairwise_distance_shapes():
    assert pairwise_distance(torch.randn(5)).shape == (5, 5)
    assert pairwise_distance(torch.randn(5, 3)).shape == (5, 5)
    assert pairwise_distance(torch.randn(2, 5, 3)).shape == (2, 5, 5)


def test_pairwise_distance_cross():
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randn(6, 3, dtype=torch.float64)
    d = pairwise_distance(x, other=y)
    assert d.shape == (4, 6)
    # Brute-force cross-check independent of torch.cdist.
    brute = torch.zeros(4, 6, dtype=torch.float64)
    for i in range(4):
        for j in range(6):
            brute[i, j] = torch.linalg.norm(x[i] - y[j])
    assert torch.allclose(d, brute, atol=1e-10)


def test_pairwise_distance_rejects_batched_ndim_mismatch():
    # Regression: batching only one trajectory silently broadcast to the wrong shape instead of raising.
    unbatched = torch.randn(3)
    batched = torch.randn(2, 5, 1)
    for x, y in ((unbatched, batched), (batched, unbatched)):
        with pytest.raises(ValueError):
            pairwise_distance(x, other=y)


def test_pairwise_distance_rejects_unequal_batch_size():
    x = torch.randn(3, 5, 1)
    y = torch.randn(4, 6, 1)
    with pytest.raises(ValueError):
        pairwise_distance(x, other=y)


def test_pairwise_distance_rejects_feature_dim_mismatch():
    x = torch.randn(5, 2)
    y = torch.randn(6, 4)
    with pytest.raises(ValueError):
        pairwise_distance(x, other=y)


def test_pairwise_distance_chebyshev():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d = pairwise_distance(x, norm_order=float("inf"))
    assert torch.allclose(d, torch.tensor([[0.0, 4.0], [4.0, 0.0]]))


@pytest.mark.parametrize("bad", [-1.0, float("nan"), "2", "abc", None, 1j, 10**400, torch.tensor([1.0, 2.0])])
def test_pairwise_distance_rejects_invalid_norm_order(bad):
    with pytest.raises(ValueError, match="norm_order"):
        pairwise_distance(torch.randn(5, 2), norm_order=bad)


@pytest.mark.parametrize("good", [1, numpy.float64(1.0), torch.tensor(1.0)])
def test_pairwise_distance_accepts_real_scalar_norm_order(good):
    trajectory = torch.randn(5, 2, dtype=torch.float64)
    assert torch.equal(pairwise_distance(trajectory, norm_order=good), pairwise_distance(trajectory, norm_order=1.0))


def test_pairwise_distance_gradcheck():
    # Cross-distance between independent point sets avoids the r=0 singularity.
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda points, other: pairwise_distance(points, other=other), (x, y))


def test_pairwise_distance_float32_large_offset_stable():
    # Regression: cdist's matmul mode (T > 25) cancels catastrophically in float32 for large offsets.
    x = (10_000 + torch.arange(30, dtype=torch.float32)).reshape(-1, 1)
    d = pairwise_distance(x)
    expected = torch.abs(
        torch.arange(30, dtype=torch.float32).unsqueeze(0) - torch.arange(30, dtype=torch.float32).unsqueeze(1)
    )
    assert torch.allclose(d, expected, atol=1e-3)


def test_pairwise_distance_matches_julia_distances_jl():
    torch.manual_seed(0)
    x = torch.randn(6, 3, dtype=torch.float64)
    d_torch = pairwise_distance(x)

    rows = ";".join(" ".join(f"{v:.17g}" for v in row) for row in x.numpy())
    julia_code = f"""
    using Distances
    X = [{rows}]'
    D = pairwise(Euclidean(), X, dims=2)
    for i in 1:size(D, 1)
        println(join(D[i, :], ","))
    end
    """
    stdout = run_julia(julia_code)
    d_julia = torch.tensor(
        [[float(v) for v in line.split(",")] for line in stdout.strip().splitlines()], dtype=torch.float64
    )
    assert d_julia.shape == d_torch.shape == (6, 6)
    assert torch.allclose(d_julia, d_torch, atol=1e-10, rtol=0)
