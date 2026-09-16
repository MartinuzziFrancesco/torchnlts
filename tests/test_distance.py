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

import subprocess

import torch

from torchnlts.neighbors.distance import pairwise_distance

JULIA_PROJECT = "juliaenv"


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
    d = pairwise_distance(x, y)
    assert d.shape == (4, 6)
    # Brute-force cross-check independent of torch.cdist.
    brute = torch.zeros(4, 6, dtype=torch.float64)
    for i in range(4):
        for j in range(6):
            brute[i, j] = torch.linalg.norm(x[i] - y[j])
    assert torch.allclose(d, brute, atol=1e-10)


def test_pairwise_distance_chebyshev():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d = pairwise_distance(x, norm_order=float("inf"))
    assert torch.allclose(d, torch.tensor([[0.0, 4.0], [4.0, 0.0]]))


def test_pairwise_distance_gradcheck():
    # Avoid the r=0 singularity (non-differentiable coincident points,
    # documented in distance.py) by checking cross-distance between two
    # independently-parametrized point sets, which never coincide.
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pairwise_distance, (x, y))


def test_pairwise_distance_float32_large_offset_stable():
    # torch.cdist's default compute_mode switches p=2 to a matmul-based
    # formula once T > 25, which catastrophically cancels in float32 for
    # points sharing a large offset. Regression guard for the fix in
    # distance.py (compute_mode="donot_use_mm_for_euclid_dist").
    x = (10_000 + torch.arange(30, dtype=torch.float32)).reshape(-1, 1)
    d = pairwise_distance(x)
    expected = torch.abs(torch.arange(30, dtype=torch.float32).unsqueeze(0) - torch.arange(30, dtype=torch.float32).unsqueeze(1))
    assert torch.allclose(d, expected, atol=1e-3)


def test_pairwise_distance_matches_julia_distances_jl():
    torch.manual_seed(0)
    x = torch.randn(6, 3, dtype=torch.float64)
    d_torch = pairwise_distance(x).numpy()

    rows = ";".join(" ".join(f"{v:.17g}" for v in row) for row in x.numpy())
    julia_code = f"""
    using Distances
    X = [{rows}]'
    D = pairwise(Euclidean(), X, dims=2)
    for i in 1:size(D, 1)
        println(join(D[i, :], ","))
    end
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    d_julia = [
        [float(v) for v in line.split(",")]
        for line in result.stdout.strip().splitlines()
    ]
    for i, row in enumerate(d_julia):
        for j, v in enumerate(row):
            assert abs(v - d_torch[i, j]) < 1e-10, (i, j, v, d_torch[i, j])
