"""Tests for torchnlts.embedding.fnn.hard_fnn.

Dual ground truth:
  1. Paper: Kennel, Brown & Abarbanel (1992), Phys. Rev. A 45, 3403,
     eq. 4-5 -- checked here via an independent brute-force reference
     implementation using their original (division-based) formula,
     confirming it agrees with the division-free reformulation actually
     used in fnn.py.
  2. Julia: DelayEmbeddings.jl's `delay_fnn(s, tau, ds; rtol, atol)`.
"""

import math
import subprocess

import torch

from torchnlts.embedding.delay import delay_embedding
from torchnlts.embedding.fnn import hard_fnn

JULIA_PROJECT = "juliaenv"


def _brute_force_fnn(trajectory, embedding_dim, delay, theiler_window, distance_tol, attractor_tol):
    """Direct, unoptimized transcription of Kennel et al.'s eq. 4-5,
    used only to cross-check hard_fnn's vectorized (division-free)
    reformulation -- deliberately independent code, not a copy.
    """
    embedded_low = delay_embedding(trajectory, embedding_dim, delay)
    embedded_high = delay_embedding(trajectory, embedding_dim + 1, delay)
    num_common = embedded_high.shape[0]
    embedded_low = embedded_low[:num_common]
    attractor_size = trajectory.std(correction=0).item()

    count, valid_count = 0, 0
    for i in range(num_common):
        best_j, best_d = None, math.inf
        for j in range(num_common):
            if abs(i - j) <= theiler_window:
                continue
            d = torch.linalg.vector_norm(embedded_low[i] - embedded_low[j]).item()
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            continue
        valid_count += 1
        r_low = best_d
        r_high = torch.linalg.vector_norm(embedded_high[i] - embedded_high[best_j]).item()
        if r_low == 0.0:
            is_false = r_high > 0.0
        else:
            is_false = (math.sqrt(r_high**2 - r_low**2) / r_low) > distance_tol
        is_false = is_false or (r_high / attractor_size) > attractor_tol
        count += int(is_false)
    return count / valid_count


def test_hard_fnn_matches_brute_force_reference():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    for embedding_dim in (1, 2, 3):
        expected = _brute_force_fnn(
            trajectory, embedding_dim, delay=1, theiler_window=1, distance_tol=10.0, attractor_tol=2.0
        )
        actual = hard_fnn(trajectory, embedding_dim, delay=1, theiler_window=1)
        assert math.isclose(actual.item(), expected), (embedding_dim, actual.item(), expected)


def test_hard_fnn_zero_low_distance_no_nan():
    # Duplicate windows give low_distance == 0; the division-free form must not produce NaN.
    trajectory = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 5.0, -3.0, 8.0])
    result = hard_fnn(trajectory, embedding_dim=2, delay=1, theiler_window=0)
    assert torch.isfinite(result)


def test_hard_fnn_theiler_exhausts_candidates_no_crash():
    # Every point excluded: the 0/0 fraction is NaN, not 0.
    trajectory = torch.tensor([0.0, 1.0, 2.0, 3.0])
    result = hard_fnn(trajectory, embedding_dim=1, delay=1, theiler_window=5)
    assert torch.isnan(result)


def test_hard_fnn_multichannel_attractor_size_not_mixed_across_channels():
    # Regression: attractor_size mixed time and channel in the std, so a huge-offset channel inflated it.
    torch.manual_seed(4)
    base = torch.randn(60, dtype=torch.float64)
    baseline = hard_fnn(base.unsqueeze(-1), embedding_dim=2, delay=1, theiler_window=1)

    shifted_channel = torch.full((60,), 1e6, dtype=torch.float64)
    two_channel = torch.stack([base, shifted_channel], dim=-1)
    actual = hard_fnn(two_channel, embedding_dim=2, delay=1, theiler_window=1)
    assert math.isclose(actual.item(), baseline.item())


def test_hard_fnn_decreases_toward_true_dimension():
    # FNN is high below the true embedding dimension and drops once reached (2D limit cycle seen through one coordinate).
    t = torch.linspace(0, 40 * math.pi, 400)
    trajectory = torch.sin(t)
    fnn_1 = hard_fnn(trajectory, embedding_dim=1, delay=5, theiler_window=5)
    fnn_2 = hard_fnn(trajectory, embedding_dim=2, delay=5, theiler_window=5)
    assert fnn_2 <= fnn_1


def test_hard_fnn_matches_delayembeddings_jl():
    torch.manual_seed(1)
    trajectory = torch.randn(80, dtype=torch.float64)
    delay, embedding_dim = 2, 3

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using DelayEmbeddings
    s = [{values}]
    fnns = delay_fnn(s, {delay}, {embedding_dim}:{embedding_dim})
    println(fnns[1])
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_fraction = float(result.stdout.strip())
    torch_fraction = hard_fnn(trajectory, embedding_dim, delay=delay, theiler_window=0).item()
    assert math.isclose(torch_fraction, julia_fraction), (torch_fraction, julia_fraction)
