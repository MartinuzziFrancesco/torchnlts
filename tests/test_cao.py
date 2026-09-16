"""Tests for torchnlts.embedding.cao.cao_e1 / cao_e2.

Dual ground truth:
  1. Paper: Cao (1997), Physica D 110, eq. 1-3 (E1) and eq. 5 (E2) --
     checked against an independent brute-force reference implementation.
  2. Julia: DelayEmbeddings.jl's delay_afnn, for E1 only. Its source
     (not just its docstring) revealed two conventions not obvious from
     the paper alone -- a coincident nearest neighbor falls back to the
     second-nearest, and the average divides by num_common - 1 -- both
     replicated in cao_e1 and in the brute-force reference below.
     stochastic_indicator (E2) has its own indexing quirks that don't
     match a clean reading of eq. 5 (see cao.py's module docstring), so
     per the fallback rule E2 is validated against the paper and
     brute-force only, not Julia.
"""

import math
import subprocess

import torch

from torchnlts.embedding.cao import cao_e1, cao_e2
from torchnlts.embedding.delay import delay_embedding

JULIA_PROJECT = "juliaenv"


def _brute_force_cao(trajectory, embedding_dim, delay, theiler_window, new_coordinate_only):
    """Independent, unoptimized transcription of Cao (1997) eq. 1-3/5,
    including delay_afnn's coincident-neighbor fallback (see cao.py).
    """

    def e_of(dim):
        embedded_low = delay_embedding(trajectory, dim, delay)
        embedded_high = delay_embedding(trajectory, dim + 1, delay)
        num_common = embedded_high.shape[0]
        embedded_low = embedded_low[:num_common]
        total, count = 0.0, 0
        for i in range(num_common):
            candidates = []
            for j in range(num_common):
                if abs(i - j) <= theiler_window:
                    continue
                d = torch.linalg.vector_norm(embedded_low[i] - embedded_low[j]).item()
                candidates.append((d, j))
            if not candidates:
                continue
            candidates.sort(key=lambda pair: pair[0])
            best_d, best_j = candidates[0]
            if best_d == 0.0 and len(candidates) > 1:
                best_d, best_j = candidates[1]
            if best_d == 0.0:
                continue
            if new_coordinate_only:
                high = abs(trajectory[i + dim * delay].item() - trajectory[best_j + dim * delay].item())
            else:
                high = torch.linalg.vector_norm(embedded_high[i] - embedded_high[best_j]).item()
            total += high / best_d
            count += 1
        denominator = (num_common - 1) if not new_coordinate_only else count
        return total / denominator

    return e_of(embedding_dim + 1) / e_of(embedding_dim)


def test_cao_e1_matches_brute_force_reference():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    for embedding_dim in (1, 2, 3):
        expected = _brute_force_cao(trajectory, embedding_dim, delay=1, theiler_window=1, new_coordinate_only=False)
        actual = cao_e1(trajectory, embedding_dim, delay=1, theiler_window=1).item()
        assert math.isclose(actual, expected), (embedding_dim, actual, expected)


def test_cao_e2_matches_brute_force_reference():
    torch.manual_seed(0)
    trajectory = torch.randn(60, dtype=torch.float64)
    for embedding_dim in (1, 2, 3):
        expected = _brute_force_cao(trajectory, embedding_dim, delay=1, theiler_window=1, new_coordinate_only=True)
        actual = cao_e2(trajectory, embedding_dim, delay=1, theiler_window=1).item()
        assert math.isclose(actual, expected), (embedding_dim, actual, expected)


def test_cao_e2_random_noise_stays_further_from_one_than_deterministic():
    # Cao's qualitative claim (eq. 5's purpose): E2 approaches 1 much
    # more slowly, as embedding_dim grows, for a random signal than for
    # a deterministic one -- not that E2 sits near 1 at any single small
    # embedding_dim (a common misreading; finite-sample E2 can be far
    # from 1 at low dimensions even for pure noise).
    torch.manual_seed(2)
    noise = torch.randn(600, dtype=torch.float64)
    t = torch.linspace(0, 60 * math.pi, 600, dtype=torch.float64)
    deterministic = torch.sin(t)
    embedding_dim = 4
    e2_noise = cao_e2(noise, embedding_dim, delay=1, theiler_window=1).item()
    e2_deterministic = cao_e2(deterministic, embedding_dim, delay=5, theiler_window=5).item()
    assert abs(e2_noise - 1.0) > abs(e2_deterministic - 1.0)


def test_cao_e1_saturates_for_low_dimensional_system():
    # A 2D limit cycle observed through one coordinate: E1 should be
    # markedly below 1 (still unfolding) at d=1, and near 1 by d=2-3.
    t = torch.linspace(0, 60 * math.pi, 600, dtype=torch.float64)
    trajectory = torch.sin(t)
    e1_low = cao_e1(trajectory, embedding_dim=1, delay=5, theiler_window=5).item()
    e1_high = cao_e1(trajectory, embedding_dim=3, delay=5, theiler_window=5).item()
    assert abs(e1_high - 1.0) < abs(e1_low - 1.0)


def test_cao_e1_matches_delayembeddings_jl():
    torch.manual_seed(1)
    trajectory = torch.randn(100, dtype=torch.float64)
    delay, embedding_dim = 2, 2

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using DelayEmbeddings
    s = [{values}]
    e1s = delay_afnn(s, {delay}, {embedding_dim}:{embedding_dim})
    println(e1s[1])
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_e1 = float(result.stdout.strip())
    torch_e1 = cao_e1(trajectory, embedding_dim, delay=delay, theiler_window=0).item()
    assert math.isclose(torch_e1, julia_e1, rel_tol=1e-6), (torch_e1, julia_e1)
