"""Tests for torchnlts.embedding.delay.delay_embedding.

Dual ground truth:
  1. Paper: Takens (1981) delay-coordinate reconstruction; hand-derived
     on an integer sequence small enough to check by hand.
  2. Julia: numerically matches DelayEmbeddings.jl's `embed(s, m, tau)`
     exactly (same forward-offset convention).
"""

import subprocess

import torch

from torchnlts.embedding.delay import delay_embedding

JULIA_PROJECT = "juliaenv"


def test_delay_embedding_hand_derived():
    x = torch.arange(10.0)
    y = delay_embedding(x, embedding_dim=3, delay=2)
    expected = torch.tensor(
        [
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
            [4.0, 6.0, 8.0],
            [5.0, 7.0, 9.0],
        ]
    )
    assert torch.equal(y, expected)


def test_delay_embedding_tau1_m1_is_identity():
    x = torch.randn(10)
    y = delay_embedding(x, embedding_dim=1, delay=1)
    assert torch.equal(y.squeeze(-1), x)


def test_delay_embedding_multivariate_and_batched():
    x = torch.randn(2, 20, 3)
    y = delay_embedding(x, embedding_dim=4, delay=2)
    assert y.shape == (2, 20 - 3 * 2, 4 * 3)
    expected_row0 = torch.cat([x[0, 0], x[0, 2], x[0, 4], x[0, 6]])
    assert torch.equal(y[0, 0], expected_row0)


def test_delay_embedding_too_short_raises():
    try:
        delay_embedding(torch.randn(5), embedding_dim=4, delay=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a too-short trajectory")


def test_delay_embedding_gradcheck():
    x = torch.randn(10, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: delay_embedding(x, embedding_dim=3, delay=2), (x,))


def test_delay_embedding_matches_delayembeddings_jl():
    torch.manual_seed(3)
    x = torch.randn(15, dtype=torch.float64)
    m, tau = 4, 3
    y = delay_embedding(x, embedding_dim=m, delay=tau)

    values = ", ".join(f"{v:.17g}" for v in x.numpy())
    julia_code = f"""
    using DelayEmbeddings
    s = [{values}]
    Y = embed(s, {m}, {tau})
    for row in Y
        println(join(row, ","))
    end
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_rows = [[float(v) for v in line.split(",")] for line in result.stdout.strip().splitlines()]
    julia_y = torch.tensor(julia_rows, dtype=torch.float64)
    assert torch.allclose(y, julia_y, atol=1e-10)
