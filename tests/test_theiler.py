"""Tests for torchnlts.neighbors.theiler.

Dual ground truth:
  1. Paper: Theiler (1986), Phys. Rev. A 34, 2427, Sec. III -- exclude
     pairs with |i - j| <= w; hand-derived for a small T.
  2. Julia: Neighborhood.jl's `Theiler(w)` predicate (the exact object
     DynamicalSystemsBase/ChaosTools pass into neighbor searches), which
     this test confirms uses the identical |i - j| <= w rule.
"""

import numpy
import pytest
import torch

from tests.conftest import run_julia
from torchnlts.neighbors.theiler import apply_theiler_window, theiler_window_mask


def test_theiler_window_mask_hand_derived():
    # T=5, w=1: exclude |i-j| <= 1 (Theiler 1986, Sec. III).
    mask = theiler_window_mask(5, 1)
    expected = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, False, True, True],
            [True, False, False, False, True],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_theiler_window_mask_w0_excludes_only_diagonal():
    mask = theiler_window_mask(4, 0)
    assert torch.equal(mask, ~torch.eye(4, dtype=torch.bool))


def test_apply_theiler_window_fills_excluded():
    d = torch.arange(1.0, 17.0).reshape(4, 4)
    out = apply_theiler_window(d, window=1, fill_value=float("inf"))
    assert torch.isinf(out[0, 0]) and torch.isinf(out[0, 1])
    assert out[0, 2] == d[0, 2]
    assert out[0, 3] == d[0, 3]


def test_apply_theiler_window_batched():
    d = torch.rand(3, 5, 5)
    out = apply_theiler_window(d, window=1)
    mask = theiler_window_mask(5, 1)
    assert torch.equal(torch.isinf(out), (~mask).unsqueeze(0).expand(3, 5, 5))


def test_apply_theiler_window_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        apply_theiler_window(torch.zeros(3, 5), window=1)


def test_apply_theiler_window_gradcheck():
    d = torch.rand(5, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: apply_theiler_window(x, window=1, fill_value=0.0), (d,))


def test_theiler_window_matches_neighborhood_jl():
    t, w = 7, 2
    mask = theiler_window_mask(t, w)
    julia_code = f"""
    using Neighborhood
    th = Theiler({w})
    for i in 0:{t - 1}, j in 0:{t - 1}
        print(th(i, j) ? 1 : 0)
    end
    """
    stdout = run_julia(julia_code)
    excluded_flat = [c == "1" for c in stdout.strip()]
    julia_excluded = torch.tensor(excluded_flat).reshape(t, t)
    # Neighborhood.jl's Theiler(i, j) is True where the pair is EXCLUDED, the negation of our "valid pair" mask.
    assert torch.equal(mask, ~julia_excluded)


@pytest.mark.parametrize("bad", [1.5, 2.0, "1", None, torch.tensor(1.5)])
def test_theiler_window_must_be_an_integer(bad):
    with pytest.raises(ValueError, match="theiler_window must be an integer"):
        theiler_window_mask(5, bad)
    with pytest.raises(ValueError, match="theiler_window must be an integer"):
        apply_theiler_window(torch.zeros(5, 5), bad)


def test_theiler_window_accepts_numpy_and_huge_integers():
    assert torch.equal(theiler_window_mask(5, numpy.int64(1)), theiler_window_mask(5, 1))
    assert not theiler_window_mask(5, 10**30).any()
    assert theiler_window_mask(5, -(10**30)).all()
