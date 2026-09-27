import pytest
import torch

from torchnlts._shape import _ensure_batched, _ols_slope, _restore_batch, _restore_shape_preserving


def test_ensure_batched_1d():
    x = torch.arange(5.0)
    b, ndim = _ensure_batched(x)
    assert b.shape == (1, 5, 1)
    assert ndim == 1
    assert torch.equal(b.flatten(), x)


def test_ensure_batched_2d():
    x = torch.randn(5, 3)
    b, ndim = _ensure_batched(x)
    assert b.shape == (1, 5, 3)
    assert ndim == 2
    assert torch.equal(b[0], x)


def test_ensure_batched_3d_passthrough():
    x = torch.randn(2, 5, 3)
    b, ndim = _ensure_batched(x)
    assert b is x
    assert ndim == 3


def test_ensure_batched_rejects_bad_ndim():
    try:
        _ensure_batched(torch.randn(2, 3, 4, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError for 4D input")


def test_roundtrip_1d_and_2d():
    x1 = torch.randn(5)
    b1, n1 = _ensure_batched(x1)
    out1 = _restore_batch(b1.squeeze(-1), n1)
    assert torch.equal(out1, x1)

    x2 = torch.randn(5, 3)
    b2, n2 = _ensure_batched(x2)
    out2 = _restore_batch(b2, n2)
    assert torch.equal(out2, x2)


def test_restore_batch_keeps_3d():
    x = torch.randn(2, 5, 3)
    out = _restore_batch(x, 3)
    assert out is x


def test_restore_shape_preserving_roundtrips_all_three_ndims():
    x1 = torch.randn(5)
    b1, n1 = _ensure_batched(x1)
    assert torch.equal(_restore_shape_preserving(b1, n1), x1)

    x2 = torch.randn(5, 3)
    b2, n2 = _ensure_batched(x2)
    assert torch.equal(_restore_shape_preserving(b2, n2), x2)

    x3 = torch.randn(2, 5, 3)
    b3, n3 = _ensure_batched(x3)
    out3 = _restore_shape_preserving(b3, n3)
    assert out3 is x3


def test_restore_shape_preserving_rejects_bad_ndim():
    try:
        _restore_shape_preserving(torch.randn(2, 5, 3), 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for ndim=4")


def test_ols_slope_repeated_sweep_is_nan():
    # log(0.1) repeated: centering leaves a rounding-sized nonzero
    # denominator, which used to give a finite slope instead of NaN.
    sweep = torch.log(torch.full((11,), 0.1, dtype=torch.float64))
    values = torch.arange(1.0, 12.0, dtype=torch.float64)
    assert torch.isnan(_ols_slope(sweep, values))
    assert torch.isnan(_ols_slope(sweep, values.unsqueeze(-1).repeat(1, 3))).all()


def test_ols_slope_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="matching"):
        _ols_slope(torch.tensor([1.0, 2.0]), torch.tensor([2.0]))


def test_ols_slope_supports_vmap():
    sweep = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
    assert torch.allclose(torch.vmap(_ols_slope)(sweep, 2 * sweep + 1), torch.tensor([2.0, 2.0]))
