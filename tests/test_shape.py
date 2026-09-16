import torch

from torchnlts._shape import _ensure_batched, _restore_batch


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
