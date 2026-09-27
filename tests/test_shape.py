import pytest
import torch

from torchnlts._shape import _ensure_batched, _ols_slope, _restore_batch, _restore_shape_preserving, _safe_log


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


def test_ols_slope_heterogeneous_batch_and_gradient():
    sweep = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    values = torch.stack([3 * sweep + 1, -2 * sweep, sweep**2], dim=1).requires_grad_(True)
    slope = _ols_slope(sweep, values)
    centered = sweep - sweep.mean()
    expected_quadratic = (centered * (sweep**2 - (sweep**2).mean())).sum() / (centered**2).sum()
    assert torch.allclose(slope, torch.stack([torch.tensor(3.0, dtype=torch.float64), torch.tensor(-2.0, dtype=torch.float64), expected_quadratic]))
    slope[0].backward()
    # d slope / d values[:, 0] = centered / sum(centered**2); other columns untouched.
    assert torch.allclose(values.grad[:, 0], centered / (centered**2).sum())
    assert (values.grad[:, 1:] == 0).all()
    assert torch.autograd.gradcheck(_ols_slope, (sweep.clone().requires_grad_(True), values.detach().clone().requires_grad_(True)))


def test_ols_slope_degenerate_sweep_does_not_poison_shared_gradient():
    # Row 0 of the vmapped batch has a repeated sweep; its NaN output must
    # not put NaN into the gradient of a parameter shared with row 1.
    parameter = torch.tensor(2.0, requires_grad=True)
    sweep = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    result = torch.vmap(_ols_slope)(sweep, parameter * sweep)
    assert torch.isnan(result[0])
    (gradient,) = torch.autograd.grad(result[1], parameter)
    assert torch.isclose(gradient, torch.tensor(1.0))


def test_ols_slope_nan_column_does_not_poison_sweep_gradient():
    sweep = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    values = torch.tensor([[float("nan"), 2.0], [float("nan"), 4.0], [float("nan"), 6.0]])
    slope = _ols_slope(sweep, values)
    assert torch.isnan(slope[0]) and torch.isclose(slope[1], torch.tensor(2.0))
    (gradient,) = torch.autograd.grad(slope[1], sweep)
    # d slope / d sweep for values = 2 * sweep, sweep = [1, 2, 3]: [1, 0, -1].
    assert torch.allclose(gradient, torch.tensor([1.0, 0.0, -1.0]))


def test_ols_slope_edge_sweeps_are_nan():
    for sweep, values in ((torch.tensor([]), torch.tensor([])), (torch.tensor([1.0]), torch.tensor([2.0])), (torch.tensor([1.0, float("nan"), 3.0]), torch.tensor([1.0, 2.0, 3.0]))):
        assert torch.isnan(_ols_slope(sweep, values))


def test_ols_slope_invalid_rank_raises():
    with pytest.raises(ValueError):
        _ols_slope(torch.ones(3, 1), torch.ones(3))
    with pytest.raises(ValueError):
        _ols_slope(torch.ones(3), torch.ones(3, 2, 1))


def test_ols_slope_dtype_promotion():
    sweep = torch.tensor([0.0, 200.0, 400.0])
    # float16 squares of the centered sweep (200**2 * 2) overflow half.
    half = _ols_slope(sweep.half(), sweep.half())
    assert half.dtype == torch.float16 and half.item() == 1.0
    integer = _ols_slope(sweep.long(), 3 * sweep.long())
    assert integer.dtype == torch.get_default_dtype() and integer.item() == 3.0


def test_safe_log_invalid_column_has_zero_gradient():
    values = torch.tensor([[0.0, 2.0], [1.0, 4.0]], requires_grad=True)
    logged = _safe_log(values)
    assert torch.isnan(logged[:, 0]).all() and torch.allclose(logged[:, 1], torch.log(torch.tensor([2.0, 4.0])))
    logged[:, 1].sum().backward()
    assert torch.isfinite(values.grad).all() and (values.grad[:, 0] == 0).all()


def test_ols_slope_degenerate_sweep_gives_zero_gradient():
    # A repeated, NaN-containing, or single-point sweep: the NaN slope must
    # leave zero, not NaN, gradients on both sweep and values.
    for raw in ([1.0, 1.0, 1.0], [1.0, float("nan"), 3.0], [1.0]):
        sweep = torch.tensor(raw, requires_grad=True)
        values = torch.arange(1.0, len(raw) + 1, requires_grad=True)
        slope = _ols_slope(sweep, values)
        assert torch.isnan(slope)
        sweep_gradient, values_gradient = torch.autograd.grad(slope, (sweep, values))
        assert (sweep_gradient == 0).all() and (values_gradient == 0).all()


def test_safe_log_integer_and_half_input():
    assert torch.allclose(_safe_log(torch.tensor([1, 4])), torch.log(torch.tensor([1.0, 4.0])))
    # log of half values near 1e4 cannot resolve a 1e-3 ratio; computed in float32.
    logged = _safe_log(torch.tensor([10000.0, 10008.0]).half())
    assert logged.dtype == torch.float32 and logged[1] > logged[0]


def test_ols_slope_keeps_close_float64_sweep_points_distinct():
    # float64 sweep points 1e-9 apart collapse in float32; the wider dtype is used.
    sweep = torch.tensor([1.0, 1.0 + 1e-9, 1.0 + 2e-9], dtype=torch.float64)
    values = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    slope = _ols_slope(sweep, values)
    assert slope.dtype == torch.float32 and torch.isclose(slope, torch.tensor(1e9), rtol=1e-4)


def test_slope_callers_zero_sweep_does_not_poison_shared_gradient():
    from torchnlts.dimension.higuchi import higuchi_dimension
    from torchnlts.dimension.variogram import variogram_dimension

    for function in (higuchi_dimension, variogram_dimension):
        parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        result = function(parameter * torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))
        assert torch.isnan(result)
        (gradient,) = torch.autograd.grad(torch.nan_to_num(result) + parameter, parameter)
        assert gradient.item() == 1.0
