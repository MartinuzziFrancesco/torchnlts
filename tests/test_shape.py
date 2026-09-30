import math

import numpy

import pytest
import torch

from torchnlts._shape import _ensure_batched, _ols_slope, _output_dtype, _restore_batch, _restore_shape_preserving, _safe_log, _work_dtype


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


def test_slope_callers_keep_sweep_precision_and_accept_integer_values():
    from torchnlts.dimension.correlation import correlation_dimension
    from torchnlts.dimension.generalized import generalized_dimension
    from torchnlts.lyapunov.rosenstein import rosenstein_lyapunov

    # Integer values with a fractional sweep: the sweep used to be cast to int.
    assert torch.isclose(correlation_dimension([0.25, 0.5, 1.0], torch.tensor([1, 2, 4])), torch.tensor(1.0))
    assert torch.isfinite(generalized_dimension([0.25, 0.5, 1.0], torch.tensor([1, 2, 4])))
    # float64 sweep points that float32 cannot separate (2**24 + 1).
    sweep = torch.tensor([16777216.0, 16777217.0, 16777218.0], dtype=torch.float64)
    assert torch.isclose(rosenstein_lyapunov(sweep, torch.tensor([0.0, 1.0, 2.0])), torch.tensor(1.0))


def test_work_dtype_promotes_bfloat16():
    assert _work_dtype(torch.bfloat16) == torch.float32
    assert _work_dtype(torch.float16) == torch.float32
    assert _work_dtype(torch.int64) == torch.get_default_dtype()
    assert _work_dtype(torch.float64) == torch.float64


def test_distance_based_functions_accept_integer_and_half_input():
    from torchnlts.dimension.correlation import correlation_sum, soft_correlation_sum
    from torchnlts.dimension.generalized import box_counting_entropy
    from torchnlts.lyapunov.rosenstein import rosenstein_divergence

    reference = torch.arange(8.0, dtype=torch.float64)
    for trajectory in (torch.arange(8), reference.half(), reference.bfloat16()):
        expected_dtype = _output_dtype(trajectory)
        for function in (lambda values: correlation_sum(values, 2), lambda values: soft_correlation_sum(values, 2), lambda values: rosenstein_divergence(values, [0, 1], embedding_dim=1)):
            result = function(trajectory)
            assert result.dtype == expected_dtype
            assert torch.allclose(result.double(), function(reference), atol=1e-2, equal_nan=True)
    # Integer input to box counting used to return a truncated int64 entropy.
    entropy = box_counting_entropy(torch.tensor([0, 1]), 1)
    assert entropy.is_floating_point() and torch.isclose(entropy.double(), torch.tensor(math.log(2), dtype=torch.float64))


def test_ols_slope_large_and_small_sweeps_float32():
    # The denominator squares the centered sweep: 1e20**2 overflows float32
    # and 1e-25**2 underflows to 0.
    for magnitude in (1e20, 1e-25):
        sweep = torch.tensor([1.0, 2.0, 3.0]) * magnitude
        assert torch.isclose(_ols_slope(sweep, sweep), torch.tensor(1.0))
        assert torch.isclose(_ols_slope(sweep, 2 * sweep), torch.tensor(2.0))


def test_complex_input_rejected_by_promoting_functions():
    from torchnlts.dimension.correlation import correlation_sum
    from torchnlts.dimension.generalized import box_counting_entropy
    from torchnlts.lyapunov.rosenstein import rosenstein_divergence

    trajectory = torch.tensor([0j, 1 + 1j, 2j, 3 + 0j, 1j, 2 + 2j])
    for function in (lambda values: correlation_sum(values, 1.0), lambda values: box_counting_entropy(values, 1.0), lambda values: rosenstein_divergence(values, [0, 1], embedding_dim=1)):
        with pytest.raises(ValueError):
            function(trajectory)


def test_ols_slope_scale_ratio_does_not_overflow():
    # Column 0 has values ~2**70 over a sweep ~2**-70; the ratio of the two
    # scales (2**140) overflows float32 even though its slope is exactly 0.
    parameter = torch.tensor(1.0, requires_grad=True)
    sweep = torch.tensor([1.0, 2.0, 3.0]) * 2.0**-70
    values = torch.stack([torch.tensor([1.0, -2.0, 1.0]) * 2.0**70, sweep], dim=1)
    result = _ols_slope(sweep, parameter * values)
    assert torch.allclose(result, torch.tensor([0.0, 1.0]))
    (gradient,) = torch.autograd.grad(result[1], parameter)
    assert torch.isclose(gradient, torch.tensor(1.0))


def test_ols_slope_close_points_far_from_zero():
    # 2**24 and 2**24 + 2 are exact in float32, their rounded mean is not.
    sweep = torch.tensor([16777216.0, 16777218.0], requires_grad=True)
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    result = _ols_slope(sweep, values)
    assert torch.isclose(result, torch.tensor(0.5))
    sweep_gradient, values_gradient = torch.autograd.grad(result, (sweep, values))
    assert torch.allclose(sweep_gradient, torch.tensor([0.25, -0.25]))
    assert torch.allclose(values_gradient, torch.tensor([-0.5, 0.5]))


def test_ols_slope_rejects_complex_and_keeps_empty_graph():
    with pytest.raises(ValueError):
        _ols_slope(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1 + 1j, 2 + 2j, 3 + 3j]))
    sweep = torch.empty(0, requires_grad=True)
    values = torch.empty(0, requires_grad=True)
    result = _ols_slope(sweep, values)
    assert torch.isnan(result)
    sweep_gradient, values_gradient = torch.autograd.grad(result, (sweep, values))
    assert sweep_gradient.shape == (0,) and values_gradient.shape == (0,)


def test_dtype_helpers_follow_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert _work_dtype(torch.int64) == torch.float64
        assert _output_dtype(torch.tensor([1, 2])) == torch.float64
    finally:
        torch.set_default_dtype(previous)
    assert _output_dtype(torch.tensor([1.0]).half()) == torch.float16
    for helper, argument in ((_work_dtype, torch.complex64), (_output_dtype, torch.tensor([1j]))):
        with pytest.raises(ValueError):
            helper(argument)


def test_pairwise_distance_callers_at_moderate_extremes():
    # cdist squares coordinates: 2**70 overflows and 2**-80 underflows float32.
    from torchnlts.dimension.correlation import correlation_sum
    from torchnlts.lyapunov.rosenstein import rosenstein_divergence

    for magnitude in (2.0**70, 2.0**-80):
        trajectory = torch.arange(1.0, 9.0) * magnitude
        assert torch.isclose(correlation_sum(trajectory, 1.5 * magnitude), torch.tensor(0.25))
        assert torch.isfinite(rosenstein_divergence(trajectory, [0, 1])).all()


def test_pairwise_distance_scale_is_per_row():
    # A shared batch scale would push the small row into float32 subnormals.
    from torchnlts.lyapunov.rosenstein import rosenstein_divergence
    from torchnlts.neighbors.distance import pairwise_distance

    small = torch.arange(1.0, 9.0) * 2.0**-60
    trajectory = torch.stack([torch.arange(1.0, 9.0) * 2.0**60, small])[..., None]
    distance = pairwise_distance(trajectory)
    assert torch.equal(distance[1], pairwise_distance(small[:, None]))
    batched = rosenstein_divergence(trajectory, [0, 1])
    assert torch.allclose(batched[:, 1], rosenstein_divergence(small, [0, 1]))
    assert torch.allclose(batched[:, 0] - batched[:, 1], torch.full((2,), 120 * math.log(2.0)))


def test_pairwise_distance_zero_norm_counts_unequal_coordinates():
    from torchnlts.lyapunov.rosenstein import rosenstein_divergence
    from torchnlts.neighbors.distance import pairwise_distance

    assert pairwise_distance(torch.tensor([1.0, 2.0, 4.0]), norm_order=0)[0, 1] == 1
    divergence = rosenstein_divergence(torch.tensor([1.0, 6.0, 2.0, 7.0, 13.0]), 0, embedding_dim=1, norm_order=0)
    assert divergence == 0
    for norm_order in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="norm_order"):
            pairwise_distance(torch.arange(3.0), norm_order=norm_order)


def test_ols_slope_of_constant_values_is_zero():
    sweep = torch.randn(13, generator=torch.Generator().manual_seed(0)).sort().values
    assert _ols_slope(sweep, torch.full((13,), 1e20)) == 0


def test_pairwise_distance_high_norm_order_does_not_overflow():
    from torchnlts.neighbors.distance import pairwise_distance

    assert pairwise_distance(torch.tensor([-3.0, 3.0]), norm_order=100)[0, 1] == 6


def test_pairwise_distance_max_norm_propagates_nan():
    from torchnlts.neighbors.distance import pairwise_distance

    distance = pairwise_distance(torch.tensor([[float("nan"), 1.0], [1.0, 2.0], [0.0, 0.0]]), norm_order=float("inf"))
    assert distance[0].isnan().all() and distance[:, 0].isnan().all() and distance[1, 2] == 2


def test_ols_slope_rejects_empty_complex_sweep():
    with pytest.raises(ValueError):
        _ols_slope(torch.empty(0, dtype=torch.complex64), torch.empty(0))


def test_pairwise_distance_norm_order_must_be_real_scalar():
    from torchnlts.neighbors.distance import pairwise_distance

    for norm_order in (1j, torch.tensor(1j), torch.tensor([1.0, 2.0]), numpy.complex128(2 + 3j), torch.tensor(2 + 0j), None):
        with pytest.raises(ValueError, match="norm_order"):
            pairwise_distance(torch.arange(3.0), norm_order=norm_order)
