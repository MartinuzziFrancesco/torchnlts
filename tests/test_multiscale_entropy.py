"""Tests for torchnlts.entropy.multiscale.

Dual ground truth:
  1. Paper: Costa, Goldberger & Peng (2002), the (unnumbered)
     coarse-graining formula on p. 2 (their own eq. 1 is Shannon
     entropy) -- hand-derived coarse-graining checks, plus the identity
     that scale=1 must recover plain sample_entropy exactly (the
     composition's own correctness, not a re-test of sample_entropy
     itself).
  2. Julia: ComplexityMeasures.jl's `multiscale(RegularDownsampling,
     SampleEntropy, x)`. Its SampleEntropy has the same documented
     boundary quirk as tests/test_sample_entropy.py's own dual
     validation (a leaked out-of-range B(r) neighbor candidate for
     delay=1); neutralized here the same way, at every scale tested,
     since coarse-graining still leaves the padding values as
     far outliers relative to any radius used.
"""

import pytest
import torch

from tests.conftest import assert_first_row_gradients_unaffected, run_julia
from torchnlts.entropy.multiscale import (
    coarse_grain,
    composite_multiscale_entropy,
    multiscale_entropy,
    soft_composite_multiscale_entropy,
    soft_multiscale_entropy,
)
from torchnlts.entropy.sample import sample_entropy, soft_sample_entropy


def test_coarse_grain_hand_derived():
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    expected_scale_2 = torch.tensor([1.5, 3.5, 5.5], dtype=torch.float64)
    expected_scale_3 = torch.tensor([2.0, 5.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 2), expected_scale_2)
    assert torch.allclose(coarse_grain(trajectory, 3), expected_scale_3)


def test_coarse_grain_scale_1_is_identity():
    torch.manual_seed(0)
    trajectory = torch.randn(20, dtype=torch.float64)
    assert torch.equal(coarse_grain(trajectory, 1), trajectory)


def test_coarse_grain_drops_trailing_remainder():
    # 7 points at scale=3: floor(7/3)=2 windows, last point dropped.
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 3), torch.tensor([2.0, 5.0], dtype=torch.float64))


def test_coarse_grain_preserves_shape_convention():
    torch.manual_seed(1)
    scalar = torch.randn(30, dtype=torch.float64)
    multivariate = torch.randn(30, 4, dtype=torch.float64)
    batched = torch.randn(2, 30, 4, dtype=torch.float64)
    assert coarse_grain(scalar, 5).shape == (6,)
    assert coarse_grain(multivariate, 5).shape == (6, 4)
    assert coarse_grain(batched, 5).shape == (2, 6, 4)


def test_coarse_grain_rejects_non_positive_scale():
    trajectory = torch.randn(10, dtype=torch.float64)
    for bad_scale in (0, -1):
        with pytest.raises(ValueError):
            coarse_grain(trajectory, bad_scale)


def test_multiscale_entropy_scale_1_matches_sample_entropy():
    torch.manual_seed(2)
    trajectory = torch.randn(100, dtype=torch.float64)
    radius = 0.15 * trajectory.std()
    curve = multiscale_entropy(trajectory, scales=[1, 5], radius=radius, embedding_dim=2)
    assert torch.isclose(curve[0], sample_entropy(trajectory, radius, embedding_dim=2))


def test_multiscale_entropy_shape_batched_and_swept():
    torch.manual_seed(3)
    trajectory = torch.randn(2, 100, 1, dtype=torch.float64)
    radius = torch.tensor([0.3, 0.6], dtype=torch.float64)
    curve = multiscale_entropy(trajectory, scales=[1, 2, 3], radius=radius, embedding_dim=2)
    assert curve.shape == (3, 2, 2)
    for scale_idx, scale in enumerate((1, 2, 3)):
        expected = sample_entropy(coarse_grain(trajectory, scale), radius, embedding_dim=2)
        assert torch.allclose(curve[scale_idx], expected, equal_nan=True)


def test_soft_multiscale_entropy_gradcheck():
    torch.manual_seed(4)
    trajectory = torch.randn(20, dtype=torch.float64)
    temperature = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_multiscale_entropy(trajectory, scales=[1, 2], temperature=t, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (temperature,))


def test_soft_multiscale_entropy_gradcheck_trajectory():
    # Gradient through coarse_grain's reshape/mean, which the temperature-only gradcheck above does not cover.
    torch.manual_seed(6)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def entropy(t):
        return soft_multiscale_entropy(t, scales=[1, 2, 3], temperature=0.4, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory,))


def test_multiscale_entropy_too_coarse_scale_raises():
    # Coarse series shorter than embedding_dim+1 raises (from delay_embedding); a degenerate but embeddable scale gives NaN.
    torch.manual_seed(7)
    trajectory = torch.randn(20, dtype=torch.float64)
    embedding_dim = 2

    # scale=6: coarse length 3 == embedding_dim+1, still embeddable.
    result = multiscale_entropy(trajectory, scales=[6], radius=1.0, embedding_dim=embedding_dim)
    assert result.shape == (1,)

    # scale=7 -> coarse length 2 < embedding_dim+1: too short, must raise.
    with pytest.raises(ValueError):
        multiscale_entropy(trajectory, scales=[7], radius=1.0, embedding_dim=embedding_dim)


def test_multiscale_entropy_matches_complexitymeasures_jl():
    # Same ComplexityMeasures.jl SampleEntropy boundary deviation as test_sample_entropy.py: pad with far outliers
    # that stay far after coarse-graining at every scale swept.
    torch.manual_seed(5)
    trajectory = torch.cat([torch.randn(40, dtype=torch.float64), torch.tensor([1e6, 2e6], dtype=torch.float64)])
    radius = 1.0
    embedding_dim = 2
    scales = [1, 2, 3]

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = SampleEntropy(m={embedding_dim}, τ=1, r={radius})
    ds = RegularDownsampling(scales={scales[0]}:{scales[-1]})
    println(join(multiscale(ds, est, x), ","))
    """
    stdout = run_julia(julia_code)
    julia_curve = torch.tensor([float(v) for v in stdout.strip().split(",")], dtype=torch.float64)
    torch_curve = multiscale_entropy(trajectory, scales=scales, radius=radius, embedding_dim=embedding_dim, delay=1)
    assert torch.allclose(torch_curve, julia_curve, atol=1e-9)


def test_coarse_grain_offset_hand_derived():
    trajectory = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    assert torch.allclose(coarse_grain(trajectory, 3, offset=0), torch.tensor([2.0, 5.0], dtype=torch.float64))
    assert torch.allclose(coarse_grain(trajectory, 3, offset=1), torch.tensor([3.0, 6.0], dtype=torch.float64))
    assert torch.allclose(coarse_grain(trajectory, 3, offset=2), torch.tensor([4.0], dtype=torch.float64))
    assert torch.equal(coarse_grain(trajectory, 3), coarse_grain(trajectory, 3, offset=0))


def test_coarse_grain_rejects_bad_offset():
    trajectory = torch.randn(10, dtype=torch.float64)
    for bad_offset in (-1, 3):
        with pytest.raises(ValueError):
            coarse_grain(trajectory, 3, offset=bad_offset)


def test_composite_multiscale_entropy_hand_derived():
    # Period-3 signal at scale 2: both offset series are period-3, so A = B and each SampEn = 0.
    trajectory = torch.tensor([0.0, 2.0, 4.0] * 8, dtype=torch.float64)[:23]
    assert torch.equal(coarse_grain(trajectory, 2, offset=0)[:3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.equal(coarse_grain(trajectory, 2, offset=1)[:3], torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(
        composite_multiscale_entropy(trajectory, scales=[2], radius=0.5), torch.zeros(1, dtype=torch.float64)
    )
    perturbed = trajectory.clone()
    perturbed[10] += 1.7
    perturbed[3] -= 1.2
    per_offset = [sample_entropy(coarse_grain(perturbed, 2, offset=offset), 0.5) for offset in range(2)]
    assert per_offset[0] != per_offset[1] and all(torch.isfinite(value) for value in per_offset)
    assert torch.isclose(composite_multiscale_entropy(perturbed, scales=[2], radius=0.5)[0], sum(per_offset) / 2)


def test_composite_multiscale_entropy_scale_1_matches_sample_entropy():
    torch.manual_seed(2)
    trajectory = torch.randn(100, dtype=torch.float64)
    radius = 0.15 * trajectory.std()
    curve = composite_multiscale_entropy(trajectory, scales=[1, 4], radius=radius)
    assert torch.isclose(curve[0], sample_entropy(trajectory, radius))
    expected = torch.stack(
        [sample_entropy(coarse_grain(trajectory, 4, offset=offset), radius) for offset in range(4)]
    ).mean()
    assert torch.isclose(curve[1], expected)


def test_composite_multiscale_entropy_shape_batched_and_swept():
    torch.manual_seed(3)
    trajectory = torch.randn(2, 100, 1, dtype=torch.float64)
    radius = torch.tensor([0.3, 0.6], dtype=torch.float64)
    curve = composite_multiscale_entropy(trajectory, scales=[1, 2, 3], radius=radius)
    assert curve.shape == (3, 2, 2)
    for scale_idx, scale in enumerate((1, 2, 3)):
        expected = torch.stack(
            [sample_entropy(coarse_grain(trajectory, scale, offset=k), radius) for k in range(scale)]
        ).mean(0)
        assert torch.allclose(curve[scale_idx], expected, equal_nan=True)
    assert composite_multiscale_entropy(trajectory[0, :, 0], scales=[2], radius=0.3).shape == (1,)


def test_composite_multiscale_entropy_nonfinite_rows_and_propagation():
    torch.manual_seed(8)
    batch = torch.randn(3, 60, 1, dtype=torch.float64)
    batch[1, 5] = float("nan")
    curve = composite_multiscale_entropy(batch, scales=[1, 3], radius=0.5)
    # Regression: a non-finite row behaves exactly as under sample_entropy at scale 1.
    assert torch.allclose(curve[0], sample_entropy(batch, 0.5), equal_nan=True)
    for row in (0, 2):
        assert torch.equal(curve[:, row], composite_multiscale_entropy(batch[row], scales=[1, 3], radius=0.5))
    # An inf offset entropy (matches at length 2, none at length 3) is not excluded: plain mean.
    inf_series = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 9.0, 9.0, 9.0], dtype=torch.float64)
    assert torch.isinf(sample_entropy(inf_series, 0.5))
    doubled = inf_series.repeat_interleave(2)
    assert torch.equal(coarse_grain(doubled, 2, offset=0), inf_series)
    assert not torch.isfinite(composite_multiscale_entropy(doubled, scales=[2], radius=0.5)).any()


def test_composite_multiscale_entropy_too_coarse_scale_raises():
    trajectory = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError):
        composite_multiscale_entropy(trajectory, scales=[7], radius=1.0)


def test_soft_composite_multiscale_entropy_matches_composition():
    torch.manual_seed(9)
    trajectory = torch.randn(25, dtype=torch.float64)
    curve = soft_composite_multiscale_entropy(trajectory, scales=[1, 3], temperature=0.5)
    expected = torch.stack([soft_sample_entropy(coarse_grain(trajectory, 3, offset=k), 0.5) for k in range(3)]).mean()
    assert torch.isclose(curve[1], expected)


def test_soft_composite_multiscale_entropy_gradcheck():
    torch.manual_seed(10)
    trajectory = torch.randn(24, dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def entropy(series, bandwidth):
        return soft_composite_multiscale_entropy(series, scales=[1, 2, 3], temperature=bandwidth, embedding_dim=2)

    assert torch.autograd.gradcheck(entropy, (trajectory, temperature))


def test_composite_multiscale_entropy_matches_complexitymeasures_jl():
    # Same boundary deviation as above. Far outliers sit in the last window of every offset series at scales 1-3;
    # N = 41 = 5 mod 6 makes Julia's common truncation equal every offset's natural length.
    torch.manual_seed(11)
    trajectory = torch.cat([torch.randn(38, dtype=torch.float64), torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64)])
    radius = 1.0
    scales = [1, 2, 3]
    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ComplexityMeasures
    x = [{values}]
    est = SampleEntropy(m=2, τ=1, r={radius})
    ds = CompositeDownsampling(scales={scales[0]}:{scales[-1]})
    println(join(multiscale(ds, est, x), ","))
    """
    stdout = run_julia(julia_code)
    julia_curve = torch.tensor([float(v) for v in stdout.strip().split(",")], dtype=torch.float64)
    torch_curve = composite_multiscale_entropy(trajectory, scales=scales, radius=radius)
    assert torch.allclose(torch_curve, julia_curve, atol=1e-9)


def test_composite_multiscale_entropy_nan_offset_propagates():
    # Regression: one offset NaN (no template matches) and another finite gives NaN, not the finite value.
    torch.manual_seed(1)
    trajectory = torch.randn(24, dtype=torch.float64)
    per_offset = [sample_entropy(coarse_grain(trajectory, 2, offset=offset), 0.3) for offset in range(2)]
    assert torch.isfinite(per_offset[0]) and torch.isnan(per_offset[1])
    assert torch.isnan(composite_multiscale_entropy(trajectory, scales=[2], radius=0.3)).all()


def test_composite_multiscale_entropy_forwards_nondefault_arguments():
    torch.manual_seed(12)
    trajectory = torch.randn(80, dtype=torch.float64)
    options = dict(embedding_dim=3, delay=2, theiler_window=2, norm_order=1.0)
    curve = composite_multiscale_entropy(trajectory, scales=[3], radius=1.2, **options)
    expected = torch.stack(
        [sample_entropy(coarse_grain(trajectory, 3, offset=k), 1.2, **options) for k in range(3)]
    ).mean()
    assert torch.isclose(curve[0], expected)
    default = composite_multiscale_entropy(trajectory, scales=[3], radius=1.2)
    assert not torch.isclose(curve[0], default[0])


def test_soft_composite_multiscale_entropy_forwards_nondefault_arguments():
    torch.manual_seed(13)
    trajectory = torch.randn(80, dtype=torch.float64)
    options = dict(embedding_dim=3, delay=2, kind="exponential", theiler_window=2, norm_order=1.0)
    curve = soft_composite_multiscale_entropy(trajectory, scales=[3], temperature=0.8, **options)
    expected = torch.stack(
        [soft_sample_entropy(coarse_grain(trajectory, 3, offset=k), 0.8, **options) for k in range(3)]
    ).mean()
    assert torch.isclose(curve[0], expected)
    default = soft_composite_multiscale_entropy(trajectory, scales=[3], temperature=0.8)
    assert not torch.isclose(curve[0], default[0])


def test_composite_multiscale_entropy_validates_scales_and_offset():
    trajectory = torch.randn(40, dtype=torch.float64)
    for bad_scales in ([], [0], [-1], [2.0], [1.5]):
        for function, argument in ((composite_multiscale_entropy, 0.5), (soft_composite_multiscale_entropy, 0.5)):
            with pytest.raises(ValueError):
                function(
                    trajectory,
                    scales=bad_scales,
                    **{("radius" if function is composite_multiscale_entropy else "temperature"): argument},
                )
    for bad_offset in (0.5, 1.0):
        with pytest.raises(ValueError):
            coarse_grain(trajectory, 3, offset=bad_offset)


def test_coarse_grain_and_composite_dtype_handling():
    integers = torch.arange(12)
    assert coarse_grain(integers, 2).dtype == torch.get_default_dtype()
    assert torch.allclose(coarse_grain(integers, 2, offset=1), torch.tensor([1.5, 3.5, 5.5, 7.5, 9.5]))
    torch.manual_seed(14)
    half = torch.randn(60).to(torch.float16)
    curve = composite_multiscale_entropy(half, scales=[1, 2], radius=0.5)
    assert curve.dtype == torch.float16
    reference = composite_multiscale_entropy(half.float(), scales=[1, 2], radius=0.5)
    assert torch.allclose(curve.float(), reference.to(torch.float16).float(), equal_nan=True)
    assert composite_multiscale_entropy(torch.arange(30) % 5, scales=[2], radius=0.5).dtype == torch.get_default_dtype()


def test_soft_composite_multiscale_entropy_shared_input_backward_is_finite():
    # Regression: a batch row whose kernel sums underflow to 0 must not put NaN into gradients of an input shared with other rows.
    torch.manual_seed(1)
    base = torch.randn(30, dtype=torch.float64, requires_grad=True)
    batch = torch.stack([base, base + 100 * torch.arange(30)]).unsqueeze(-1)
    temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    curve = soft_composite_multiscale_entropy(batch, [2], temperature)
    assert torch.isnan(curve[0, 1])
    grad_base, grad_temperature = torch.autograd.grad(curve[0, 0], (base, temperature))
    assert torch.isfinite(grad_base).all() and torch.isfinite(grad_temperature)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("function, parameter", [(multiscale_entropy, 0.3), (soft_multiscale_entropy, 0.05)])
def test_multiscale_entropy_half_computed_in_float32(function, parameter, dtype):
    torch.manual_seed(1)
    series = torch.randn(2, 300, 1).to(dtype)
    curve = function(series, [1, 2, 3], parameter)
    assert curve.dtype == dtype and torch.isfinite(curve).all()
    assert torch.equal(curve, function(series.float(), [1, 2, 3], parameter).to(dtype))


def test_composite_multiscale_entropy_half_computed_in_float32():
    torch.manual_seed(0)
    series = torch.randn(60).half()
    expected = composite_multiscale_entropy(series.float(), [1, 2], 0.5).half()
    assert torch.equal(composite_multiscale_entropy(series, [1, 2], 0.5), expected)
    for scales in (None, 2):
        for function in (composite_multiscale_entropy, soft_composite_multiscale_entropy):
            with pytest.raises(ValueError):
                function(series, scales, 0.5)


@pytest.mark.parametrize("bad_scales", [[], [0], [-1], [2.0], [1.5], None, 2])
@pytest.mark.parametrize(
    "function, argument",
    [
        (multiscale_entropy, "radius"),
        (soft_multiscale_entropy, "temperature"),
        (composite_multiscale_entropy, "radius"),
        (soft_composite_multiscale_entropy, "temperature"),
    ],
)
def test_every_multiscale_function_validates_scales(function, argument, bad_scales):
    # Regression: an empty `scales` reached torch.stack and raised a RuntimeError.
    with pytest.raises(ValueError):
        function(torch.randn(40, dtype=torch.float64), bad_scales, **{argument: 0.5})


def test_multiscale_entropy_accepts_any_iterable_of_scales():
    torch.manual_seed(15)
    trajectory = torch.randn(60, dtype=torch.float64)
    expected = multiscale_entropy(trajectory, [1, 2, 3], 0.5)
    assert torch.equal(multiscale_entropy(trajectory, range(1, 4), 0.5), expected)
    assert torch.equal(multiscale_entropy(trajectory, (scale for scale in (1, 2, 3)), 0.5), expected)
    assert torch.equal(
        soft_multiscale_entropy(trajectory, torch.tensor([1, 2, 3]), 0.5),
        soft_multiscale_entropy(trajectory, [1, 2, 3], 0.5),
    )


ALL_MULTISCALE = [
    multiscale_entropy,
    soft_multiscale_entropy,
    composite_multiscale_entropy,
    soft_composite_multiscale_entropy,
]


@pytest.mark.parametrize("function", ALL_MULTISCALE)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_multiscale_functions_nan_for_nonfinite_sample_dropped_by_coarse_graining(function, bad):
    # Regression: at scale 3, coarse_grain drops sample 60 of 61, so a bad value there used to leave the row finite.
    torch.manual_seed(0)
    batch = torch.randn(3, 61, 1, dtype=torch.float64)
    batch[1, 60] = bad
    curve = function(batch, [1, 3], 0.5)
    assert curve.shape == (2, 3) and torch.isnan(curve[:, 1]).all() and torch.isfinite(curve[:, [0, 2]]).all()
    for row in (0, 2):
        assert torch.equal(curve[:, row], function(batch[row], [1, 3], 0.5))
    assert torch.isnan(function(batch[1, :, 0], [1, 3], 0.5)).all()


@pytest.mark.parametrize("function", ALL_MULTISCALE)
def test_multiscale_functions_nan_for_a_nonfinite_sample_at_any_position(function):
    torch.manual_seed(1)
    series = torch.randn(61, dtype=torch.float64)
    assert not torch.isnan(function(series, [3], 1.0)).any()
    for position in range(61):
        bad = series.clone()
        bad[position] = float("nan")
        assert torch.isnan(function(bad, [3], 1.0)).all(), position


@pytest.mark.parametrize("function", [soft_multiscale_entropy, soft_composite_multiscale_entropy])
def test_soft_multiscale_functions_nonfinite_sample_leaves_other_gradients(function):
    # Regression: the bad row moved the shared temperature's gradient (dropped sample) or put NaN into it (composite).
    torch.manual_seed(2)
    clean, bad_row = torch.randn(61, dtype=torch.float64), torch.randn(61, dtype=torch.float64)
    bad_row[60] = float("nan")

    def gradients(rows):
        batch = torch.stack(rows).unsqueeze(-1).requires_grad_()
        temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        function(batch, [1, 3], temperature).nansum().backward()
        return batch.grad, temperature.grad

    batch_grad, temperature_grad = gradients([bad_row, clean])
    single_grad, single_temperature_grad = gradients([clean])
    assert torch.isfinite(batch_grad).all() and (batch_grad[0] == 0).all()
    assert torch.allclose(batch_grad[1], single_grad[0])
    assert torch.isfinite(temperature_grad) and torch.allclose(temperature_grad, single_temperature_grad)


@pytest.mark.parametrize("function", ALL_MULTISCALE)
def test_multiscale_functions_vmap_with_nonfinite_sample_dropped_by_coarse_graining(function):
    torch.manual_seed(0)
    batch = torch.randn(3, 61, 1, dtype=torch.float64)
    batch[1, 60] = float("nan")
    direct = function(batch, [1, 3], 0.5)
    mapped = torch.vmap(lambda series: function(series, [1, 3], 0.5))(batch)
    assert torch.allclose(direct, mapped.movedim(0, -1), equal_nan=True) and torch.isnan(mapped[1]).all()


@pytest.mark.parametrize("function", [soft_multiscale_entropy, soft_composite_multiscale_entropy])
@pytest.mark.parametrize(
    "spread_series",
    [[10.0 * index**2 for index in range(12)], [1.374 * index for index in range(12)]],
    ids=["underflowed_to_zero", "denormal_in_float32"],
)
def test_soft_multiscale_functions_degenerate_kernel_sums_keep_other_gradients_finite(function, spread_series):
    small = [0.00, 0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.06, -0.01, 0.00, 0.04, -0.03]
    result = assert_first_row_gradients_unaffected(
        lambda series, temperature: function(series, [1, 2], temperature), [small, spread_series], 0.1, torch.float32
    )
    assert torch.isfinite(result[:, 0]).all()
