"""Tests for torchnlts.lyapunov.rosenstein.

Ground truth: a hand-derived 5-point divergence curve; an exactly linear ``E(k)`` for the slope fit alone; the logistic
map at r = 4, whose largest Lyapunov exponent is exactly ln(2); ChaosTools.jl's ``lyapunov_from_data`` at float64
precision, including its reference set fixed by the largest ``k``; and plain-torch loops for delays and norms.
"""

import math

import numpy
import pytest
import torch

from tests.conftest import logistic_map_trajectory, run_julia
from torchnlts.lyapunov.rosenstein import rosenstein_divergence, rosenstein_lyapunov


def test_rosenstein_divergence_hand_derived():
    # Nearest neighbors (k=1): 0->2, 1->3, 2->0, 3->1; i=4 has no k=1 step.
    # Divergences 1, 11, 1, 11, so E(1) = mean(log(1, 11, 1, 11)).
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, delay=1, theiler_window=0)
    expected = (math.log(1) + math.log(11) + math.log(1) + math.log(11)) / 4
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_rosenstein_divergence_sweep_uses_max_k_reference_set_throughout():
    # Reference set is fixed by max_k=2 for both entries (ChaosTools.jl convention): only i=0 and i=2 qualify.
    # E(1) = mean(log(1), log(1)) = 0, E(2) = log(11); both entries are checked, not just the last.
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    k_values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    swept = rosenstein_divergence(trajectory, k_values, embedding_dim=1, delay=1, theiler_window=0)
    expected = torch.tensor([0.0, math.log(11)], dtype=torch.float64)
    assert torch.allclose(swept, expected)

    expected_at_max_k = rosenstein_divergence(trajectory, k_values=2.0, embedding_dim=1, delay=1, theiler_window=0)
    assert torch.isclose(swept[-1], expected_at_max_k)


def test_rosenstein_divergence_batched_and_sweep_shapes():
    # Each batch row's sweep must match calling that trajectory alone.
    torch.manual_seed(0)
    trajectory = torch.randn(4, 40, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    swept = rosenstein_divergence(trajectory, k_values, embedding_dim=2, theiler_window=1)
    assert swept.shape == (3, 4)
    for i in range(4):
        expected = rosenstein_divergence(trajectory[i], k_values, embedding_dim=2, theiler_window=1)
        assert expected.shape == (3,)
        assert torch.allclose(swept[:, i], expected, equal_nan=True)
    expected_at_max_k = rosenstein_divergence(trajectory, k_values=5.0, embedding_dim=2, theiler_window=1)
    assert torch.allclose(swept[-1], expected_at_max_k, equal_nan=True)


def test_rosenstein_divergence_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 30, 2, dtype=torch.float64)
    batched = rosenstein_divergence(trajectory, k_values=2, embedding_dim=2)
    unbatched = rosenstein_divergence(trajectory[0], k_values=2, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_rosenstein_divergence_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=2)
    assert torch.isnan(result)


def test_rosenstein_divergence_k_too_large_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=10, embedding_dim=1, theiler_window=0)
    assert torch.isnan(result)


def test_rosenstein_divergence_zero_evolved_distance_is_neg_inf_not_clamped():
    # Revisited value gives a zero distance, so log(0) = -inf as in ChaosTools.jl (not clamped).
    trajectory = torch.tensor([0.0, 5.0, 0.0, 5.0, 0.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=0)
    assert torch.isneginf(result)


def test_rosenstein_divergence_rejects_negative_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=-1, embedding_dim=1)


def test_rosenstein_divergence_rejects_fractional_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=1.5, embedding_dim=1)


def test_rosenstein_divergence_rejects_empty_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=torch.tensor([]), embedding_dim=1)


def test_rosenstein_divergence_rejects_multidim_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=torch.tensor([[1.0, 2.0]]), embedding_dim=1)


def test_rosenstein_divergence_rejects_infinite_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=float("inf"), embedding_dim=1)


def test_rosenstein_divergence_k_validation_survives_float32_rounding():
    # -1e-50 and 1.00000001 would pass the checks if cast to float32 before validation.
    trajectory = torch.arange(5.0, dtype=torch.float64)
    for bad_k in (-1e-50, 1.00000001):
        with pytest.raises(ValueError):
            rosenstein_divergence(trajectory, k_values=bad_k, embedding_dim=1)


def test_rosenstein_divergence_absurdly_large_k_is_nan_not_wrong_answer():
    # A huge k must not wrap around in the int64 bounds check.
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=9223372036854775807, embedding_dim=1)
    assert torch.isnan(result)


def test_rosenstein_divergence_rejects_negative_theiler_window():
    # A negative window would admit self as its own nearest neighbor.
    trajectory = torch.arange(5.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        rosenstein_divergence(trajectory, k_values=1, embedding_dim=1, theiler_window=-1)


def test_rosenstein_divergence_k_zero_is_original_neighbor_distance():
    # k=0: E(0) is the log of each nearest-neighbor distance; all 5 points are valid, including i=4 (distance 6).
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    result = rosenstein_divergence(trajectory, k_values=0, embedding_dim=1, theiler_window=0)
    expected = (math.log(1) * 4 + math.log(6)) / 5
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_rosenstein_divergence_gradcheck():
    # Away from a nearest-neighbor tie (verified for this seed).
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def divergence(x):
        return rosenstein_divergence(x, k_values=2, embedding_dim=2, delay=1, theiler_window=0)

    assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_rosenstein_lyapunov_hand_derived_exact_linear():
    # E(k) = 3*k + 1 exactly, so slope 3.
    k_values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    divergence = 3.0 * k_values + 1.0
    assert torch.isclose(rosenstein_lyapunov(k_values, divergence), torch.tensor(3.0, dtype=torch.float64))


def test_rosenstein_lyapunov_two_points_is_exact_slope():
    k_values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    divergence = torch.tensor([0.5, 1.3], dtype=torch.float64)
    expected = (divergence[1] - divergence[0]) / (k_values[1] - k_values[0])
    assert torch.isclose(rosenstein_lyapunov(k_values, divergence), expected)


def test_rosenstein_lyapunov_batched_shape():
    torch.manual_seed(2)
    trajectory = torch.randn(3, 40, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, theiler_window=1)
    lyapunov = rosenstein_lyapunov(k_values, divergence)
    assert lyapunov.shape == (3,)
    for i in range(3):
        expected = rosenstein_lyapunov(
            k_values, rosenstein_divergence(trajectory[i], k_values, embedding_dim=2, theiler_window=1)
        )
        assert torch.isclose(lyapunov[i], expected, equal_nan=True)


def test_rosenstein_lyapunov_single_k_is_nan_not_crash():
    result = rosenstein_lyapunov(torch.tensor([1.0]), torch.tensor([0.5]))
    assert torch.isnan(result)


def test_rosenstein_lyapunov_identical_k_is_nan_not_crash():
    result = rosenstein_lyapunov(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.2, 0.5, 0.3]))
    assert torch.isnan(result)


def test_rosenstein_lyapunov_gradcheck_direct_k_and_divergence():
    k_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    divergence = torch.tensor([0.1, 0.5, 0.9, 1.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rosenstein_lyapunov, (k_values, divergence))


def test_rosenstein_matches_analytic_logistic_map_lyapunov_exponent():
    # Logistic map x -> 4x(1-x) has largest Lyapunov exponent ln(2); finite-sample estimate, so a loose tolerance.
    trajectory = logistic_map_trajectory(2000)
    k_values = torch.arange(1, 12, dtype=torch.float64)
    divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, delay=1, theiler_window=10)
    # Fit range [1, 7] is the linear-growth region before E(k) saturates.
    estimate = rosenstein_lyapunov(k_values[:7], divergence[:7])
    assert abs(estimate.item() - math.log(2)) < 0.05


def test_rosenstein_matches_chaostools_jl():
    # Same exact neighbor search and max(k)-sized fixed reference set as ChaosTools.jl, so the curves agree closely.
    trajectory = logistic_map_trajectory(300)
    k_values = torch.arange(1, 9, dtype=torch.float64)

    values = ", ".join(f"{v:.17g}" for v in trajectory.numpy())
    julia_code = f"""
    using ChaosTools
    using DelayEmbeddings
    x = [{values}]
    R = embed(x, 2, 1)
    E = lyapunov_from_data(R, 1:8; w=10, ntype=NeighborNumber(1), distance=Euclidean())
    println(join(E, ","))
    """
    stdout = run_julia(julia_code)
    julia_divergence = torch.tensor([float(v) for v in stdout.strip().split(",")], dtype=torch.float64)
    torch_divergence = rosenstein_divergence(trajectory, k_values, embedding_dim=2, delay=1, theiler_window=10)
    # rtol=0: allclose's default rtol=1e-5 would dwarf atol=1e-9 and hide a real mismatch.
    assert torch.allclose(torch_divergence, julia_divergence, atol=1e-9, rtol=0)

    julia_slope = rosenstein_lyapunov(k_values, julia_divergence)
    torch_slope = rosenstein_lyapunov(k_values, torch_divergence)
    assert torch.isclose(julia_slope, torch_slope, atol=1e-9, rtol=0)


def test_rosenstein_divergence_out_of_range_preserves_shape_and_dtype():
    for dtype in (torch.float32, torch.float64):
        for shape in ((5,), (5, 1), (1, 5, 1), (2, 5, 1)):
            trajectory = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
            for large_k in (5, 2**63 - 1, 2**64, 10**30, 1e300):
                for k_values, normal_k in ((large_k, 0), ([0, large_k], [0, 1])):
                    result = rosenstein_divergence(trajectory, k_values, embedding_dim=1)
                    normal = rosenstein_divergence(trajectory, normal_k, embedding_dim=1)
                    assert result.shape == normal.shape
                    assert result.dtype == normal.dtype == dtype
                    assert result.device == normal.device
                    assert torch.isnan(result).all()


def test_rosenstein_divergence_large_theiler_window_excludes_self():
    trajectory = torch.arange(5.0, dtype=torch.float64).reshape(1, 5, 1)
    for window in (2**63 - 1, 2**63, 2**64, 10**30):
        result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1, theiler_window=window)
        assert result.shape == (2, 1)
        assert torch.isnan(result).all()


def test_rosenstein_divergence_one_copy_embedding_ignores_large_delay():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    expected = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1)
    result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=1, delay=10**30)
    assert torch.equal(result, expected)


def test_rosenstein_divergence_embedding_and_validation_edges():
    trajectory = torch.arange(5.0, dtype=torch.float32).reshape(1, 5, 1)
    for options in (
        {"embedding_dim": 0},
        {"delay": 0},
        {"embedding_dim": 4, "delay": 2},
        {"embedding_dim": 2, "delay": 10**30},
    ):
        with pytest.raises(ValueError):
            rosenstein_divergence(trajectory, 0, **options)
    for bad_k in (-1e-50, 1.00000001, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            rosenstein_divergence(trajectory, [0, bad_k])
    result = rosenstein_divergence(trajectory, [0, 1], embedding_dim=3, delay=2)
    assert result.shape == (2, 1)
    assert result.dtype == trajectory.dtype
    assert torch.isnan(result).all()


def test_rosenstein_divergence_matches_independent_loop_for_delays_and_norms():
    torch.manual_seed(31)
    trajectory = torch.randn(2, 18, 2, dtype=torch.float64)
    k_values = [2, 0, 1, 2]
    for embedding_dim, delay in ((1, 1), (2, 3), (3, 2)):
        for norm_order in (1.0, 2.0, float("inf")):
            expected = []
            for batch in trajectory:
                embedded = [
                    torch.cat([batch[start + offset * delay] for offset in range(embedding_dim)])
                    for start in range(len(batch) - (embedding_dim - 1) * delay)
                ]
                logs = [[] for _ in k_values]
                for start, point in enumerate(embedded):
                    candidates = [index for index in range(len(embedded)) if abs(index - start) > 4]
                    neighbor = min(candidates, key=lambda index: (point - embedded[index]).norm(p=norm_order).item())
                    nearest_distance = (point - embedded[neighbor]).norm(p=norm_order)
                    assert (
                        sum((point - embedded[index]).norm(p=norm_order) == nearest_distance for index in candidates)
                        == 1
                    )
                    if max(start, neighbor) + max(k_values) >= len(embedded):
                        continue
                    for position, step in enumerate(k_values):
                        logs[position].append(
                            (embedded[start + step] - embedded[neighbor + step]).norm(p=norm_order).log()
                        )
                expected.append(torch.stack([torch.stack(values).mean() for values in logs]))
            result = rosenstein_divergence(
                trajectory, k_values, embedding_dim=embedding_dim, delay=delay, theiler_window=4, norm_order=norm_order
            )
            assert torch.allclose(result, torch.stack(expected, dim=1), atol=1e-12, rtol=0)


def test_rosenstein_divergence_batched_sweep_gradcheck():
    torch.manual_seed(32)
    trajectory = torch.randn(1, 12, 2, dtype=torch.float64, requires_grad=True)
    for norm_order in (1.0, 2.0, float("inf")):

        def divergence(values):
            return rosenstein_divergence(values, [0, 2], embedding_dim=2, delay=2, norm_order=norm_order)

        assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_rosenstein_divergence_small_separation_amplifies_gradient():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64, requires_grad=True)
    result = rosenstein_divergence(trajectory, 0, embedding_dim=1)
    gradient = torch.autograd.grad(result, trajectory)[0]
    small = (trajectory.detach() * 1e-12).requires_grad_()
    small_result = rosenstein_divergence(small, 0, embedding_dim=1)
    small_gradient = torch.autograd.grad(small_result, small)[0]
    assert torch.allclose(small_gradient * 1e-12, gradient, atol=1e-12, rtol=0)
    assert small_gradient.abs().max() > 1e11


def test_rosenstein_divergence_one_duplicate_pair_is_neg_inf():
    trajectory = torch.tensor([0.0, 0.0, 4.0, 10.0, 17.0], dtype=torch.float64)
    assert torch.isneginf(rosenstein_divergence(trajectory, 0, embedding_dim=1))


def test_rosenstein_lyapunov_rejects_mismatched_sweep_shapes():
    for k_values, divergence in (
        ([0, 1, 2], torch.tensor([7.0])),
        ([[0, 1]], torch.tensor([1.0, 2.0])),
        ([0, 1], torch.ones(2, 1, 1)),
        (0, torch.tensor(1.0)),
    ):
        with pytest.raises(ValueError, match="matching"):
            rosenstein_lyapunov(k_values, divergence)


def test_rosenstein_zero_distance_row_does_not_poison_shared_gradient():
    # Row 0 is constant: curve -inf, slope NaN; log(0)'s infinite gradient must not reach row 1.
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    trajectories = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 1.0, 6.0, 12.0]], dtype=torch.float64).unsqueeze(
        -1
    )
    curve = rosenstein_divergence(parameter * trajectories, [0, 1], embedding_dim=1)
    assert torch.isinf(curve[:, 0]).all()
    result = rosenstein_lyapunov([0, 1], curve)
    assert torch.isnan(result[0]) and torch.isfinite(result[1])
    (gradient,) = torch.autograd.grad(result[1], parameter)
    assert torch.isclose(gradient, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_rosenstein_divergence_rejects_complex_k_values():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0])
    for k_values in (torch.tensor([0.0 + 2j, 1.0 + 3j]), [0, 1j]):
        with pytest.raises(ValueError, match="complex"):
            rosenstein_divergence(trajectory, k_values)


def test_rosenstein_divergence_high_norm_order_small_separations():
    # (1e-15 / 10) ** 40 underflows float64 unless each pair is scaled by itself.
    ulp = math.ulp(10.0)
    trajectory = torch.tensor(
        [[0.0, 0.0], [1e-15, 1e-15], [10.0, 10.0], [10.0 + 2 * ulp, 10.0 + 2 * ulp]], dtype=torch.float64
    )
    expected = (math.log(1e-15) + math.log(2 * ulp)) / 2 + math.log(2.0) / 40
    assert math.isclose(
        rosenstein_divergence(trajectory, 0, embedding_dim=1, norm_order=40).item(), expected, rel_tol=1e-12
    )
    # Separations 1e-15 and 1 in one row: a row-wide scale still underflows.
    trajectory = torch.tensor([[0.0, 0.0], [1e-15, 1e-15], [10.0, 10.0], [11.0, 11.0]], dtype=torch.float64)
    expected = math.log(1e-15) / 2 + math.log(2.0) / 40
    assert math.isclose(
        rosenstein_divergence(trajectory, 0, embedding_dim=1, norm_order=40).item(), expected, rel_tol=1e-12
    )


def test_rosenstein_divergence_accepts_numpy_norm_order():
    trajectory = torch.arange(5.0)
    for norm_order in (numpy.float32(2), numpy.float64(2)):
        assert rosenstein_divergence(trajectory, 0, embedding_dim=1, norm_order=norm_order) == 0


def test_rosenstein_divergence_accepts_any_integer_valued_k_sequence():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0, 3.0, 8.0, 2.0, 9.0, 4.0], dtype=torch.float64)
    expected = rosenstein_divergence(trajectory, [0, 1, 2], embedding_dim=1)
    assert torch.isfinite(expected).all()
    for k_values in (numpy.array([0, 1, 2]), numpy.array([0.0, 1.0, 2.0]), torch.tensor([0, 1, 2]), (0, 1, 2)):
        assert torch.equal(rosenstein_divergence(trajectory, k_values, embedding_dim=1), expected)
    expected_scalar = rosenstein_divergence(trajectory, 1, embedding_dim=1)
    for k_values in (numpy.int64(1), numpy.array(1), torch.tensor(1), 1.0):
        scalar = rosenstein_divergence(trajectory, k_values, embedding_dim=1)
        assert scalar.shape == () and torch.equal(scalar, expected_scalar)


def test_rosenstein_divergence_single_element_sweep_keeps_its_axis():
    torch.manual_seed(15)
    trajectory = torch.randn(3, 30, 1, dtype=torch.float64)
    assert rosenstein_divergence(trajectory, [2], embedding_dim=2).shape == (1, 3)
    assert rosenstein_divergence(trajectory[0], [2], embedding_dim=2).shape == (1,)
    assert rosenstein_divergence(trajectory[:1], [2], embedding_dim=2).shape == (1, 1)
    assert rosenstein_divergence(trajectory[:1], 2, embedding_dim=2).shape == (1,)
    assert rosenstein_divergence(trajectory[0], 2, embedding_dim=2).shape == ()


def test_rosenstein_divergence_k_values_error_conditions():
    trajectory = torch.arange(8.0)
    bad_values = (
        torch.tensor([[1, 2]]),
        [],
        [0, 1.5],
        [0, -1],
        [0, float("nan")],
        [0, float("inf")],
        "a",
        None,
        [0, 1j],
    )
    for k_values in bad_values:
        with pytest.raises(ValueError, match="k_values"):
            rosenstein_divergence(trajectory, k_values, embedding_dim=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rosenstein_divergence_half_precision_is_computed_in_float32_and_cast_back(dtype):
    torch.manual_seed(17)
    trajectory = torch.randn(60, dtype=torch.float32).to(dtype)
    result = rosenstein_divergence(trajectory, [1, 2], embedding_dim=2)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    expected = rosenstein_divergence(trajectory.float(), [1, 2], embedding_dim=2).to(dtype)
    assert torch.equal(result, expected)
