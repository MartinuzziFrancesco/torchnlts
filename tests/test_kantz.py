"""Tests for torchnlts.lyapunov.kantz.

Dual ground truth:
  1. Paper: Kantz (1994) -- a hand-derived divergence example (log of
     the mean neighborhood distance, computed by hand and independently
     re-derived as "mean of logs" to precisely pin down the Julia
     disagreement below) plus the same natural analytic benchmark used
     for Rosenstein: the logistic map's known ln(2) Lyapunov exponent.
  2. Julia: ChaosTools.jl's `lyapunov_from_data` with `ntype=WithinRange`
     is a confirmed, deliberate mismatch, not a match -- reading its
     source shows it averages the *log* of each neighbor's distance
     (mean of logs), not the log of the *mean* distance the cited paper
     defines. Locked in as a disagreement regression below, same
     playbook as sample_entropy/mutual_information/approximate_entropy's
     own confirmed Julia deviations.
"""

import math
import subprocess

import torch

from tests.conftest import logistic_map_trajectory
from torchnlts.lyapunov.kantz import kantz_divergence
from torchnlts.lyapunov.rosenstein import rosenstein_divergence, rosenstein_lyapunov

JULIA_PROJECT = "juliaenv"


def test_kantz_divergence_hand_derived():
    # embedding_dim=1, radius=1.5, k=1: neighbors i0:{1} i1:{0,2} i2:{1} i3:{4} i4:{3} (5 has no k=1 step, dropped).
    # Evolved mean distances 1, 4.5, 8, 1, 1, so S(1) = mean(log(1), log(4.5), log(8), log(1), log(1)).
    trajectory = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=1, radius=1.5, embedding_dim=1, delay=1, theiler_window=0)
    expected = (math.log(1) + math.log(4.5) + math.log(8) + math.log(1) + math.log(1)) / 5
    assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64))


def test_kantz_divergence_batched_and_sweep_shapes():
    torch.manual_seed(0)
    trajectory = torch.randn(4, 40, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    swept = kantz_divergence(trajectory, k_values, radius=1.0, embedding_dim=2, theiler_window=1)
    assert swept.shape == (3, 4)
    for i in range(4):
        expected = kantz_divergence(trajectory[i], k_values, radius=1.0, embedding_dim=2, theiler_window=1)
        assert expected.shape == (3,)
        assert torch.allclose(swept[:, i], expected, equal_nan=True)


def test_kantz_divergence_batch_size_one_matches_unbatched():
    torch.manual_seed(1)
    trajectory = torch.randn(1, 30, 2, dtype=torch.float64)
    batched = kantz_divergence(trajectory, k_values=2, radius=1.0, embedding_dim=2)
    unbatched = kantz_divergence(trajectory[0], k_values=2, radius=1.0, embedding_dim=2)
    assert batched.shape == (1,)
    assert torch.isclose(batched[0], unbatched)


def test_kantz_divergence_no_neighbors_in_radius_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 100.0, 200.0, 300.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=1, radius=0.5, embedding_dim=1, theiler_window=0)
    assert torch.isnan(result)


def test_kantz_divergence_theiler_window_excludes_everything_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=1, radius=10.0, embedding_dim=1, theiler_window=2)
    assert torch.isnan(result)


def test_kantz_divergence_k_too_large_is_nan_not_crash():
    trajectory = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=10, radius=1.5, embedding_dim=1, theiler_window=0)
    assert torch.isnan(result)


def test_kantz_divergence_absurdly_large_k_is_nan_not_wrong_answer():
    trajectory = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=9223372036854775807, radius=1.5, embedding_dim=1)
    assert torch.isnan(result)


def test_kantz_divergence_zero_mean_distance_is_neg_inf_not_clamped():
    # All-duplicate neighborhood: mean distance 0 gives -inf, as in the paper's formula.
    trajectory = torch.tensor([0.0, 5.0, 0.0, 5.0, 0.0], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=1, radius=0.5, embedding_dim=1, theiler_window=0)
    assert torch.isneginf(result)


def test_kantz_divergence_rejects_negative_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=-1, radius=1.0, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_rejects_fractional_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=1.5, radius=1.0, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_rejects_empty_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=torch.tensor([]), radius=1.0, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_rejects_multidim_k_values():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=torch.tensor([[1.0, 2.0]]), radius=1.0, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_rejects_infinite_k():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=float("inf"), radius=1.0, embedding_dim=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_k_validation_survives_float32_rounding():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    for bad_k in (-1e-50, 1.00000001):
        try:
            kantz_divergence(trajectory, k_values=bad_k, radius=1.0, embedding_dim=1)
            assert False, f"expected ValueError for k_values={bad_k}"
        except ValueError:
            pass


def test_kantz_divergence_rejects_negative_theiler_window():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=1, radius=1.0, embedding_dim=1, theiler_window=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_delay_0_still_validated_at_embedding_dim_1():
    # The large-delay OverflowError workaround must not swallow an invalid delay.
    trajectory = torch.arange(5.0, dtype=torch.float64)
    try:
        kantz_divergence(trajectory, k_values=0, radius=1.0, embedding_dim=1, delay=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kantz_divergence_one_copy_embedding_ignores_large_delay():
    trajectory = torch.tensor([0.0, 5.0, 1.0, 6.0, 12.0], dtype=torch.float64)
    expected = kantz_divergence(trajectory, [0, 1], radius=10.0, embedding_dim=1)
    result = kantz_divergence(trajectory, [0, 1], radius=10.0, embedding_dim=1, delay=10**30)
    assert torch.equal(result, expected)


def test_kantz_divergence_large_theiler_window_excludes_everything():
    trajectory = torch.arange(5.0, dtype=torch.float64)
    for window in (2**63 - 1, 2**63, 2**64, 10**30):
        result = kantz_divergence(trajectory, k_values=[0, 1], radius=10.0, embedding_dim=1, theiler_window=window)
        assert torch.isnan(result).all()


def test_kantz_divergence_dtype_preserved_float32():
    torch.manual_seed(7)
    trajectory = torch.randn(20, dtype=torch.float32)
    result = kantz_divergence(trajectory, k_values=2, radius=1.0, embedding_dim=2)
    assert result.dtype == torch.float32


def test_kantz_divergence_batch_size_one_sweep_matches_unbatched():
    torch.manual_seed(8)
    trajectory = torch.randn(1, 30, 2, dtype=torch.float64)
    k_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    batched = kantz_divergence(trajectory, k_values, radius=1.0, embedding_dim=2)
    unbatched = kantz_divergence(trajectory[0], k_values, radius=1.0, embedding_dim=2)
    assert batched.shape == (3, 1)
    assert torch.allclose(batched[:, 0], unbatched, equal_nan=True)


def test_kantz_divergence_single_element_sweep_matches_scalar():
    torch.manual_seed(9)
    trajectory = torch.randn(20, dtype=torch.float64)
    swept = kantz_divergence(trajectory, k_values=[2], radius=1.0, embedding_dim=2)
    scalar = kantz_divergence(trajectory, k_values=2, radius=1.0, embedding_dim=2)
    assert swept.shape == (1,)
    assert torch.isclose(swept[0], scalar)


def test_kantz_divergence_inclusive_radius_boundary():
    # radius equals the neighbor distance: neighborhoods are inclusive (<=).
    trajectory = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=torch.float64)
    at_boundary = kantz_divergence(trajectory, k_values=1, radius=1.0, embedding_dim=1, delay=1, theiler_window=0)
    above_boundary = kantz_divergence(trajectory, k_values=1, radius=1.5, embedding_dim=1, delay=1, theiler_window=0)
    assert torch.isclose(at_boundary, above_boundary)


def test_kantz_divergence_mixed_zero_and_positive_neighborhood_rows():
    # Rows 0 and 1 are duplicates giving -inf; it must dominate the average, not be dropped.
    trajectory = torch.tensor([0.0, 0.0, 4.0, 4.1, 4.2], dtype=torch.float64)
    result = kantz_divergence(trajectory, k_values=0, radius=0.5, embedding_dim=1, theiler_window=0)
    assert torch.isneginf(result)


def test_kantz_divergence_matches_independent_loop_for_delays_and_norms():
    torch.manual_seed(31)
    trajectory = torch.randn(2, 18, 2, dtype=torch.float64)
    k_values = [2, 0, 1, 2]
    radius = 2.0
    theiler_window = 3

    def oracle_mean(values):
        return torch.stack(values).mean() if values else torch.tensor(float("nan"), dtype=torch.float64)

    for embedding_dim, delay in ((1, 1), (2, 3), (3, 2)):
        for norm_order in (1.0, 2.0, float("inf")):
            expected = []
            for batch in trajectory:
                embedded = [
                    torch.cat([batch[start + offset * delay] for offset in range(embedding_dim)])
                    for start in range(len(batch) - (embedding_dim - 1) * delay)
                ]
                num_points = len(embedded)
                num_usable = max(0, num_points - max(k_values))
                logs = [[] for _ in k_values]
                for i in range(num_usable):
                    neighbors = [
                        j
                        for j in range(num_usable)
                        if j != i
                        and abs(i - j) > theiler_window
                        and (embedded[i] - embedded[j]).norm(p=norm_order) <= radius
                    ]
                    if not neighbors:
                        continue
                    for position, step in enumerate(k_values):
                        distances = [(embedded[i + step] - embedded[j + step]).norm(p=norm_order) for j in neighbors]
                        logs[position].append(torch.stack(distances).mean().log())
                expected.append(torch.stack([oracle_mean(values) for values in logs]))
            result = kantz_divergence(trajectory, k_values, radius, embedding_dim, delay, theiler_window, norm_order)
            assert torch.allclose(result, torch.stack(expected, dim=1), atol=1e-12, rtol=0, equal_nan=True)


def test_kantz_divergence_gradcheck():
    torch.manual_seed(5)
    trajectory = torch.randn(20, dtype=torch.float64, requires_grad=True)

    def divergence(x):
        return kantz_divergence(x, k_values=2, radius=1.0, embedding_dim=2, delay=1, theiler_window=0)

    assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_kantz_divergence_batched_sweep_gradcheck():
    # radius=3.0 keeps the curve finite under all norms; mixed valid and excluded rows exercise the masked gradient.
    torch.manual_seed(6)
    trajectory = torch.randn(1, 12, 2, dtype=torch.float64, requires_grad=True)
    for norm_order in (1.0, 2.0, float("inf")):

        def divergence(x, norm_order=norm_order):
            return kantz_divergence(x, [0, 2], radius=3.0, embedding_dim=2, delay=2, norm_order=norm_order)

        assert torch.autograd.gradcheck(divergence, (trajectory,))


def test_kantz_matches_analytic_logistic_map_lyapunov_exponent():
    # Logistic map: the largest Lyapunov exponent is ln(2).
    trajectory = logistic_map_trajectory(2000)
    k_values = torch.arange(1, 12, dtype=torch.float64)
    divergence = kantz_divergence(trajectory, k_values, radius=0.01, embedding_dim=2, delay=1, theiler_window=10)
    estimate = rosenstein_lyapunov(k_values[:6], divergence[:6])
    assert abs(estimate.item() - math.log(2)) < 0.05


def test_kantz_divergence_disagrees_with_chaostools_jl():
    # ChaosTools.jl averages the log of each neighbor distance, not the log of the mean distance as in the paper.
    # If ChaosTools switches convention this fails and the documented discrepancy needs revisiting.
    trajectory = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    julia_code = f"""
    using ChaosTools
    using DelayEmbeddings
    x = {trajectory}
    R = embed(x, 1, 1)
    E = lyapunov_from_data(R, 1:1; w=0, ntype=WithinRange(1.5), distance=Euclidean())
    println(E[1])
    """
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", julia_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    julia_divergence = torch.tensor(float(result.stdout.strip()), dtype=torch.float64)
    torch_divergence = kantz_divergence(
        torch.tensor(trajectory, dtype=torch.float64),
        k_values=1,
        radius=1.5,
        embedding_dim=1,
        delay=1,
        theiler_window=0,
    )
    # Mean-of-logs (Julia): (log 1 + (log 1 + log 8)/2 + log 8 + log 1 + log 1)/5;
    # log-of-mean (this module): (log 1 + log 4.5 + log 8 + log 1 + log 1)/5.
    julia_expected = (math.log(1) + (math.log(1) + math.log(8)) / 2 + math.log(8) + math.log(1) + math.log(1)) / 5
    torch_expected = (math.log(1) + math.log(4.5) + math.log(8) + math.log(1) + math.log(1)) / 5
    # Default isclose tolerance: expected values are analytic constants, so only float64 round-trip noise remains.
    assert torch.isclose(julia_divergence, torch.tensor(julia_expected, dtype=torch.float64))
    assert torch.isclose(torch_divergence, torch.tensor(torch_expected, dtype=torch.float64))
    # atol=0.05: the statistics differ by ~0.09 (Jensen's inequality), so the gap is real and stable.
    assert not torch.isclose(torch_divergence, julia_divergence, atol=0.05)


def test_kantz_zero_distance_row_keeps_shared_gradient():
    # Regression: a -inf row's infinite log gradient reached a parameter shared with other rows.
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    for norm_order in (1.0, 2.0, float("inf")):
        parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        result = kantz_divergence(
            (values * parameter)[..., None], 0, radius=1.1, embedding_dim=1, norm_order=norm_order
        )
        assert result[1] == float("-inf")
        gradient = torch.autograd.grad(result[0], parameter)[0]
        parameter_alone = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        alone = kantz_divergence(values[0] * parameter_alone, 0, radius=1.1, embedding_dim=1, norm_order=norm_order)
        assert torch.allclose(gradient, torch.autograd.grad(alone, parameter_alone)[0])


def test_kantz_invalid_arguments_raise_value_error():
    import pytest

    series = torch.arange(10.0)
    for override in (
        {"k_values": torch.tensor(1 + 2j)},
        {"radius": -1.0},
        {"radius": float("nan")},
        {"embedding_dim": 1.5},
        {"delay": 1.5},
    ):
        arguments = dict(k_values=1, radius=2.0, embedding_dim=1)
        arguments.update(override)
        with pytest.raises(ValueError):
            kantz_divergence(series, **arguments)


def test_kantz_radius_and_theiler_window_validation():
    import pytest

    series = torch.arange(5.0)
    for radius in ("2", 10**400):
        with pytest.raises(ValueError):
            kantz_divergence(series, 0, radius, embedding_dim=1)
    assert kantz_divergence(series, 0, torch.tensor([[[2.0]]]), embedding_dim=1).shape == ()
    for window in (1 + 0j, torch.tensor(1 + 0j), torch.tensor([1, 2]), None, 1.5):
        with pytest.raises(ValueError):
            kantz_divergence(series, 0, 2.0, embedding_dim=1, theiler_window=window)
        with pytest.raises(ValueError):
            rosenstein_divergence(series, 0, embedding_dim=1, theiler_window=window)
