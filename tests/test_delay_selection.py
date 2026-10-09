"""Tests for torchnlts.embedding.delay_selection.

Ground truth: hand-derived binary examples, the sine wave's quarter-period
minimum, StatsBase-style autocorrelation, and DelayEmbeddings.jl's
``selfmutualinfo`` / ``estimate_delay`` (see delay_selection.py for the
documented differences from ``selfmutualinfo``).
"""

import math
import subprocess

import pytest
import torch

from torchnlts.embedding.delay_selection import auto_mutual_information, autocorrelation, estimate_delay

JULIA_PROJECT = "juliaenv"


def _julia(code):
    result = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", "-e", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _julia_vector(trajectory, body):
    values = ", ".join(f"{v:.17g}" for v in trajectory.tolist())
    return _julia(f"using DelayEmbeddings\nx = [{values}]\n{body}")


def _parse(text):
    return [float(item) for item in text.strip("[]").split(",")]


def _sine(length=800, period=40):
    return torch.sin(2 * math.pi * torch.arange(length, dtype=torch.float64) / period)


def _mixed(length=200):
    time = torch.arange(length, dtype=torch.float64)
    return torch.sin(2 * math.pi * time / 20) + 0.3 * torch.sin(2 * math.pi * time / 7)


def _binary_entropy(prob):
    return -(prob * math.log(prob) + (1 - prob) * math.log(1 - prob))


def test_autocorrelation_hand_derived():
    # mean 2, centered [-1, 0, 1]: c(1) = 0 / 2, c(2) = -1 / 2
    result = autocorrelation(torch.tensor([1.0, 2.0, 3.0]), 2)
    assert torch.allclose(result, torch.tensor([1.0, 0.0, -0.5]))


def test_autocorrelation_gradient_is_finite_and_nonzero():
    trajectory = torch.randn(30, dtype=torch.float64, requires_grad=True)
    autocorrelation(trajectory, 3)[2].backward()
    assert torch.isfinite(trajectory.grad).all() and trajectory.grad.abs().sum() > 0


def test_ami_hand_derived_alternating_sequence():
    # 7 pairs; delay 0 and 1 both determine one coordinate from the other, so
    # I equals the entropy of the marginal (4/7, 3/7).
    trajectory = torch.tensor([0.0, 1.0] * 4, dtype=torch.float64)
    result = auto_mutual_information(trajectory, 1, num_bins=2)
    assert torch.allclose(result, torch.full((2,), _binary_entropy(4 / 7), dtype=torch.float64))


def test_ami_independent_pattern_is_zero():
    # Pairs (0,0),(0,1),(1,0),(1,1) each once: exactly independent.
    trajectory = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    # pairs at delay 1 over the first 4 samples: (0,0),(0,1),(1,1),(1,0)
    result = auto_mutual_information(trajectory, 1, num_bins=2)
    assert abs(result[1].item()) < 1e-12


def test_ami_sine_minimum_plateau_contains_quarter_period():
    # Given x_t, a sine's x_{t+k} takes two values (about one bit of
    # information lost) once k passes the first ramp, so the minimum is a
    # plateau: the quarter period (10) sits on it and the first local
    # minimum is detected inside it.
    ami = auto_mutual_information(_sine(), 30)
    assert ami[10] <= ami[1:].min() + 0.035
    assert ami[10] < ami[1] - 0.35
    assert 6 <= estimate_delay(_sine(), 30, "mi_min").item() <= 10


def test_estimate_delay_autocorrelation_methods_on_sine():
    sine = _sine()
    assert abs(estimate_delay(sine, 30, "ac_zero").item() - 10) <= 1
    assert abs(estimate_delay(sine, 30, "ac_min").item() - 20) <= 1
    # cos(2 pi k / 40) < 1/e first at k = 10 * 0.69... -> 8 (arccos(1/e) = 1.1944 rad)
    assert estimate_delay(sine, 30, "ac_1/e").item() == math.ceil(1.1944 / (2 * math.pi) * 40)


def test_estimate_delay_hand_derived_search_rules():
    # Autocorrelation of this series is not needed: use the rules via a crafted series.
    # Strictly decaying exponential: no zero crossing -> max_delay; 1/e found.
    decay = torch.exp(-torch.arange(100, dtype=torch.float64) / 10)
    assert estimate_delay(decay, 5, "ac_zero").item() == 5
    assert estimate_delay(decay, 5, "ac_min").item() == 5  # no minimum found: max_delay


def test_ami_matches_julia_at_delay_zero_and_periodic_window():
    # Pair window T - max_delay = 200 = 10 periods, so the delayed marginal
    # equals the original one and both implementations agree exactly.
    torch.manual_seed(0)
    trajectory = torch.randn(20, dtype=torch.float64).repeat(11)[:215]
    expected = _parse(_julia_vector(trajectory, "println(selfmutualinfo(x, 0:15; nbins=8))"))
    actual = auto_mutual_information(trajectory, 15, num_bins=8)
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64) * math.log(2), atol=1e-12, rtol=0)


def test_ami_julia_deviation_is_locked_in_off_periodic_windows():
    # Delayed marginal counted from the delayed samples (see module docstring):
    # delay 0 is exact, other delays differ by a small amount.
    trajectory = _mixed()
    expected = torch.tensor(
        _parse(_julia_vector(trajectory, "println(selfmutualinfo(x, 0:15; nbins=8))")), dtype=torch.float64
    ) * math.log(2)
    actual = auto_mutual_information(trajectory, 15, num_bins=8)
    assert abs(actual[0] - expected[0]) < 1e-12
    difference = (actual - expected).abs().max().item()
    assert 0 < difference < 0.02, difference


def test_estimate_delay_matches_julia_for_each_method():
    trajectory = _mixed()
    julia = _julia_vector(
        trajectory, 'println([estimate_delay(x, m, 1:30; nbins=8) for m in ["mi_min", "ac_zero", "ac_min"]])'
    )
    expected = [int(value) for value in _parse(julia)]
    actual = [estimate_delay(trajectory, 30, method, num_bins=8).item() for method in ("mi_min", "ac_zero", "ac_min")]
    assert actual == expected


def test_autocorrelation_matches_julia():
    trajectory = _mixed()
    expected = _parse(_julia_vector(trajectory, "using StatsBase\nprintln(autocor(x, 0:20; demean=true))"))
    assert torch.allclose(
        autocorrelation(trajectory, 20), torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_shapes_agree_and_batch_matches_loop():
    rows = torch.stack([_mixed(), _sine(200)])
    batched = auto_mutual_information(rows[..., None], 10)
    assert batched.shape == (2, 11)
    for index in range(2):
        assert torch.allclose(batched[index], auto_mutual_information(rows[index], 10))
        assert torch.allclose(auto_mutual_information(rows[index][:, None], 10), batched[index])
    assert torch.equal(estimate_delay(rows[..., None], 20), torch.stack([estimate_delay(row, 20) for row in rows]))
    assert autocorrelation(rows[..., None], 10).shape == (2, 11)


def test_nonfinite_row_is_nan_and_does_not_poison_others():
    rows = torch.stack([_mixed(), _mixed()])
    rows[1, 7] = float("nan")
    ami = auto_mutual_information(rows[..., None], 10)
    corr = autocorrelation(rows[..., None], 10)
    assert torch.isnan(ami[1]).all() and torch.isnan(corr[1]).all()
    assert torch.allclose(ami[0], auto_mutual_information(rows[0], 10))
    assert estimate_delay(rows[..., None], 20, "mi_min").tolist()[1] == -1
    assert estimate_delay(rows[..., None], 20, "ac_zero")[0] >= 1


def test_constant_row_autocorrelation_is_nan():
    assert torch.isnan(autocorrelation(torch.ones(20), 3)).all()
    assert all(
        estimate_delay(torch.ones(20), 5, method).item() == -1 for method in ("mi_min", "ac_zero", "ac_min", "ac_1/e")
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: auto_mutual_information(torch.randn(20, 2), 3),
        lambda: auto_mutual_information(torch.randn(20), 0),
        lambda: auto_mutual_information(torch.randn(20), 20),
        lambda: auto_mutual_information(torch.randn(20), 3, num_bins=1),
        lambda: autocorrelation(torch.randn(20), 2.5),
        lambda: estimate_delay(torch.randn(20), 3, "nope"),
    ],
)
def test_invalid_arguments_raise(call):
    with pytest.raises(ValueError):
        call()


def test_vmap_compatible():
    rows = torch.stack([_mixed(), _sine(200)])
    mapped = torch.vmap(lambda row: autocorrelation(row, 5))(rows)
    assert torch.allclose(mapped, autocorrelation(rows[..., None], 5))


def test_estimate_delay_hand_derived_ties_and_zero():
    # AMI of an alternating series is identical at every delay: a tie counts as
    # "not decreasing", so the first delay is returned.
    alternating = torch.tensor([0.0, 1.0] * 8, dtype=torch.float64)
    assert estimate_delay(alternating, 4, "mi_min", num_bins=2).item() == 1
    # autocorrelation [1, 0, -0.5]: exactly zero at delay 1 counts as a crossing.
    assert estimate_delay(torch.tensor([1.0, 2.0, 3.0]), 2, "ac_zero").item() == 1


def test_ami_range_uses_only_the_pair_window():
    # Pair window is the first 7 samples (max_delay = 2); the outliers at the
    # end must not stretch the bin edges.
    trajectory = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 100.0, 100.0], dtype=torch.float64)
    assert abs(auto_mutual_information(trajectory, 2, num_bins=2)[0].item() - _binary_entropy(4 / 7)) < 1e-12


@pytest.mark.parametrize("scale", [1e20, 1e-25])
def test_autocorrelation_is_scale_invariant_in_float32(scale):
    trajectory = torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float32) * scale
    assert torch.allclose(autocorrelation(trajectory, 2), torch.tensor([1.0, 0.15, -0.5]), atol=1e-6)


def test_autocorrelation_large_offset_float32():
    # Regression: subtracting the mean of [2**24, 2**24 + 2] in float32 lost the variation.
    trajectory = torch.tensor([2.0**24, 2.0**24 + 2], dtype=torch.float32)
    assert torch.allclose(autocorrelation(trajectory, 1), torch.tensor([1.0, -0.5]))


def test_autocorrelation_gradcheck():
    trajectory = torch.randn(12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: autocorrelation(values, 3), (trajectory,))


def test_autocorrelation_gradient_isolated_from_nonfinite_row():
    valid = torch.randn(15, dtype=torch.float64)
    bad = valid.clone()
    bad[3] = float("nan")
    rows = torch.stack([valid, bad])[..., None].requires_grad_()
    torch.nan_to_num(autocorrelation(rows, 3)).sum().backward()
    standalone = valid.clone().requires_grad_()
    autocorrelation(standalone, 3).sum().backward()
    assert torch.isfinite(rows.grad).all()
    assert torch.allclose(rows.grad[0, :, 0], standalone.grad)


@pytest.mark.parametrize("method", ["ac_min", "ac_zero", "ac_1/e", "mi_min"])
def test_estimate_delay_half_precision_matches_float32(method):
    # Regression: intermediate curves were rounded to half precision.
    values = [
        0.0992431640625,
        0.70166015625,
        0.357421875,
        1.6484375,
        0.708984375,
        1.701171875,
        0.5830078125,
        -0.1846923828125,
        -1.6748046875,
        -0.7333984375,
        0.289794921875,
        -0.0281219482421875,
    ]
    half = torch.tensor(values, dtype=torch.float16)
    assert (
        estimate_delay(half, 6, method, num_bins=4).item() == estimate_delay(half.float(), 6, method, num_bins=4).item()
    )
    assert estimate_delay(half, 6, "ac_min").item() == 6


def test_estimate_delay_short_search_matches_julia():
    trajectory = torch.tensor([0.0, 1.0] * 4, dtype=torch.float64)
    julia = _julia_vector(trajectory, 'println(estimate_delay(x, "mi_min", 1:2; nbins=2))')
    assert int(julia) == 1 == estimate_delay(trajectory, 2, "mi_min", num_bins=2).item()


def test_estimate_delay_ignores_minimum_at_penultimate_delay():
    sine = _sine()
    minimum = estimate_delay(sine, 30, "ac_min").item()
    assert estimate_delay(sine, minimum + 1, "ac_min").item() == minimum + 1
    assert estimate_delay(sine, minimum + 2, "ac_min").item() == minimum


def test_estimate_delay_threshold_is_one_over_e():
    # cos(2 pi k / 70) is about 0.394 at k = 13 (above 1/e, below 0.4) and 0.309 at k = 14.
    assert estimate_delay(_sine(2000, 70), 30, "ac_1/e").item() == 14


def test_ami_sample_on_interior_edge_goes_to_lower_bin():
    # Edge at 0.5: the 0.5 sample is not above it, so bins are {0, 0.5} / {1, 1} and I(delay 0) = ln 2.
    trajectory = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0], dtype=torch.float64)
    assert abs(auto_mutual_information(trajectory, 1, num_bins=2)[0].item() - math.log(2)) < 1e-12


def test_estimate_delay_mi_min_half_uses_float32_curve():
    # Regression: rounding the AMI curve to half before the search changes the selected delay.
    torch.manual_seed(17)
    series = torch.randn(1000, 20).half()[..., None]
    expected = estimate_delay(series.float(), 10, "mi_min", 4)
    assert torch.equal(estimate_delay(series, 10, "mi_min", 4), expected)


def test_estimate_delay_short_search_fallbacks():
    increasing = torch.arange(1.0, 9.0)
    assert estimate_delay(increasing, 1, "ac_min").item() == 1
    assert estimate_delay(increasing, 2, "ac_min").item() == 2
