"""Tests for torchnlts.chaos.zero_one.

Dual ground truth:
  1. Julia: ChaosTools.jl's ``testchaos01(x, c, N0)`` per-frequency K_c on
     the r=4 logistic map and a quasi-periodic series. ChaosTools starts its
     sums at p(1) = 0 and never uses the last sample, so it is given
     ``[x[2:N]; x[1]]``, which makes it agree with this module's paper-form
     convention exactly (see the module docstring).
  2. Paper: a hand-derived 3-sample case, a brute-force evaluation of the
     paper's definitions, and the logistic-map results of Gottwald and
     Melbourne (2009): K near 1 for chaos, K near 0 for the periodic
     r = 3.55 window.
"""

import math

import numpy
import pytest
import torch

from tests.conftest import logistic_map_trajectory, run_julia
from torchnlts.chaos.zero_one import zero_one_growth_rate, zero_one_test

FREQUENCIES = torch.linspace(math.pi / 5, 4 * math.pi / 5, 7, dtype=torch.float64)
SWEEP = torch.linspace(math.pi / 5, 4 * math.pi / 5, 100, dtype=torch.float64)


def _logistic(parameter: float, num_points: int) -> torch.Tensor:
    value, values = 0.4, []
    for step in range(500 + num_points):
        value = parameter * value * (1 - value)
        if step >= 500:
            values.append(value)
    return torch.tensor(values, dtype=torch.float64)


def _oracle(series: list[float], frequency: float, max_lag: int) -> float:
    """Paper's definitions evaluated term by term in 70-digit arithmetic."""
    mpmath = pytest.importorskip("mpmath")
    with mpmath.workdps(70):
        count = len(series)
        values = [mpmath.mpf(value) for value in series]
        freq = mpmath.mpf(frequency)
        trans_cos, trans_sin = [mpmath.mpf(0)], [mpmath.mpf(0)]  # index 0 unused by the sums below
        for index, value in enumerate(values, start=1):
            trans_cos.append(trans_cos[-1] + value * mpmath.cos(index * freq))
            trans_sin.append(trans_sin[-1] + value * mpmath.sin(index * freq))
        mean = sum(values) / count
        modified = []
        for lag in range(1, max_lag + 1):
            msd = sum(
                (trans_cos[j + lag] - trans_cos[j]) ** 2 + (trans_sin[j + lag] - trans_sin[j]) ** 2
                for j in range(1, count - lag + 1)
            ) / (count - lag)
            modified.append(msd - mean**2 * (1 - mpmath.cos(lag * freq)) / (1 - mpmath.cos(freq)))
        lags = [mpmath.mpf(lag) for lag in range(1, max_lag + 1)]
        lag_mean, modified_mean = sum(lags) / max_lag, sum(modified) / max_lag
        covariance = sum((lag - lag_mean) * (item - modified_mean) for lag, item in zip(lags, modified))
        spread = mpmath.sqrt(
            sum((lag - lag_mean) ** 2 for lag in lags) * sum((item - modified_mean) ** 2 for item in modified)
        )
        return float(covariance / spread)


def _julia_growth_rates(series: torch.Tensor, frequencies: torch.Tensor, max_lag: int) -> torch.Tensor:
    shifted = torch.cat([series[1:], series[:1]])
    julia_code = f"""
    using ChaosTools
    series = {shifted.tolist()}
    for frequency in {frequencies.tolist()}
        println(ChaosTools.testchaos01(series, frequency, {max_lag}))
    end
    """
    return torch.tensor([float(line) for line in run_julia(julia_code).split()], dtype=torch.float64)


def test_hand_derived_three_samples():
    # x=[1,2,3], c=pi/2, max_lag=2: p=[0,-2,-2], q=[1,1,-2]; M(1)=8.5, M(2)=13, mean=2, V_osc=4, 8,
    # so D=[4.5, 5]: two points, K_c=+1.
    trajectory = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert torch.isclose(
        zero_one_growth_rate(trajectory, math.pi / 2, max_lag=2), torch.tensor(1.0, dtype=torch.float64)
    )


def test_matches_high_precision_paper_definitions():
    series = _logistic(3.9, 60)
    result = zero_one_growth_rate(series, FREQUENCIES, max_lag=6)
    expected = [_oracle(series.tolist(), frequency.item(), 6) for frequency in FREQUENCIES]
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-10)


def test_matches_chaostools_logistic_map():
    series = logistic_map_trajectory(1000)
    julia = _julia_growth_rates(series, FREQUENCIES, 100)
    assert torch.allclose(zero_one_growth_rate(series, FREQUENCIES), julia, atol=1e-9)


def test_matches_chaostools_quasi_periodic():
    steps = torch.arange(1000, dtype=torch.float64)
    series = 1.0 + torch.sin(0.3 * steps) + torch.sin(math.sqrt(2) * steps)
    julia = _julia_growth_rates(series, FREQUENCIES, 100)
    result = zero_one_growth_rate(series, FREQUENCIES)
    assert torch.allclose(result, julia, atol=1e-9)
    assert result.abs().max() < 0.3


def test_paper_logistic_map_chaotic_vs_periodic():
    # Gottwald & Melbourne (2009), Fig. 4/7: K -> 1 for chaotic mu, K -> 0 for the periodic window mu = 3.55.
    for chaotic in (3.9, 3.97, 4.0):
        assert zero_one_test(_logistic(chaotic, 2000), SWEEP) > 0.95
    assert zero_one_test(_logistic(3.55, 2000), SWEEP).abs() < 0.05
    assert zero_one_test(_logistic(3.2, 2000), SWEEP).abs() < 0.05


def test_median_even_and_odd_counts():
    series = _logistic(3.7, 300)
    rates = zero_one_growth_rate(series, FREQUENCIES)
    assert torch.isclose(zero_one_test(series, FREQUENCIES), rates.median())
    even = zero_one_growth_rate(series, FREQUENCIES[:6])
    assert torch.isclose(zero_one_test(series, FREQUENCIES[:6]), even.sort().values[2:4].mean())


def test_shapes():
    series = _logistic(3.9, 200)
    batch = torch.stack([series, series.flip(0)])[..., None]
    assert zero_one_growth_rate(series, FREQUENCIES).shape == (7,)
    assert zero_one_growth_rate(series[:, None], FREQUENCIES).shape == (7,)
    assert zero_one_growth_rate(series, 1.0).shape == ()
    assert zero_one_growth_rate(batch, FREQUENCIES).shape == (7, 2)
    assert zero_one_growth_rate(batch, 1.0).shape == (2,)
    assert zero_one_test(series, FREQUENCIES).shape == ()
    assert zero_one_test(batch, FREQUENCIES).shape == (2,)
    assert zero_one_test(batch[:1], FREQUENCIES).shape == (1,)


def test_batched_matches_individual_rows():
    torch.manual_seed(0)
    batch = torch.rand(4, 150, 1, dtype=torch.float64)
    rates = zero_one_growth_rate(batch, FREQUENCIES, max_lag=12)
    tests = zero_one_test(batch, FREQUENCIES, max_lag=12)
    for row in range(4):
        assert torch.allclose(rates[:, row], zero_one_growth_rate(batch[row], FREQUENCIES, max_lag=12))
        assert torch.isclose(tests[row], zero_one_test(batch[row], FREQUENCIES, max_lag=12))


def test_max_lag_changes_result_and_default_is_tenth():
    series = _logistic(3.9, 400)
    assert torch.allclose(
        zero_one_growth_rate(series, FREQUENCIES), zero_one_growth_rate(series, FREQUENCIES, max_lag=40)
    )
    assert not torch.allclose(
        zero_one_growth_rate(series, FREQUENCIES, max_lag=20), zero_one_growth_rate(series, FREQUENCIES, max_lag=40)
    )


def test_dtypes():
    series = _logistic(3.9, 300)
    reference = zero_one_test(series, FREQUENCIES)
    integer = zero_one_test(torch.arange(300) % 7, FREQUENCIES)
    assert integer.dtype == torch.get_default_dtype()
    assert zero_one_test(series.float(), FREQUENCIES).dtype == torch.float32
    assert torch.isclose(zero_one_test(series.float(), FREQUENCIES).double(), reference, atol=1e-5)
    half = zero_one_growth_rate(series.half(), FREQUENCIES)
    assert half.dtype == torch.float16 and torch.isfinite(half).all()
    assert zero_one_growth_rate(series.bfloat16(), FREQUENCIES).dtype == torch.bfloat16


def test_invalid_arguments_raise():
    series = _logistic(3.9, 100)
    with pytest.raises(ValueError, match="univariate"):
        zero_one_test(torch.randn(100, 2), FREQUENCIES)
    with pytest.raises(ValueError, match="complex"):
        zero_one_test(torch.randn(100, dtype=torch.complex64), FREQUENCIES)
    for bad in (0.0, math.pi, -1.0, 4.0, float("nan"), float("inf"), [], [[1.0, 2.0]], [1.0, 3.5]):
        with pytest.raises(ValueError):
            zero_one_growth_rate(series, bad)
    with pytest.raises(ValueError):
        zero_one_growth_rate(series, torch.tensor([1.0 + 0j]))
    for bad in (1, 0, -3, 2.5, 6.0, [5, 6], [6], "6", float("nan")):
        with pytest.raises(ValueError, match="max_lag"):
            zero_one_growth_rate(series, FREQUENCIES, max_lag=bad)
        with pytest.raises(ValueError, match="max_lag"):
            zero_one_test(series, FREQUENCIES, max_lag=bad)
    with pytest.raises(ValueError):
        zero_one_growth_rate(torch.randn(2, 2, 2, 2), FREQUENCIES)


def test_max_lag_accepts_numpy_integers():
    series = _logistic(3.9, 100)
    expected = zero_one_growth_rate(series, FREQUENCIES, max_lag=12)
    assert torch.equal(zero_one_growth_rate(series, FREQUENCIES, max_lag=numpy.int64(12)), expected)


def test_degenerate_inputs_are_nan():
    assert torch.isnan(zero_one_growth_rate(torch.ones(100, dtype=torch.float64), FREQUENCIES)).all()
    assert torch.isnan(zero_one_growth_rate(torch.zeros(100), FREQUENCIES)).all()
    assert torch.isnan(zero_one_growth_rate(torch.full((100,), 1e8, dtype=torch.float64), FREQUENCIES)).all()
    assert torch.isnan(zero_one_test(torch.ones(100), FREQUENCIES))
    assert torch.isnan(zero_one_growth_rate(torch.randn(15), FREQUENCIES)).all()  # default max_lag = 1
    assert torch.isnan(zero_one_growth_rate(torch.randn(20), FREQUENCIES, max_lag=20)).all()  # no window at lag 20
    assert torch.isnan(zero_one_growth_rate(torch.randn(0), FREQUENCIES)).all()
    assert torch.isfinite(zero_one_growth_rate(torch.randn(20), FREQUENCIES, max_lag=19)).all()


def test_nonfinite_rows_are_nan_and_isolated():
    torch.manual_seed(0)
    batch = torch.randn(4, 100, 1, dtype=torch.float64)
    clean = zero_one_test(batch, FREQUENCIES)
    batch[1, 3] = float("nan")
    batch[2, 50] = float("inf")
    result = zero_one_test(batch, FREQUENCIES)
    assert torch.isnan(result[1]) and torch.isnan(result[2])
    assert torch.allclose(result[[0, 3]], clean[[0, 3]])
    assert torch.isnan(zero_one_growth_rate(batch, FREQUENCIES)[:, 1]).all()


def test_vmap():
    torch.manual_seed(1)
    batch = torch.randn(3, 1, 100, 1, dtype=torch.float64)
    batch[1, 0, 5] = float("nan")
    mapped = torch.vmap(lambda row: zero_one_test(row, FREQUENCIES))(batch)
    direct = zero_one_test(batch[:, 0], FREQUENCIES)
    assert torch.allclose(mapped[:, 0], direct, equal_nan=True)
    mapped_rates = torch.vmap(lambda row: zero_one_growth_rate(row, FREQUENCIES))(batch)
    assert mapped_rates.shape == (3, 7, 1)


def test_gradcheck_growth_rate():
    torch.manual_seed(0)
    series = torch.rand(30, dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([1.0, 1.7, 2.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, c: zero_one_growth_rate(x, c, max_lag=6), (series, frequencies))
    assert torch.autograd.gradcheck(lambda x: zero_one_test(x, frequencies.detach(), max_lag=6), (series,))


def test_degenerate_row_does_not_poison_gradients():
    torch.manual_seed(0)
    batch = torch.rand(3, 60, 1, dtype=torch.float64)
    batch[1] = 2.0
    batch[2, 7] = float("nan")
    batch.requires_grad_(True)
    zero_one_growth_rate(batch, FREQUENCIES).nansum().backward()
    assert torch.isfinite(batch.grad).all()
    assert (batch.grad[0] != 0).any()
    assert (batch.grad[1] == 0).all() and (batch.grad[2] == 0).all()


OFFSET_SERIES = [10**13 + index % 7 for index in range(100)]


def test_large_offset_matches_high_precision_oracle():
    # 70-digit values at c = 1.0, 0.7, 2.0 (default max_lag 10).
    series = torch.tensor(OFFSET_SERIES, dtype=torch.float64)
    expected = [0.09762307643592048711785722, -0.15796468473095808408376439, -0.54222770155741613724805779]
    result = zero_one_growth_rate(series, torch.tensor([1.0, 0.7, 2.0], dtype=torch.float64))
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-12)
    for frequency, value in zip((1.0, 0.7, 2.0), expected):
        assert _oracle(OFFSET_SERIES, frequency, 10) == pytest.approx(value, abs=1e-15)


def test_small_frequency_matches_oracle_and_is_stable():
    series = [0.1 + 0.9 * math.sin(index * index) for index in range(200)]
    tensor = torch.tensor(series, dtype=torch.float64)
    for frequency in (1e-8, 0.01):
        assert zero_one_growth_rate(tensor, frequency, max_lag=20).item() == pytest.approx(
            _oracle(series, frequency, 20), abs=1e-10
        )
    rates = zero_one_growth_rate(tensor, torch.tensor([1e-8, 0.5], dtype=torch.float64))
    assert torch.isfinite(rates).all()


@pytest.mark.parametrize("frequency", [1e-8, 0.01, 1.0, 3.1])
def test_constant_series_is_nan_at_any_frequency(frequency):
    for length in (100, 10000):
        assert torch.isnan(zero_one_growth_rate(torch.full((length,), 2.5, dtype=torch.float64), frequency))
    assert torch.isnan(zero_one_growth_rate(torch.full((100,), 1e13, dtype=torch.float64), frequency))


def test_small_frequency_does_not_poison_other_frequencies_or_gradients():
    torch.manual_seed(0)
    series = torch.randn(80, dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([1e-8, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    rates = zero_one_growth_rate(series, frequencies, max_lag=8)
    assert torch.isfinite(rates).all()
    rates.sum().backward()
    assert torch.isfinite(series.grad).all() and torch.isfinite(frequencies.grad).all()


def test_median_propagates_nan():
    torch.manual_seed(0)
    batch = torch.randn(2, 80, 1, dtype=torch.float64)
    batch[1] = 3.0  # constant row: every K_c is NaN
    result = zero_one_test(batch, FREQUENCIES)
    assert torch.isfinite(result[0]) and torch.isnan(result[1])


def test_median_with_some_nan_growth_rates_is_nan(monkeypatch):
    # K_c is NaN per row, not per frequency, so force a mixed [C, B] matrix.
    import torchnlts.chaos.zero_one as module

    original = module._growth_rate

    def mixed(tensor, frequencies, max_lag):
        rates, swept = original(tensor, frequencies, max_lag)
        rates = rates.clone()
        rates[1, 0] = float("nan")
        return rates, swept

    monkeypatch.setattr(module, "_growth_rate", mixed)
    torch.manual_seed(0)
    batch = torch.randn(2, 80, 1, dtype=torch.float64)
    result = module.zero_one_test(batch, FREQUENCIES)
    assert torch.isnan(result[0]) and torch.isfinite(result[1])


def test_backward_through_constant_row_is_finite_and_zero():
    torch.manual_seed(0)
    batch = torch.rand(3, 60, 1, dtype=torch.float64)
    batch[1] = 0.0
    batch.requires_grad_(True)
    frequencies = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    zero_one_test(batch, frequencies).nansum().backward()
    assert torch.isfinite(batch.grad).all() and torch.isfinite(frequencies.grad).all()
    assert (batch.grad[1] == 0).all()
    assert (batch.grad[0] != 0).any() and (batch.grad[2] != 0).any()


def test_inf_rows_with_shared_parameter_and_differentiable_frequencies():
    torch.manual_seed(0)
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([1.0, 2.0, 2.5], dtype=torch.float64, requires_grad=True)
    base = torch.rand(4, 60, 1, dtype=torch.float64)
    base[1, 5] = float("inf")
    base[2, 9] = -float("inf")
    nonfinite = ~torch.isfinite(base)
    # Non-finite entries are written in place, so they carry no gradient of their own.
    scaled = (torch.where(nonfinite, 1.0, base) * scale).masked_scatter(nonfinite, base[nonfinite])
    result = zero_one_growth_rate(scaled, frequencies)
    assert (
        torch.isnan(result[:, 1]).all() and torch.isnan(result[:, 2]).all() and torch.isfinite(result[:, [0, 3]]).all()
    )
    result.nansum().backward()
    assert torch.isfinite(scale.grad) and torch.isfinite(frequencies.grad).all()
    # The finite rows alone give the same gradients (non-finite rows contribute nothing).
    scale_clean = scale.detach().clone().requires_grad_(True)
    frequencies_clean = frequencies.detach().clone().requires_grad_(True)
    zero_one_growth_rate(base[[0, 3]] * scale_clean, frequencies_clean).sum().backward()
    assert torch.allclose(scale.grad, scale_clean.grad) and torch.allclose(frequencies.grad, frequencies_clean.grad)


def test_short_series_backward_is_connected():
    series = torch.rand(15, dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    result = zero_one_growth_rate(series, frequencies)  # default max_lag = 1: NaN
    assert torch.isnan(result).all() and result.requires_grad
    result.sum().backward()
    assert series.grad is not None and frequencies.grad is not None


def test_nonconstant_series_with_exactly_zero_modified_displacement_is_nan():
    # Regression: for [-199, 1, ..., 1] the mean is exactly -1 and D(n) = M(n) - V_osc(n) is exactly 0 although
    # the series is not constant; only the rounding guard catches it.
    series = torch.ones(100, dtype=torch.float64)
    series[0] = -199.0
    series.requires_grad_()
    result = zero_one_growth_rate(series, torch.tensor([1.0, 1.7, 2.4], dtype=torch.float64))
    assert torch.isnan(result).all()
    result.nan_to_num().sum().backward()
    assert (series.grad == 0).all()
