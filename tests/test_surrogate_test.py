"""Tests for torchnlts.surrogate.shuffle and torchnlts.surrogate.hypothesis.

Ground truth:
  1. Paper: Theiler et al. (1992) -- shuffle surrogates keep the
     amplitude distribution and destroy temporal structure; the rank test
     of Theiler et al. / Schreiber & Schmitz (2000) with ``M`` surrogates
     has p-values on the grid ``j / (M + 1)``, uniform under the null.
     Hand-derived p-values are checked on deterministic statistics.
  2. Julia: TimeseriesSurrogates.jl's ``pvalue(SurrogateTest(...))``, run
     via subprocess. It counts ``k`` surrogates ``>=`` (right) or
     ``<=`` (left) the data statistic and returns ``k / M``; this module
     returns ``(1 + k) / (M + 1)``. Compared on a deterministic ``CircShift``
     (identical ``k``) and on random shuffles (equal up to Monte Carlo error).
"""

import pytest
import torch

from tests.conftest import run_julia
from torchnlts.surrogate import hypothesis
from torchnlts.surrogate.hypothesis import surrogate_test
from torchnlts.surrogate.iaaft import iaaft_surrogate
from torchnlts.surrogate.phase import phase_randomized_surrogate
from torchnlts.surrogate.shuffle import random_shift_surrogate, random_shuffle_surrogate


def _first_value(data):
    return data[:, 0, 0]


def _lag_one_autocorrelation(data):
    centered = data - data.mean(dim=1, keepdim=True)
    return (centered[:, 1:] * centered[:, :-1]).sum(dim=(1, 2)) / (centered**2).sum(dim=(1, 2))


def _time_reversal_asymmetry(data):
    series = data[..., 0]
    return ((series[:, 2:] - series[:, 1:-1]) ** 3).mean(dim=1) / ((series[:, 1:] - series[:, :-1]) ** 2).mean(
        dim=1
    ) ** 1.5


def _henon_x(num_points):
    state = torch.tensor([0.1, 0.1], dtype=torch.float64)
    values = []
    for step in range(num_points + 100):
        state = torch.stack([1.0 - 1.4 * state[0] ** 2 + state[1], 0.3 * state[0]])
        if step >= 100:
            values.append(state[0])
    return torch.stack(values)


def test_shuffle_preserves_values_and_shape_and_dtype():
    x = torch.randn(3, 50, 2, dtype=torch.float64)
    out = random_shuffle_surrogate(x, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out.sort(dim=1).values, x.sort(dim=1).values)
    assert not torch.equal(out, x)
    ints = torch.arange(10)
    assert random_shuffle_surrogate(ints).dtype == ints.dtype


def test_shuffle_rows_and_channels_get_own_permutation():
    x = torch.arange(30.0).view(1, 30, 1).repeat(4, 1, 2)
    out = random_shuffle_surrogate(x, generator=torch.Generator().manual_seed(1))
    assert len({tuple(out[row, :, channel].tolist()) for row in range(4) for channel in range(2)}) == 8


def test_shuffle_destroys_autocorrelation():
    x = torch.cumsum(torch.randn(2000, dtype=torch.float64), 0)
    assert _lag_one_autocorrelation(x.view(1, -1, 1)).item() > 0.9
    out = random_shuffle_surrogate(x, generator=torch.Generator().manual_seed(2))
    assert abs(_lag_one_autocorrelation(out.view(1, -1, 1)).item()) < 0.1


def test_shuffle_keys_are_float64():
    # Regression: float32 keys tie in long series; check the keys' dtype and distinct-ness.
    seen = []
    real = torch.rand

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dtype"))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "rand", spy)
        out = random_shuffle_surrogate(torch.arange(40000.0), generator=torch.Generator().manual_seed(0))
    assert seen == [torch.float64]
    assert torch.equal(out.sort().values, torch.arange(40000.0))


def test_shuffle_is_uniform_over_permutations():
    generator = torch.Generator().manual_seed(3)
    counts = {}
    x = torch.tensor([0.0, 1.0, 2.0])
    for _ in range(6000):
        key = tuple(random_shuffle_surrogate(x, generator=generator).tolist())
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 6 and min(counts.values()) > 850


def test_shift_is_cyclic_rotation_uniform_including_identity():
    x = torch.arange(7.0)
    generator = torch.Generator().manual_seed(4)
    counts = torch.zeros(7)
    for _ in range(700):
        out = random_shift_surrogate(x, generator=generator)
        # surrogate[t] = x[(t - s) mod T], so out[0] = x[-s mod 7]
        shift = int(-out[0].item()) % 7
        assert torch.equal(out, x[(torch.arange(7) - shift) % 7])
        counts[shift] += 1
    assert counts.min() > 60  # all 7 shifts, including 0, near 100


def test_shift_direction_matches_circshift(monkeypatch):
    # Julia/Base circshift(x, 2): out[t] = x[t - 2], i.e. torch.roll(x, 2).
    monkeypatch.setattr(torch, "randint", lambda low, high, size, **kwargs: torch.full(size, 2))
    x = torch.arange(7.0)
    assert torch.equal(random_shift_surrogate(x), torch.roll(x, 2))


def test_shift_rank_test_is_exact_on_two_point_group():
    # Regression: excluding the identity shift made p in {1, 1/100} for
    # x = [0, 1] vs [1, 0], i.e. 50% false rejection at 5%.
    x = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]]).repeat(200, 1, 1)
    p_values = surrogate_test(
        x, _first_value, random_shift_surrogate, num_surrogates=99, generator=torch.Generator().manual_seed(0)
    ).p_value
    assert (p_values <= 0.05).double().mean().item() < 0.15


def test_shift_batch_channels_shapes_and_short_series():
    x = torch.arange(40.0).view(1, 20, 2).repeat(5, 1, 1)
    out = random_shift_surrogate(x, generator=torch.Generator().manual_seed(5))
    assert out.shape == x.shape
    assert torch.equal(out.sort(dim=1).values, x.sort(dim=1).values)
    assert len({tuple(out[row, :, 0].tolist()) for row in range(5)}) > 1
    assert torch.equal(random_shift_surrogate(torch.tensor([3.0])), torch.tensor([3.0]))
    assert random_shift_surrogate(torch.arange(5.0)).shape == (5,)
    assert random_shift_surrogate(torch.randn(5, 3)).shape == (5, 3)


@pytest.mark.parametrize("surrogate", [random_shuffle_surrogate, random_shift_surrogate])
def test_generators_reproducible_and_nonfinite_values_move_not_raise(surrogate):
    x = torch.randn(30, dtype=torch.float64)
    x[3] = float("nan")
    a = surrogate(x, generator=torch.Generator().manual_seed(6))
    b = surrogate(x, generator=torch.Generator().manual_seed(6))
    assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
    assert torch.isnan(a).sum() == 1
    with pytest.raises(ValueError):
        surrogate(torch.zeros(2, 2, 2, 2))


def test_p_value_hand_derived_on_deterministic_statistic():
    # Statistic = first value; shuffles of 1..10 put each value first with probability 0.1.
    # A stub generator fixes the surrogates' first values to 1, 2, 5, 5, 9.
    firsts = torch.tensor([1.0, 2.0, 5.0, 5.0, 9.0])

    def stub(data, generator=None):
        out = data.clone()
        out[:, 0, 0] = firsts
        return out

    x = torch.tensor([5.0, 2.0, 3.0])
    expected = {
        "right": (1 + 3) / 6,
        "left": (1 + 4) / 6,
        "both": 1.0,
    }  # both: 2 * min(4, 5) / 6 clipped; >=5: 5,5,9; <=5: 1,2,5,5
    for tail, p_value in expected.items():
        result = surrogate_test(x, _first_value, stub, num_surrogates=5, tail=tail)
        assert result.p_value.item() == pytest.approx(p_value)
        assert result.statistic.item() == 5.0
        assert torch.equal(result.surrogate_statistics, firsts)
    assert surrogate_test(x, _first_value, stub, num_surrogates=5, tail="both").p_value.item() <= 1.0
    constant = surrogate_test(
        torch.tensor([5.0, 1.0, 1.0]),
        _first_value,
        lambda data, generator=None: data.clone(),
        num_surrogates=4,
        tail="both",
    )
    assert constant.p_value.item() == 1.0  # 2 * 5 / 5 clipped


def test_shapes_batched_and_per_row_independent_surrogates():
    x = torch.randn(4, 30, 1, dtype=torch.float64)
    result = surrogate_test(
        x, _first_value, random_shuffle_surrogate, num_surrogates=7, generator=torch.Generator().manual_seed(0)
    )
    assert (
        result.p_value.shape == (4,) and result.statistic.shape == (4,) and result.surrogate_statistics.shape == (4, 7)
    )
    assert torch.equal(result.statistic, x[:, 0, 0])
    same = x[:1].repeat(4, 1, 1)
    out = surrogate_test(
        same, _first_value, random_shuffle_surrogate, num_surrogates=5, generator=torch.Generator().manual_seed(0)
    )
    assert len({tuple(row.tolist()) for row in out.surrogate_statistics}) == 4
    vector = surrogate_test(x[0, :, 0], _first_value, random_shuffle_surrogate, num_surrogates=7)
    assert vector.p_value.shape == () and vector.surrogate_statistics.shape == (7,)
    matrix = surrogate_test(x[0], _first_value, random_shuffle_surrogate, num_surrogates=7)
    assert matrix.p_value.shape == ()
    assert result.p_value.dtype == torch.float64


def test_reproducible_with_generator():
    x = torch.randn(40, dtype=torch.float64)
    runs = [
        surrogate_test(
            x,
            _lag_one_autocorrelation,
            random_shuffle_surrogate,
            num_surrogates=25,
            generator=torch.Generator().manual_seed(9),
        )
        for _ in range(2)
    ]
    assert torch.equal(runs[0].surrogate_statistics, runs[1].surrogate_statistics)


def test_chunking_equality(monkeypatch):
    x = torch.randn(3, 40, 1, dtype=torch.float64)
    full = surrogate_test(
        x,
        _lag_one_autocorrelation,
        random_shuffle_surrogate,
        num_surrogates=11,
        generator=torch.Generator().manual_seed(1),
    )
    monkeypatch.setattr(hypothesis, "_ELEMENT_BUDGET", 3 * 40 * 3)  # 3 surrogates per chunk
    calls = []

    def counting(data):
        calls.append(data.shape[0])
        return _lag_one_autocorrelation(data)

    chunked = surrogate_test(
        x, counting, random_shuffle_surrogate, num_surrogates=11, generator=torch.Generator().manual_seed(1)
    )
    assert calls == [3, 9, 9, 9, 6]
    assert torch.equal(full.surrogate_statistics, chunked.surrogate_statistics)
    assert torch.equal(full.p_value, chunked.p_value)
    # Regression: M - 1 divisible by the chunk size must not drop the last chunk.
    result = surrogate_test(x, _lag_one_autocorrelation, random_shuffle_surrogate, num_surrogates=10)
    assert result.surrogate_statistics.shape == (3, 10)


def test_nan_rows_and_nan_statistics():
    x = torch.randn(3, 30, 1, dtype=torch.float64)
    x[1, 4, 0] = float("nan")
    result = surrogate_test(
        x,
        _lag_one_autocorrelation,
        random_shuffle_surrogate,
        num_surrogates=9,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.isnan(result.p_value[1]) and torch.isfinite(result.p_value[[0, 2]]).all()
    calls = []

    def flaky(data):
        calls.append(1)
        values = _first_value(data).clone()
        if len(calls) == 2:
            values[0] = float("nan")
        return values

    result = surrogate_test(torch.randn(2, 10, 1), flaky, random_shuffle_surrogate, num_surrogates=3)
    assert torch.isnan(result.p_value[0]) and torch.isfinite(result.p_value[1])
    assert torch.isnan(
        surrogate_test(x[1], _lag_one_autocorrelation, phase_randomized_surrogate, num_surrogates=3).p_value
    )


def test_invalid_arguments():
    x = torch.randn(20)
    for kwargs in (
        {"num_surrogates": 0},
        {"num_surrogates": 1.5},
        {"tail": "up"},
        {"statistic": 3},
        {"surrogate": None},
    ):
        arguments = {"statistic": _first_value, "surrogate": random_shuffle_surrogate, "num_surrogates": 3, **kwargs}
        with pytest.raises(ValueError):
            surrogate_test(x, **arguments)
    with pytest.raises(ValueError):
        surrogate_test(x, lambda data: torch.zeros(5), random_shuffle_surrogate, num_surrogates=3)


def test_integer_statistics_ranked_without_float_cast():
    # Regression: 16777217 and 16777216 are equal in float32.
    values = torch.tensor([16777216, 16777216, 16777216, 16777216], dtype=torch.int64)

    def stub(data, generator=None):
        out = data.clone()
        out[:, 0, 0] = values[: data.shape[0]]
        return out

    x = torch.tensor([16777217.0, 0.0], dtype=torch.float64)
    result = surrogate_test(x, lambda data: data[:, 0, 0].to(torch.int64), stub, num_surrogates=4, tail="right")
    assert result.p_value.item() == pytest.approx(1 / 5)
    assert result.statistic.dtype == torch.int64 and result.p_value.dtype == torch.float64


def test_nonfinite_rows_and_infinite_statistics_give_nan():
    x = torch.randn(3, 20, 1, dtype=torch.float64)
    x[0, 3, 0] = float("inf")
    result = surrogate_test(
        x, lambda data: data[:, :, 0].mean(dim=1).clamp(max=1e9), random_shuffle_surrogate, num_surrogates=5
    )
    assert torch.isnan(result.p_value[0]) and torch.isfinite(result.p_value[1:]).all()
    data_calls = []

    def inf_on_data(data):
        data_calls.append(1)
        return data[:, 0, 0] / (0.0 if len(data_calls) == 1 else 1.0)

    inf_data = surrogate_test(x[1:], inf_on_data, random_shuffle_surrogate, num_surrogates=5)
    assert torch.isnan(inf_data.p_value).all()
    calls = []

    def inf_on_surrogates(data):
        calls.append(1)
        return data[:, 0, 0] / (1.0 if len(calls) == 1 else 0.0)

    inf_surrogate = surrogate_test(x[1:], inf_on_surrogates, random_shuffle_surrogate, num_surrogates=5)
    assert torch.isnan(inf_surrogate.p_value).all()


def test_statistic_views_do_not_keep_surrogate_chunks_alive(monkeypatch):
    # Regression: a view like data[:, 0, 0] pins its whole [chunk*B, T, D] base.
    x = torch.randn(2, 30, 1, dtype=torch.float64)
    monkeypatch.setattr(hypothesis, "_ELEMENT_BUDGET", x.numel())
    real_cat = torch.cat
    stored = []

    def spy(tensors, *args, **kwargs):
        stored.append([tensor.untyped_storage().nbytes() // tensor.element_size() for tensor in tensors])
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", spy)
    surrogate_test(x, lambda data: data[:, 0, 0], random_shuffle_surrogate, num_surrogates=4)
    assert stored and all(size == 2 for sizes in stored for size in sizes)


def test_results_carry_no_autograd_graph():
    x = torch.randn(2, 20, 1, dtype=torch.float64, requires_grad=True)
    grad_enabled = []

    def statistic(data):
        grad_enabled.append(torch.is_grad_enabled())
        return data[:, 0, 0]

    result = surrogate_test(x, statistic, random_shuffle_surrogate, num_surrogates=3)
    assert not any(grad_enabled)
    assert not any(tensor.requires_grad for tensor in result)


def test_surrogate_kwargs_forwarded():
    x = torch.randn(30, dtype=torch.float64)
    seen = []

    def spy(data, generator=None, marker=None):
        seen.append(marker)
        return data

    surrogate_test(x, _first_value, spy, num_surrogates=2, marker="m")
    assert seen == ["m"]


def test_uniform_p_values_under_the_null():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(300, 40, 1, dtype=torch.float64, generator=generator)
    p_values = surrogate_test(
        x, _lag_one_autocorrelation, random_shuffle_surrogate, num_surrogates=19, tail="both", generator=generator
    ).p_value
    # Two-sided grid 2j/20: P(p <= 0.1) = 0.1 and P(p <= 0.5) = 0.5 for iid data.
    assert abs((p_values <= 0.1).double().mean().item() - 0.1) < 0.06
    assert abs((p_values <= 0.5).double().mean().item() - 0.5) < 0.1
    one_sided = surrogate_test(
        x, _lag_one_autocorrelation, random_shuffle_surrogate, num_surrogates=19, generator=generator
    ).p_value
    assert abs((one_sided <= 0.25).double().mean().item() - 0.25) < 0.08


def test_detects_nonlinearity_in_henon_against_iaaft():
    series = _henon_x(1000)
    result = surrogate_test(
        series,
        _time_reversal_asymmetry,
        iaaft_surrogate,
        num_surrogates=19,
        tail="both",
        generator=torch.Generator().manual_seed(0),
        max_iterations=30,
    )
    assert result.p_value.item() <= 0.1
    noise = surrogate_test(
        torch.randn(1000, dtype=torch.float64, generator=torch.Generator().manual_seed(5)),
        _time_reversal_asymmetry,
        iaaft_surrogate,
        num_surrogates=19,
        tail="both",
        generator=torch.Generator().manual_seed(0),
        max_iterations=30,
    )
    assert noise.p_value.item() > 0.1


def test_matches_timeseries_surrogates_jl_convention():
    x = list(map(float, range(1, 11)))
    julia_code = f"""
    using TimeseriesSurrogates, Random
    x = {x}
    f(s) = s[1]
    # CircShift(2): every surrogate starts with x[end-1] = 9 > data value 1
    for tail in (:left, :right, :both)
        println(pvalue(SurrogateTest(f, x, CircShift(2); n = 10, threaded = false); tail = tail))
    end
    # RandomShuffle, data first value 1, 4000 surrogates
    for tail in (:left, :right)
        println(pvalue(SurrogateTest(f, x, RandomShuffle(); rng = Xoshiro(1), n = 4000, threaded = false); tail = tail))
    end
    """
    stdout = run_julia(julia_code)
    left, right, both, shuffle_left, shuffle_right = [float(v) for v in stdout.split()]
    assert (left, right, both) == (0.0, 1.0, 0.0)

    def shift_two(data, generator=None):
        return torch.roll(data, 2, dims=1)

    data = torch.tensor(x, dtype=torch.float64)
    for tail, julia_p in (("left", left), ("right", right), ("both", both)):
        ours = surrogate_test(data, _first_value, shift_two, num_surrogates=10, tail=tail).p_value.item()
        # k from Julia's convention, ours = (1 + k) / (M + 1) (clipped to 1)
        assert ours == pytest.approx(min(1.0, (1 + julia_p * 10) / 11 * (2 if tail == "both" else 1)))
    for tail, julia_p in (("left", shuffle_left), ("right", shuffle_right)):
        ours = surrogate_test(
            data,
            _first_value,
            random_shuffle_surrogate,
            num_surrogates=4000,
            tail=tail,
            generator=torch.Generator().manual_seed(2),
        )
        assert abs(ours.p_value.item() - julia_p) < 0.03


def test_half_precision_p_value_with_large_counts():
    # Regression: the rank count was cast to float16 before division and overflowed to inf.
    def surrogate(data, generator=None):
        out = torch.zeros_like(data)
        out[:65519] = 1
        return out

    for tail, count in (("right", 65520), ("left", 100001)):
        result = surrogate_test(
            torch.ones(1, dtype=torch.float16), lambda data: data[:, 0, 0], surrogate, num_surrogates=100000, tail=tail
        )
        expected = torch.tensor(count / 100001, dtype=torch.float64).to(torch.float16)
        assert result.p_value.dtype == torch.float16 and result.p_value == expected


def test_mixed_statistic_dtypes_are_promoted_before_ranking():
    calls = [0]

    def statistic(data):
        calls[0] += 1
        if calls[0] == 1:
            return torch.ones(data.shape[0], dtype=torch.float32)
        return torch.full((data.shape[0],), 1 - 2**-30, dtype=torch.float64)

    result = surrogate_test(torch.zeros(2), statistic, lambda data, generator=None: data, num_surrogates=5)
    assert result.p_value.item() == pytest.approx(1 / 6)
