"""Tests for torchnlts.prediction.analogues: a plain-loop numpy reference, causality, periodic recovery, a
hand-derived case, chunking and batching invariance, non-finite and constant windows, dtype, gradients."""

import math
import sys

import numpy as np
import pytest
import torch

from torchnlts.prediction import analogues

module = sys.modules["torchnlts.prediction.analogues"]  # the package attribute is the function


def _series(seed, length=200, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.cumsum(torch.randn(length, generator=generator, dtype=dtype), dim=0) + torch.sin(
        torch.arange(length, dtype=dtype) / 5
    )


def _znorm(values):
    std = values.std()
    return np.zeros_like(values) if std == 0 else (values - values.mean()) / std


def _reference(values, origin, context_length, horizon, num_analogues):
    """Brute force: (candidate distances, [(start, distance)], continuations by start) for one origin."""
    context = values[origin - context_length : origin]
    candidates = {}
    for start in range(0, origin - context_length - horizon + 1):
        window = values[start : start + context_length]
        distance = np.linalg.norm(_znorm(window) - _znorm(context))
        scale = 1.0 if window.std() == 0 or context.std() == 0 else context.std() / window.std()
        future = values[start + context_length : start + context_length + horizon]
        candidates[start] = (distance, (future - window.mean()) * scale + context.mean())
    zone = math.ceil(context_length / 4)
    picked, remaining = [], dict(candidates)
    for _ in range(num_analogues):
        if not remaining:
            break
        start = min(remaining, key=lambda key: (remaining[key][0], key))
        picked.append(start)
        remaining = {key: value for key, value in remaining.items() if abs(key - start) > zone}
    return candidates, picked


@pytest.mark.parametrize("num_analogues", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_brute_force(seed, num_analogues):
    series = _series(seed)
    origins = [40, 77, 120, 199, 200]
    distances, starts, continuations = analogues(series, origins, 16, 8, num_analogues)
    values = series.numpy()
    for row, origin in enumerate(origins):
        candidates, picked = _reference(values, origin, 16, 8, num_analogues)
        assert (starts[row, : len(picked)] >= 0).all()
        for slot, expected in enumerate(picked):
            start = int(starts[row, slot])
            if start != expected:  # tie: same distance
                assert abs(candidates[start][0] - candidates[expected][0]) < 1e-10
            assert abs(float(distances[row, slot]) - candidates[start][0]) < 1e-10
            np.testing.assert_allclose(continuations[row, slot].numpy(), candidates[start][1], atol=1e-10, rtol=0)
        assert (starts[row, len(picked) :] == -1).all()


@pytest.mark.parametrize("seed", range(5))
def test_causality(seed):
    generator = torch.Generator().manual_seed(seed)
    series = torch.randn(120, generator=generator, dtype=torch.float64)
    origins = torch.arange(10, 121, 3)
    context_length, horizon = 10, 7
    _, starts, _ = analogues(series, origins, context_length, horizon, 4)
    valid = starts >= 0
    assert valid.any()
    assert (starts + context_length + horizon <= origins[:, None])[valid].all()
    # the bound is tight: an origin with exactly one admissible start returns it
    _, starts, _ = analogues(series, [context_length + horizon], context_length, horizon, 2)
    assert starts.tolist() == [[0, -1]]
    _, starts, _ = analogues(series, [context_length], context_length, horizon)
    assert starts.tolist() == [[-1]]


def test_periodic_series_recovers_previous_period():
    period = 25
    time = torch.arange(300, dtype=torch.float64)
    series = torch.sin(2 * torch.pi * time / period) + 0.3 * torch.sin(4 * torch.pi * time / period)
    origin, horizon = 200, 12
    distances, starts, continuations = analogues(series, [origin], 20, horizon)
    assert float(distances[0, 0]) < 1e-6
    assert (origin - 20 - int(starts[0, 0])) % period == 0  # a whole number of periods back
    torch.testing.assert_close(continuations[0, 0], series[origin : origin + horizon], atol=1e-8, rtol=0)


def test_hand_derived_case():
    # Context [0, 1, 0, 1] at origin 13 matches the window [10, 12, 10, 12] at start 0 (same shape, scale 2,
    # offset 10); its continuation [20, 30] maps to (c - 11) / 1 * 0.5 + 0.5, with std(window) = 1, std(context) = 0.5.
    series = torch.tensor([10.0, 12, 10, 12, 20, 30, 3, 9, 4, 8, 2, 0, 1, 0, 1], dtype=torch.float64)
    distances, starts, continuations = analogues(series, [len(series)], 4, 2)
    assert int(starts[0, 0]) == 0
    assert float(distances[0, 0]) < 1e-12
    expected = torch.tensor([(20 - 11) * 0.5 + 0.5, (30 - 11) * 0.5 + 0.5], dtype=torch.float64)
    torch.testing.assert_close(continuations[0, 0], expected)


def test_chunking_invariance(monkeypatch):
    series = _series(3, 150)
    origins = torch.arange(20, 151, 5)
    expected = analogues(series, origins, 12, 6, 3)
    for elements in (1, 140, 700, 10**9):
        monkeypatch.setattr(module, "_CHUNK_ELEMENTS", elements)
        for got, want in zip(analogues(series, origins, 12, 6, 3), expected):
            torch.testing.assert_close(got, want, equal_nan=True, atol=0, rtol=0)


def test_batched_equals_per_row(monkeypatch):
    batch = torch.stack([_series(seed, 100) for seed in range(3)]).unsqueeze(-1)
    origins = [30, 60, 100]
    batched = analogues(batch, origins, 10, 5, 2)
    for row in range(3):
        for got, want in zip(batched, analogues(batch[row, :, 0], origins, 10, 5, 2)):
            torch.testing.assert_close(got[row], want, equal_nan=True, atol=0, rtol=0)
    assert batched[0].shape == (3, 3, 2) and batched[2].shape == (3, 3, 2, 5)
    monkeypatch.setattr(module, "_CHUNK_ELEMENTS", 1)
    for got, want in zip(analogues(batch, origins, 10, 5, 2), batched):
        torch.testing.assert_close(got, want, equal_nan=True, atol=0, rtol=0)


def test_shapes_and_input_forms():
    series = _series(0, 80)
    base = analogues(series, [50, 60], 10, 4, 2)
    for other in (series[:, None], series.numpy()):
        for got, want in zip(analogues(other, torch.tensor([50, 60]), 10, 4, 2), base):
            torch.testing.assert_close(got, want)
    assert [tuple(part.shape) for part in base] == [(2, 2), (2, 2), (2, 2, 4)]
    assert base[1].dtype == torch.int64
    assert analogues(series, [], 10, 4, 2)[2].shape == (0, 2, 4)


def test_nonfinite():
    series = _series(1, 120)
    # NaN in the context: all analogues missing
    broken = series.clone()
    broken[95] = float("nan")
    distances, starts, continuations = analogues(broken, [100, 130 - 10], 10, 5, 2)
    assert torch.isinf(distances[0]).all() and (starts[0] == -1).all() and continuations[0].isnan().all()
    # NaN in a candidate window or in a continuation excludes that start only
    broken = series.clone()
    broken[30] = float("nan")
    _, starts, continuations = analogues(broken, [110], 10, 5, 3)
    for start in starts[0].tolist():
        assert not (start <= 30 < start + 15)
    assert torch.isfinite(continuations).all()
    # equals the series with those starts removed by hand: no gradient/NaN leaks into other rows
    distances, _, _ = analogues(broken, [110], 10, 5, 3)
    assert torch.isfinite(distances).all()


def test_constant_windows():
    series = torch.cat([torch.full((20,), 3.0), _series(0, 60), torch.full((20,), 5.0)]).to(torch.float64)
    # constant context: matches constant windows (distance 0), continuation is shifted only
    distances, starts, continuations = analogues(series, [100], 10, 4, 1)
    assert float(distances[0, 0]) == 0.0
    start = int(starts[0, 0])
    assert (series[start : start + 10] == series[start]).all()
    future = series[start + 10 : start + 14]
    torch.testing.assert_close(continuations[0, 0], future - series[start] + 5.0)
    # constant analogue window against a varying context: shift only, so the continuation keeps its scale
    series = torch.cat(
        [
            torch.full((10,), 2.0),
            torch.tensor([1.0, 4.0, 9.0, 16.0]),
            torch.tensor([0.0, 1.0] * 5),
            torch.tensor([1.0, 0.0]),
        ]
    )
    series = series.to(torch.float64)
    _, starts, continuations = analogues(series, [len(series)], 10, 4, 1)
    context = series[-10:]
    assert int(starts[0, 0]) in (0, 1, 2, 3, 4, 5, 6)
    start = int(starts[0, 0])
    window = series[start : start + 10]
    future = series[start + 10 : start + 14]
    ratio = 1.0 if window.std(unbiased=False) == 0 else context.std(unbiased=False) / window.std(unbiased=False)
    torch.testing.assert_close(continuations[0, 0], (future - window.mean()) * ratio + context.mean())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype):
    series = _series(2, 100, dtype)
    distances, starts, continuations = analogues(series, [70, 99], 10, 5, 2)
    assert distances.dtype == dtype and continuations.dtype == dtype and starts.dtype == torch.int64
    reference = analogues(series.double(), [70, 99], 10, 5, 2)
    assert torch.equal(starts, reference[1])
    torch.testing.assert_close(continuations.double(), reference[2], atol=1e-3, rtol=1e-3)
    assert analogues(torch.arange(60), [50], 8, 3)[0].dtype == torch.get_default_dtype()


def test_gradcheck_continuations_and_distances():
    series = _series(4, 60).requires_grad_()
    origins = [40, 58]

    def function(values):
        distances, _, continuations = analogues(values, origins, 8, 4, 2)
        return distances, continuations

    assert torch.autograd.gradcheck(function, (series,), eps=1e-6, atol=1e-5)


def test_gradient_isolation_across_batch():
    batch = torch.stack([_series(seed, 80) for seed in range(2)]).unsqueeze(-1).requires_grad_()
    _, _, continuations = analogues(batch, [60, 79], 8, 4, 2)
    continuations[0].sum().backward()
    assert batch.grad[0].abs().sum() > 0
    assert (batch.grad[1] == 0).all()
    # missing analogues must not poison gradients with NaN
    batch.grad = None
    _, _, continuations = analogues(batch, [8, 79], 8, 4, 2)
    continuations.nan_to_num().sum().backward()
    assert torch.isfinite(batch.grad).all()


def test_validation():
    series = _series(0, 50)
    for kwargs in (
        dict(origins=[5], context_length=10, horizon=2),
        dict(origins=[51], context_length=10, horizon=2),
        dict(origins=[20], context_length=1, horizon=2),
        dict(origins=[20], context_length=60, horizon=2),
        dict(origins=[20], context_length=10, horizon=0),
        dict(origins=[20], context_length=10, horizon=2, num_analogues=0),
        dict(origins=[20.0], context_length=10, horizon=2),
        dict(origins=[[20]], context_length=10, horizon=2),
        dict(origins=[20], context_length=10.5, horizon=2),
    ):
        with pytest.raises(ValueError):
            analogues(series, **kwargs)
    with pytest.raises(ValueError):
        analogues(torch.zeros(50, 2), [20], 10, 2)


def test_missing_analogue_when_context_and_horizon_exceed_series():
    # Regression: the masked gather for a missing pick ran past the end of the series.
    distances, starts, continuations = analogues(_series(0, 10), [10], 5, 8, 2)
    assert torch.isinf(distances).all() and (starts == -1).all() and torch.isnan(continuations).all()


@pytest.mark.parametrize("scale", [1e-160, 1e150])
def test_continuations_scale_equivariant(scale):
    # Regression: the rescaling variance underflowed to subnormals at small amplitudes.
    series = _series(3, 200)
    _, starts, continuations = analogues(series, [150, 200], 20, 5, 2)
    _, scaled_starts, scaled_continuations = analogues(series * scale, [150, 200], 20, 5, 2)
    assert torch.equal(starts, scaled_starts)
    torch.testing.assert_close(scaled_continuations / scale, continuations, rtol=1e-12, atol=0)
