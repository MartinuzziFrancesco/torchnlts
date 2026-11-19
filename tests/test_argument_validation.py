"""Bad arguments raise ValueError (strings, NaN, negative values, wrong rank, unsupported combinations)."""

import pytest
import torch

from torchnlts.dimension import correlation_sum, soft_correlation_sum
from torchnlts.entropy import (
    approximate_entropy,
    fuzzy_entropy,
    lempel_ziv_complexity,
    multiscale_entropy,
    renyi_entropy,
    sample_entropy,
    tsallis_entropy,
)
from torchnlts.neighbors import knn_search, radius_neighbors
from torchnlts.recurrence import (
    hard_cross_recurrence_matrix,
    hard_joint_recurrence_matrix,
    hard_recurrence_matrix,
)
from torchnlts.sciml import AttractorLoss, CorrelationDimensionLoss, LyapunovLoss, RecurrenceLoss

SERIES = torch.sin(torch.linspace(0, 20, 60))
PROBABILITY = torch.tensor([0.25, 0.75])

RADIUS_FUNCTIONS = {
    "radius_neighbors": lambda radius: radius_neighbors(SERIES, radius),
    "hard_recurrence": lambda radius: hard_recurrence_matrix(SERIES, epsilon=radius),
    "hard_cross_recurrence": lambda radius: hard_cross_recurrence_matrix(SERIES, SERIES, epsilon=radius),
    "hard_joint_recurrence": lambda radius: hard_joint_recurrence_matrix(SERIES, SERIES, epsilon=radius),
    "correlation_sum": lambda radius: correlation_sum(SERIES, radius),
    "soft_correlation_sum": lambda radius: soft_correlation_sum(SERIES, radius),
    "sample_entropy": lambda radius: sample_entropy(SERIES, radius),
    "approximate_entropy": lambda radius: approximate_entropy(SERIES, radius),
    "fuzzy_entropy": lambda radius: fuzzy_entropy(SERIES, radius),
    "multiscale_entropy": lambda radius: multiscale_entropy(SERIES, [1, 2], radius),
}


@pytest.mark.parametrize("name", RADIUS_FUNCTIONS)
@pytest.mark.parametrize("radius", ["a", -1.0, float("nan"), torch.tensor(-0.5), torch.tensor(float("nan"))])
def test_radius_rejects_string_negative_and_nan(name, radius):
    with pytest.raises(ValueError):
        RADIUS_FUNCTIONS[name](radius)


@pytest.mark.parametrize("name", ["correlation_sum", "soft_correlation_sum", "sample_entropy", "approximate_entropy"])
def test_radius_sweep_must_be_at_most_one_dimensional(name):
    with pytest.raises(ValueError):
        RADIUS_FUNCTIONS[name](torch.ones(2, 2))


@pytest.mark.parametrize("name", ["radius_neighbors", "hard_recurrence", "hard_cross_recurrence"])
def test_scalar_radius_rejects_a_sweep(name):
    with pytest.raises(ValueError):
        RADIUS_FUNCTIONS[name](torch.tensor([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("name", ["soft_correlation_sum", "fuzzy_entropy"])
def test_positive_radius_rejects_zero(name):
    with pytest.raises(ValueError):
        RADIUS_FUNCTIONS[name](0.0)


@pytest.mark.parametrize("fuzzy_power", [-1.0, 0.0, float("nan"), "a"])
def test_fuzzy_power_must_be_positive(fuzzy_power):
    with pytest.raises(ValueError):
        fuzzy_entropy(SERIES, 0.2, fuzzy_power=fuzzy_power)


@pytest.mark.parametrize("function", [renyi_entropy, tsallis_entropy])
@pytest.mark.parametrize("order", ["2", float("nan")])
def test_entropy_order_must_be_a_number(function, order):
    with pytest.raises(ValueError):
        function(PROBABILITY, order)


@pytest.mark.parametrize("threshold", ["median", "Mean"])
def test_lempel_ziv_string_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be 'mean' or a number"):
        lempel_ziv_complexity(SERIES, threshold=threshold)


@pytest.mark.parametrize(
    "call",
    [
        lambda: knn_search(SERIES, 2, other=SERIES, theiler_window=3),
        lambda: knn_search(SERIES, 2, other=SERIES, theiler_window="bad"),
        lambda: radius_neighbors(SERIES, 0.5, other=SERIES, theiler_window=3),
        lambda: radius_neighbors(SERIES, 0.5, other=SERIES, theiler_window="bad"),
    ],
)
def test_theiler_window_with_other(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "build",
    [
        lambda: LyapunovLoss(0.0, num_steps=0),
        lambda: LyapunovLoss(0.0, num_steps="a"),
        lambda: LyapunovLoss(0.0, num_steps=5, num_exponents=0),
        lambda: LyapunovLoss(0.0, num_steps=5, dt=-1.0),
        lambda: LyapunovLoss(0.0, num_steps=5, transient_steps=-1),
        lambda: CorrelationDimensionLoss(1.0, torch.tensor([0.1, 0.2]), kind="bad"),
        lambda: CorrelationDimensionLoss(1.0, torch.tensor([0.1, -0.2])),
        lambda: CorrelationDimensionLoss(1.0, torch.ones(2, 2)),
        lambda: CorrelationDimensionLoss(1.0, torch.tensor([0.1, 0.2]), theiler_window="a"),
        lambda: CorrelationDimensionLoss(1.0, torch.tensor([0.1, 0.2]), norm_order="a"),
        lambda: RecurrenceLoss(0.5, 0.0),
        lambda: RecurrenceLoss(0.5, "a"),
        lambda: RecurrenceLoss(0.5, 0.5, kind="bad"),
        lambda: RecurrenceLoss(0.5, 0.5, theiler_window=1.5),
        lambda: RecurrenceLoss(0.5, 0.5, norm_order=-1.0),
        lambda: AttractorLoss(-1.0),
        lambda: AttractorLoss(0.5, kind="bad"),
    ],
)
def test_loss_constructors_validate(build):
    with pytest.raises(ValueError):
        build()


def test_soft_correlation_sum_gradient_reaches_temperature():
    temperature = torch.tensor([0.3, 0.5], requires_grad=True)
    soft_correlation_sum(SERIES, temperature).sum().backward()
    assert temperature.grad is not None
