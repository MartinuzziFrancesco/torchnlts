import pytest
import torch

from torchnlts.dimension.higuchi import higuchi_dimension, higuchi_length
from torchnlts.dimension.katz import katz_dimension
from torchnlts.dimension.petrosian import petrosian_dimension
from torchnlts.dimension.variogram import power_variogram, variogram_dimension
from torchnlts.fluctuations.dfa import dfa_exponent, dfa_fluctuation
from torchnlts.lyapunov.kantz import kantz_divergence
from torchnlts.lyapunov.rosenstein import rosenstein_divergence, rosenstein_lyapunov
from torchnlts.neighbors.distance import pairwise_distance

FUNCTIONS = {
    "higuchi": lambda batch: higuchi_length(batch, [1, 2, 4]),
    "higuchi_scalar": lambda batch: higuchi_length(batch, 2),
    "katz": katz_dimension,
    "katz_amplitude": lambda batch: katz_dimension(batch, amplitude_only=True),
    "petrosian": petrosian_dimension,
    "variogram": lambda batch: power_variogram(batch, [1, 2], power=0.5),
    "dfa": lambda batch: dfa_fluctuation(batch, [8, 16]),
    "rosenstein": lambda batch: rosenstein_divergence(batch, [0, 1, 2], embedding_dim=2, theiler_window=2),
    "kantz": lambda batch: kantz_divergence(batch, [0, 1, 2], radius=1.0, embedding_dim=2, theiler_window=2),
    "higuchi_dimension": lambda batch: higuchi_dimension([1, 2, 4], higuchi_length(batch, [1, 2, 4])),
    "variogram_dimension": lambda batch: variogram_dimension([1, 2], power_variogram(batch, [1, 2])),
    "dfa_exponent": lambda batch: dfa_exponent([8, 16], dfa_fluctuation(batch, [8, 16])),
    "rosenstein_lyapunov": lambda batch: rosenstein_lyapunov(
        [0, 1, 2], rosenstein_divergence(batch, [0, 1, 2], embedding_dim=2, theiler_window=2)
    ),
}
DIFFERENTIABLE = [name for name in FUNCTIONS if name != "petrosian"]


def _batch(dtype=torch.float64):
    torch.manual_seed(0)
    batch = torch.randn(5, 64, 1, dtype=torch.float64)
    batch[1, 0] = float("nan")
    batch[2, -1] = float("inf")
    batch[3, 9] = -float("inf")
    return batch.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", FUNCTIONS)
def test_nonfinite_rows_give_nan_and_leave_other_rows(name, dtype):
    function = FUNCTIONS[name]
    batch = _batch(dtype)
    result = function(batch)
    assert torch.isnan(result[..., 1:4]).all()
    for row in (0, 4):
        assert torch.equal(result[..., row], function(batch[row]))


@pytest.mark.parametrize("name", DIFFERENTIABLE)
def test_nonfinite_rows_leave_other_gradients(name):
    function = FUNCTIONS[name]
    batch = _batch().requires_grad_()
    (function(batch)[..., 0].sum() + function(batch)[..., 4].sum()).backward()
    assert (batch.grad[1:4] == 0).all()
    for row in (0, 4):
        single = batch[row].detach().requires_grad_()
        function(single).sum().backward()
        assert single.grad.abs().sum() > 0
        assert torch.allclose(batch.grad[row], single.grad, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("name", DIFFERENTIABLE)
def test_nonfinite_rows_shared_parameter_gradient(name):
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    clean = _batch()
    nonfinite = ~torch.isfinite(clean)
    # Non-finite entries are written in place, so they carry no gradient of their own.
    batch = (torch.where(nonfinite, 1.0, clean) * parameter).masked_scatter(nonfinite, clean[nonfinite])
    gradient = torch.autograd.grad(FUNCTIONS[name](batch)[..., 0].sum(), parameter)[0]
    assert torch.isfinite(gradient)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_nonfinite_rows_vmap(name):
    function = FUNCTIONS[name]
    batch = _batch()
    mapped = torch.vmap(function)(batch[:, None])
    assert torch.allclose(mapped.squeeze(-1).movedim(0, -1), function(batch), equal_nan=True)


def test_pairwise_distance_nonfinite_points_multivariate_cross():
    # Points with a single non-finite coordinate are invalid; other pairs keep
    # their norm-specific distances, including norm_order=0 and a separate other.
    trajectory = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, float("nan")]], dtype=torch.float64)
    other = torch.tensor([[0.0, 1.0], [float("inf"), 2.0]], dtype=torch.float64)
    expected = {0: [1.0, 2.0], 1: [1.0, 6.0], 2: [1.0, 18.0**0.5], float("inf"): [1.0, 3.0]}
    for norm_order, finite_column in expected.items():
        distance = pairwise_distance(trajectory, other, norm_order=norm_order)
        assert torch.isnan(distance[2]).all() and torch.isnan(distance[:, 1]).all()
        assert torch.allclose(distance[:2, 0], torch.tensor(finite_column, dtype=torch.float64))


def test_pairwise_distance_nonfinite_points():
    parameter = torch.tensor(1.0, requires_grad=True)
    for norm_order in (0.5, 1, 2, float("inf")):
        trajectory = (parameter * torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 5.0]]))[..., None].clone()
        trajectory[0, 0, 0] = float("nan")
        trajectory[1, 2, 0] = float("inf")
        distance = pairwise_distance(trajectory, norm_order=norm_order)
        assert torch.isnan(distance[0, 0]).all() and torch.isnan(distance[0, :, 0]).all()
        assert torch.isnan(distance[1, 2]).all() and torch.isnan(distance[1, :, 2]).all()
        assert torch.equal(distance[0, 1:, 1:], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        gradient = torch.autograd.grad(distance[0, 1:, 1:].sum(), parameter)[0]
        assert gradient.item() == 2.0


def test_rosenstein_nonfinite_sample_outside_embedding():
    # Regression: with embedding_dim=2 and delay=30 on 50 samples, samples
    # 20-29 enter no embedded point; a NaN there must still mark the row.
    torch.manual_seed(1)
    batch = torch.randn(2, 50, 1, dtype=torch.float64)
    batch[1, 25] = float("nan")
    result = rosenstein_divergence(batch, [0, 1], embedding_dim=2, delay=30, theiler_window=1)
    assert torch.isnan(result[:, 1]).all()
    assert torch.equal(result[:, 0], rosenstein_divergence(batch[0], [0, 1], embedding_dim=2, delay=30, theiler_window=1))


def test_mask_nonfinite_rows_empty_batch():
    from torchnlts._shape import _mask_nonfinite_rows

    values, nonfinite = _mask_nonfinite_rows(torch.zeros(0, 5))
    assert values.shape == (0, 5) and nonfinite.shape == (0,)


def test_knn_search_nonfinite_candidates_are_unavailable():
    from torchnlts.neighbors.knn import knn_search

    distances, indices = knn_search(torch.tensor([0.0]), 2, other=torch.tensor([float("inf"), 1.0]))
    assert distances.tolist() == [[1.0, float("inf")]] and indices.tolist() == [[1, -1]]


def test_kantz_nonfinite_sample_outside_embedding():
    # Samples 20-29 enter no embedded point for embedding_dim=2, delay=30.
    series = torch.arange(50, dtype=torch.float64)
    series[25] = float("nan")
    assert torch.isnan(kantz_divergence(series, 0, radius=100.0, embedding_dim=2, delay=30))


def test_knn_search_nonfinite_self_query():
    from torchnlts.neighbors.knn import knn_search

    distances, indices = knn_search(torch.tensor([0.0, float("nan"), 1.0]), 2)
    assert torch.isinf(distances[1]).all() and (indices[1] == -1).all()
