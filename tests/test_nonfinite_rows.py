import pytest
import torch

from torchnlts.dimension.higuchi import higuchi_length
from torchnlts.dimension.katz import katz_dimension
from torchnlts.dimension.petrosian import petrosian_dimension
from torchnlts.dimension.variogram import power_variogram
from torchnlts.fluctuations.dfa import dfa_fluctuation

FUNCTIONS = {
    "higuchi": lambda batch: higuchi_length(batch, [1, 2, 4]),
    "katz": katz_dimension,
    "katz_amplitude": lambda batch: katz_dimension(batch, amplitude_only=True),
    "petrosian": petrosian_dimension,
    "variogram": lambda batch: power_variogram(batch, [1, 2], power=0.5),
    "dfa": lambda batch: dfa_fluctuation(batch, [8, 16]),
}


def _batch():
    torch.manual_seed(0)
    batch = torch.randn(4, 64, 1, dtype=torch.float64)
    batch[1, 5] = float("nan")
    batch[2, 7] = float("inf")
    batch[3, 9] = -float("inf")
    return batch


@pytest.mark.parametrize("name", FUNCTIONS)
def test_nonfinite_rows_give_nan_and_leave_other_rows(name):
    function = FUNCTIONS[name]
    batch = _batch()
    result = function(batch)
    assert torch.isnan(result[..., 1:]).all()
    assert torch.equal(result[..., 0], function(batch[0]))


@pytest.mark.parametrize("name", [name for name in FUNCTIONS if name != "petrosian"])
def test_nonfinite_rows_keep_gradients_finite(name):
    batch = _batch().requires_grad_()
    FUNCTIONS[name](batch)[..., 0].sum().backward()
    assert torch.isfinite(batch.grad).all()
    assert batch.grad[0].abs().sum() > 0
    assert (batch.grad[1:] == 0).all()


@pytest.mark.parametrize("name", FUNCTIONS)
def test_nonfinite_rows_vmap(name):
    function = FUNCTIONS[name]
    batch = _batch()
    mapped = torch.vmap(function)(batch[:, None])
    assert torch.allclose(mapped.squeeze(-1).movedim(0, -1), function(batch), equal_nan=True)
