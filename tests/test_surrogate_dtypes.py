"""Input dtype handling of the surrogate generators."""

from collections import Counter

import pytest
import torch

from tests.conftest import pseudo_random_series
from torchnlts.surrogate import (
    aaft_surrogate,
    iaaft_surrogate,
    phase_randomized_surrogate,
    random_shift_surrogate,
    random_shuffle_surrogate,
)

GENERATORS = [phase_randomized_surrogate, aaft_surrogate, iaaft_surrogate]


@pytest.mark.parametrize("surrogate", GENERATORS)
def test_integer_input_is_computed_in_the_default_float_dtype(surrogate):
    values = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    result = surrogate(values, generator=torch.Generator().manual_seed(0))
    expected = surrogate(values.to(torch.get_default_dtype()), generator=torch.Generator().manual_seed(0))
    assert result.dtype == torch.get_default_dtype()
    assert torch.equal(result, expected)


@pytest.mark.parametrize("surrogate", GENERATORS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_is_computed_in_float32_and_cast_back(surrogate, dtype):
    values = pseudo_random_series(500, 3).to(dtype)
    result = surrogate(values, generator=torch.Generator().manual_seed(1))
    expected = surrogate(values.float(), generator=torch.Generator().manual_seed(1)).to(dtype)
    assert result.dtype == dtype
    assert torch.equal(result, expected)


@pytest.mark.parametrize("surrogate", GENERATORS)
def test_complex_input_raises(surrogate):
    with pytest.raises(ValueError, match="complex"):
        surrogate(torch.ones(8, dtype=torch.complex64))


@pytest.mark.parametrize("surrogate", [random_shuffle_surrogate, random_shift_surrogate])
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64, torch.float16])
def test_shuffle_and_shift_only_move_values_so_they_keep_any_dtype(surrogate, dtype):
    values = (torch.arange(12) % 5).to(dtype)
    result = surrogate(values, generator=torch.Generator().manual_seed(0))
    assert result.dtype == dtype
    assert Counter(result.tolist()) == Counter(values.tolist())
