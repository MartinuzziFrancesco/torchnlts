"""Tests for torchnlts.convert (numpy/pandas interop) and its wiring
into torchnlts._shape, which every trajectory-consuming public function
goes through. Data conversion, not an algorithm, so no paper/Julia
ground truth applies here.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchnlts.convert import as_tensor, to_numpy, to_pandas
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.knn import knn_search
from torchnlts.neighbors.radius import radius_neighbors


def test_as_tensor_passthrough():
    t = torch.randn(5)
    assert as_tensor(t) is t


def test_as_tensor_from_numpy():
    array = np.array([1.0, 2.0, 3.0])
    t = as_tensor(array)
    assert isinstance(t, torch.Tensor)
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_as_tensor_from_list():
    t = as_tensor([1.0, 2.0, 3.0])
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))


def test_as_tensor_from_pandas_series():
    series = pd.Series([1.0, 2.0, 3.0])
    t = as_tensor(series)
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_as_tensor_from_pandas_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    t = as_tensor(df)
    assert torch.equal(t, torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64))


def test_as_tensor_dtype_override():
    t = as_tensor(np.array([1.0, 2.0]), dtype=torch.float32)
    assert t.dtype == torch.float32


def test_to_numpy_detaches_gradient():
    t = torch.randn(4, requires_grad=True)
    array = to_numpy(t)
    assert isinstance(array, np.ndarray)
    assert np.allclose(array, t.detach().numpy())


def test_to_numpy_passthrough():
    array = np.array([1.0, 2.0])
    assert to_numpy(array) is array


def test_to_pandas_1d_is_series():
    t = torch.tensor([1.0, 2.0, 3.0])
    series = to_pandas(t)
    assert isinstance(series, pd.Series)
    assert series.tolist() == [1.0, 2.0, 3.0]


def test_to_pandas_2d_is_dataframe():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    df = to_pandas(t, columns=["a", "b"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1.0, 3.0]


def test_to_pandas_3d_raises():
    with pytest.raises(ValueError):
        to_pandas(torch.randn(2, 3, 4))


def test_to_pandas_without_pandas_raises_import_error(monkeypatch):
    import torchnlts.convert as convert_module

    monkeypatch.setattr(convert_module, "pd", None)
    with pytest.raises(ImportError):
        to_pandas(torch.tensor([1.0, 2.0]))


def test_pairwise_distance_accepts_numpy_and_pandas():
    array = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    expected = pairwise_distance(torch.as_tensor(array))
    assert torch.equal(pairwise_distance(array), expected)
    assert torch.equal(pairwise_distance(pd.DataFrame(array)), expected)


def test_delay_embedding_accepts_pandas_series():
    series = pd.Series(range(10), dtype=float)
    expected = delay_embedding(torch.as_tensor(series.to_numpy()), embedding_dim=3, delay=2)
    assert torch.equal(delay_embedding(series, embedding_dim=3, delay=2), expected)


def test_knn_search_and_radius_neighbors_accept_numpy():
    array = np.random.default_rng(0).normal(size=(20, 2))
    dist_np, idx_np = knn_search(array, num_neighbors=3, theiler_window=1)
    dist_t, idx_t = knn_search(torch.as_tensor(array), num_neighbors=3, theiler_window=1)
    assert torch.equal(dist_np, dist_t)
    assert torch.equal(idx_np, idx_t)
    assert torch.equal(radius_neighbors(array, radius=1.0), radius_neighbors(torch.as_tensor(array), radius=1.0))
