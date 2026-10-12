"""numpy/pandas interoperability.

``as_tensor`` is wired into ``torchnlts._shape``, so every public function accepts numpy arrays and pandas
Series/DataFrame on input. ``to_numpy`` and ``to_pandas`` convert results back out. They are explicit rather than
automatic: converting a return value would silently break autodiff for anyone chaining torchnlts calls.

pandas is optional; ``to_pandas`` raises ``ImportError`` if it is not installed.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import pandas as pd
except ImportError:
    pd = None

__all__ = ["as_tensor", "to_numpy", "to_pandas"]


def as_tensor(data, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert a tensor, numpy array, pandas Series/DataFrame or array-like to a tensor.

    Mirrors ``torch.as_tensor`` (no copy where possible), with pandas input accepted.

    Args:
        data: Tensor, numpy array, pandas Series/DataFrame, or anything ``torch.as_tensor`` accepts.
        dtype: Optional dtype of the result.
    """
    if pd is not None and isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()
    return torch.as_tensor(data, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor (any device) to a numpy array, detaching gradients first.

    A numpy array is returned unchanged.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_pandas(tensor: torch.Tensor, *, index=None, columns=None):
    """Convert a 1D or 2D tensor to a pandas Series or DataFrame.

    Args:
        tensor: Shape ``[T]`` (-> ``Series``) or ``[T, D]`` (-> ``DataFrame``).
        index: Forwarded to the pandas constructor.
        columns: Forwarded to ``DataFrame`` (ignored for a 1D tensor).

    Returns:
        ``pandas.Series`` or ``pandas.DataFrame``.

    Raises:
        ImportError: If pandas is not installed.
        ValueError: If ``tensor`` has more than two dimensions.
    """
    if pd is None:
        raise ImportError("to_pandas requires pandas; install it with `pip install pandas`")
    array = to_numpy(tensor)
    if array.ndim == 1:
        return pd.Series(array, index=index)
    if array.ndim == 2:
        return pd.DataFrame(array, index=index, columns=columns)
    raise ValueError(
        f"to_pandas only supports 1D or 2D tensors, got shape {array.shape}; index a batch dimension first"
    )
