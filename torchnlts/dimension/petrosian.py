"""Petrosian fractal dimension: a count-based estimate of signal roughness via sign changes in the first difference.

Reference: Petrosian (1995), "Kolmogorov complexity of finite sequences and recognition of different preictal EEG
patterns", Proc. 8th IEEE Symposium on Computer-Based Medical Systems, 212-217. A zero difference counts as positive
and ``-0.0`` as negative, as in antropy's ``petrosian_fd``.

Gradient class: none (a discrete count; the output does not require grad).
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _ensure_batched,
    _high_precision_dtype,
    _mask_nonfinite_rows,
    _output_dtype,
    _require_univariate,
    _restore_batch,
)


def petrosian_dimension(trajectory) -> torch.Tensor:
    """Petrosian fractal dimension ``log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta)))``.

    ``N`` is the series length and ``N_delta`` the number of sign changes in the first difference.

    Reference: Petrosian (1995), Proc. 8th IEEE Symp. Computer-Based Medical Systems, 212.

    Gradient class: none (a discrete count; the output does not require grad even if the input does).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas Series/DataFrame.

    Returns:
        Petrosian dimension, shape ``[]`` unbatched or ``[B]`` batched, in the trajectory's float dtype (the default
        dtype for integer input). ``NaN`` if the series has fewer than 2 points and for any row containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1 or the trajectory is complex.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor, "petrosian_dimension")
    dtype = _output_dtype(tensor)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(dtype))  # [B, T]

    batch_size = values.shape[0]
    series_length = values.shape[1]

    if series_length < 2:
        result = torch.full((batch_size,), float("nan"), dtype=values.dtype, device=values.device)
    else:
        differences = values[:, 1:] - values[:, :-1]

        # signbit is True for -0.0 too (antropy's convention); diff(signbit) is 1 where the sign changes.
        sign_bits = torch.signbit(differences)  # [B, T-1]
        sign_changes = torch.diff(sign_bits.to(torch.int32), dim=1).abs().sum(dim=1)  # [B]

        # High precision so N + 0.4 * N_delta cannot overflow a low-precision input dtype.
        work_dtype = _high_precision_dtype(values.device)
        num_sign_changes = sign_changes.to(work_dtype)
        log_points = torch.log10(torch.tensor(float(series_length), dtype=work_dtype, device=values.device))
        result = log_points / (log_points + torch.log10(series_length / (series_length + 0.4 * num_sign_changes)))
        result = torch.where(nonfinite, float("nan"), result).to(dtype)

    return _restore_batch(result, ndim)
