"""Petrosian fractal dimension: a count-based estimate of signal roughness
via sign changes in the first difference.

Reference: Petrosian, A. (1995), "Kolmogorov complexity of finite sequences
and recognition of different preictal EEG patterns", Proc. 8th IEEE Symposium
on Computer-Based Medical Systems, 212-217.

Gradient class: none (based on sign changes in the derivative, a discrete
count independent of the input's magnitude; the output does not require or
accumulate gradients).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _mask_nonfinite_rows, _output_dtype


def _require_univariate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(
            f"petrosian_dimension only supports a univariate (D=1) trajectory, "
            f"got feature dim {tensor.shape[-1]}"
        )


def petrosian_dimension(trajectory) -> torch.Tensor:
    """Petrosian fractal dimension: D = log10(N) / (log10(N) +
    log10(N / (N + 0.4 * N_delta))), where N is series length and N_delta
    is the count of sign changes in the first difference.

    Gradient class: none (a discrete count-based statistic; output does not
    require grad and does not flow gradients backward even if input does).

    Args:
        trajectory: Shape ``[T]`` or ``[T, 1]``/``[B, T, 1]`` (univariate
            only). A tensor, numpy array, or pandas Series/DataFrame.

    Returns:
        Petrosian dimension estimate(s), shape ``[]`` for an unbatched
        trajectory or ``[B]`` for a batched one. NaN if the series is
        too short (N < 2). ``NaN`` for any row containing NaN/inf.

    Raises:
        ValueError: If trajectory has feature dim other than 1, or is
            complex.
    """
    tensor, ndim = _ensure_batched(trajectory)
    _require_univariate(tensor)
    if tensor.is_complex():
        raise ValueError("petrosian_dimension does not support complex input")
    dtype = _output_dtype(tensor)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(dtype))  # [B, T]

    batch_size = values.shape[0]
    series_length = values.shape[1]

    # Edge case: series too short to have a difference
    if series_length < 2:
        result = torch.full((batch_size,), float("nan"), dtype=values.dtype, device=values.device)
    else:
        # Compute first difference: [B, T-1]
        differences = values[:, 1:] - values[:, :-1]

        # Count sign changes in the difference:
        # signbit is True for negative values and -0.0 (antropy's convention).
        # diff(signbit) gives 1 where the sign changes.
        sign_bits = torch.signbit(differences)  # [B, T-1]
        sign_changes = torch.diff(sign_bits.to(torch.int32), dim=1).abs().sum(dim=1)  # [B]

        # float64 so N + 0.4 * N_delta cannot overflow a low-precision input dtype.
        num_sign_changes = sign_changes.to(torch.float64)
        log_points = torch.log10(torch.tensor(float(series_length), dtype=torch.float64, device=values.device))
        result = log_points / (log_points + torch.log10(series_length / (series_length + 0.4 * num_sign_changes)))
        result = torch.where(nonfinite, float("nan"), result).to(dtype)

    return result if ndim == 3 else result.squeeze(0)
