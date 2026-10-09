"""Gottwald-Melbourne 0-1 test for chaos on a scalar time series.

Reference: Gottwald, G.A. and Melbourne, I. (2009), "On the implementation
of the 0-1 test for chaos", SIAM J. Appl. Dyn. Syst. 8(1), 129-145 (the
test itself: Proc. R. Soc. A 460, 603-611, 2004). For a frequency
``c`` in ``(0, pi)`` the series ``phi(1..N)`` is mapped to the translation
variables ``p(n) = sum_j phi(j) cos(j c)``, ``q(n) = sum_j phi(j) sin(j c)``
(``j <= n``). Their mean square displacement is
``M(n) = 1/(N - n) sum_{j=1}^{N-n} [p(j+n) - p(j)]**2 + [q(j+n) - q(j)]**2``;
the modified displacement is ``D(n) = M(n) - V_osc(c, n)`` with
``V_osc = mean(phi)**2 (sin(n c / 2) / sin(c / 2))**2`` (equal to the paper's (1 - cos(n c)) / (1 - cos c)). The growth rate
``K_c`` is the correlation coefficient of ``D(n)`` with ``n`` for
``n = 1..max_lag`` (the paper's correlation method), and ``K`` is the
median of ``K_c`` over the frequencies: ``K`` near 0 means regular dynamics,
near 1 chaotic. The caller supplies the frequencies; the paper draws
about 100 at random from ``(pi/5, 4*pi/5)`` to avoid resonances near
0 and pi, and ``max_lag`` defaults to its ``N // 10``. Continuous-time data
must not be oversampled (paper, section 6).

Convention: the paper's finite-N form above, with windows ``j = 1..N - n``
normalized by ``N - n`` (the paper only defines ``M`` as an ``N -> inf``
limit). ChaosTools.jl's ``testchaos01(x, c, N0)`` starts its sums at
``p(1) = 0`` and uses windows ending at ``x[N - 1]``, so it never uses the
last sample, where this module never uses the first window start; the two
agree exactly once ChaosTools is given ``[x[2:N]; x[1]]`` (the modulus of a
window sum does not depend on a common phase shift, and the mean is
unchanged), which the tests do.

Deviations from the paper: only the correlation method is implemented, and
not the noise-damping term ``D*`` of section 7. ``V_osc`` is not subtracted
numerically: with ``phi = mean + psi`` the window sum splits into a closed-form
geometric part with modulus ``sqrt(V_osc)`` and a window sum ``B`` of
``psi e^{ijc}``, so ``D(n)`` is the window mean of ``|B|**2 + 2 Re(A conj B)``
and ``V_osc`` cancels exactly (this stays accurate for a small ``c`` and for a
series with a large offset; the first sample is subtracted before the mean).
Limitation: ``B`` comes from differences of prefix sums, so its absolute error
is about ``eps * T * max|psi|``, and the lag loop is a Python loop
(``O(max_lag * T)`` per frequency and row); an FFT autocorrelation would remove
the loop at the price of more cancellation error. Limitation: the derivative of
``sin(n c / 2) / sin(c / 2)`` with respect to ``c`` cancels for small ``c``
(about 3% relative error at ``c = 1e-8``, accurate from ``c = 1e-6``), so
gradients with respect to frequencies far below the recommended range are
inaccurate; values are not affected. The computation is always in
float64. Constant series are ``NaN`` (detected exactly); other rows are ``NaN``
when ``D(n)`` varies by no more than rounding of its own terms.

Gradient class: smooth for ``K_c`` (a correlation coefficient of
differentiable quantities) wherever ``D(n)`` is not constant; piecewise for
``K`` (the median selects an order statistic of ``K_c``).
"""

from __future__ import annotations

import math

import torch

from torchnlts._shape import _as_positive_int_list, _as_sweep, _ensure_batched, _mask_nonfinite_rows, _output_dtype


def _validate(tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != 1:
        raise ValueError(
            f"the 0-1 test only supports a univariate (D=1) trajectory, got feature dim {tensor.shape[-1]}"
        )
    if tensor.is_complex():
        raise ValueError("the 0-1 test does not support complex input")


def _frequencies(frequencies, device) -> tuple[torch.Tensor, bool]:
    """Frequencies as a float64 ``[C]`` tensor, and whether a sweep was given."""
    sweep = _as_sweep(frequencies, device).to(torch.float64)
    if sweep.dim() > 1 or sweep.numel() == 0:
        raise ValueError(f"frequencies must be a scalar or a non-empty 1-D sweep, got shape {tuple(sweep.shape)}")
    if not bool(((sweep > 0) & (sweep < math.pi)).all()):
        raise ValueError(f"frequencies must lie in (0, pi), got {sweep.tolist()}")
    return sweep.reshape(-1), sweep.dim() > 0


def _max_lag(max_lag, num_points: int) -> int | None:
    """Validated ``max_lag`` (``N // 10`` by default); ``None`` when the
    series is too short for it (fewer than 2 lags, or no window at the largest lag)."""
    if max_lag is None:
        lag = num_points // 10
    else:
        items, swept = _as_positive_int_list(max_lag, "max_lag")
        if swept:
            raise ValueError("max_lag must be a single integer")
        lag = items[0]
        if lag < 2:
            raise ValueError(f"max_lag must be >= 2 to correlate against n, got {lag}")
    return lag if 2 <= lag <= num_points - 1 else None


def _growth_rate(tensor: torch.Tensor, frequencies, max_lag) -> tuple[torch.Tensor, bool]:
    """``K_c`` ``[C, B]`` in the result dtype for a batched ``[B, T, 1]``
    ``tensor``, and whether ``frequencies`` was a sweep."""
    _validate(tensor)
    dtype = _output_dtype(tensor)
    values, nonfinite = _mask_nonfinite_rows(tensor.squeeze(-1).to(torch.float64))  # [B, N]
    frequency, swept = _frequencies(frequencies, values.device)
    num_points = values.shape[-1]
    lag = _max_lag(max_lag, num_points)

    if lag is None:
        # Zero-weighted sums keep the result connected to both inputs.
        result = torch.full(
            (frequency.numel(), values.shape[0]), float("nan"), dtype=torch.float64, device=values.device
        )
        result = result + values.sum(dim=-1) * 0 + frequency.sum() * 0
    else:
        # phi = mean + psi. A window sum of phi e^{ijc} is A + B with
        # A = mean * (closed-form geometric sum), |A|^2 = V_osc exactly, and
        # B the window sum of psi e^{ijc}, so D = mean(|B|^2 + 2 Re(A conj B))
        # has V_osc cancelled analytically instead of by subtracting two
        # large numbers. The first sample is removed before the mean so a
        # large common offset cancels exactly.
        constant = (values == values[:, :1]).all(dim=-1)  # [B]
        offset = values[:, :1]
        shifted = values - offset
        shifted_mean = shifted.mean(dim=-1, keepdim=True)
        deviation = shifted - shifted_mean  # psi
        mean = (offset + shifted_mean).squeeze(-1)[None, :, None]  # [1, B, 1]
        index = torch.arange(1, num_points + 1, dtype=torch.float64, device=values.device)
        phase = frequency[:, None] * index  # [C, N]
        translation_cos = (deviation[None] * torch.cos(phase)[:, None]).cumsum(dim=-1)  # [C, B, N]
        translation_sin = (deviation[None] * torch.sin(phase)[:, None]).cumsum(dim=-1)
        half_frequency = (frequency / 2)[:, None, None]
        displacement, rounding_scale = [], []
        for step in range(1, lag + 1):
            window_cos = translation_cos[..., step:] - translation_cos[..., : num_points - step]
            window_sin = translation_sin[..., step:] - translation_sin[..., : num_points - step]
            # Window j+1..j+step of sum e^{ijc}: amplitude sin(step c/2) / sin(c/2), centre phase (j + (step + 1)/2) c.
            amplitude = torch.sin(step * half_frequency) / torch.sin(half_frequency)
            centre = frequency[:, None] * (index[: num_points - step] + (step + 1) / 2)  # [C, N - step]
            cross = amplitude * (torch.cos(centre)[:, None] * window_cos + torch.sin(centre)[:, None] * window_sin)
            square = (window_cos**2 + window_sin**2).mean(dim=-1)
            displacement.append(square + 2 * mean[..., 0] * cross.mean(dim=-1))
            rounding_scale.append(square + 2 * mean[..., 0].abs() * cross.detach().abs().mean(dim=-1))
        modified = torch.stack(displacement, dim=-1)  # [C, B, L]
        rounding_scale = torch.stack(rounding_scale, dim=-1).detach()
        lags = torch.arange(1, lag + 1, dtype=torch.float64, device=values.device)
        centered = modified - modified.mean(dim=-1, keepdim=True)
        lags_centered = lags - lags.mean()
        # Rows whose D(n) is constant to within rounding of its own terms
        # have no defined correlation; constant and non-finite rows never
        # have one. They get the lag profile as safe operands so no 0/0
        # reaches backward.
        with torch.no_grad():
            degenerate = (
                constant[None, :]
                | nonfinite[None, :]
                | (centered.abs().amax(dim=-1) <= 64 * torch.finfo(torch.float64).eps * rounding_scale.amax(dim=-1))
            )
        safe = torch.where(degenerate[..., None], lags_centered.expand_as(centered), centered)
        scale = safe.detach().abs().amax(dim=-1, keepdim=True)
        safe = safe / scale
        correlation = (safe * lags_centered).sum(dim=-1) / (
            safe.square().sum(dim=-1) * lags_centered.square().sum()
        ).sqrt()
        result = torch.where(degenerate, torch.full_like(correlation, float("nan")), correlation)

    return result.to(dtype), swept


def zero_one_growth_rate(trajectory, frequencies, max_lag: int | None = None) -> torch.Tensor:
    """Asymptotic growth rate ``K_c`` of the modified mean square
    displacement of the 0-1 test, one per frequency.

    Reference: Gottwald and Melbourne (2009), SIAM J. Appl. Dyn. Syst. 8(1), 129.

    Gradient class: smooth (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        frequencies: One frequency ``c`` or a 1-D sweep, each in ``(0, pi)``
            (the paper recommends ``(pi/5, 4*pi/5)``).
        max_lag: Largest lag ``n`` of the displacement, ``>= 2``. Defaults
            to ``T // 10``.

    Returns:
        ``K_c``, shape ``[C]``/``[C, B]`` for a sweep of ``C`` frequencies
        (``[]``/``[B]`` for a scalar frequency), for an unbatched/batched
        trajectory. ``NaN`` where the default ``max_lag`` (``T // 10``) is below 2 or
        any ``max_lag`` exceeds ``T - 1`` (a series too short), for a
        constant series, and where ``D(n)`` is constant to within rounding. ``NaN`` for any row
        containing NaN/inf.

    Raises:
        ValueError: If the feature dim is not 1, the trajectory is complex,
            a frequency is not real and in ``(0, pi)`` or the sweep is not
            1-D and non-empty, or ``max_lag`` is not a single integer ``>= 2``.
    """
    tensor, ndim = _ensure_batched(trajectory)
    result, swept = _growth_rate(tensor, frequencies, max_lag)
    if not swept:
        result = result.squeeze(0)
    return result if ndim == 3 else result.squeeze(-1)


def zero_one_test(trajectory, frequencies, max_lag: int | None = None) -> torch.Tensor:
    """The 0-1 test statistic ``K``: the median over ``frequencies`` of
    :func:`~torchnlts.chaos.zero_one_growth_rate`. ``K`` near 0 indicates regular dynamics,
    near 1 chaotic.

    Reference: Gottwald and Melbourne (2009), SIAM J. Appl. Dyn. Syst. 8(1), 129.

    Gradient class: piecewise (see module docstring).

    Args:
        trajectory: As in :func:`~torchnlts.chaos.zero_one_growth_rate`.
        frequencies: A 1-D sweep of frequencies in ``(0, pi)``, as in
            :func:`~torchnlts.chaos.zero_one_growth_rate` (a scalar gives its ``K_c``).
        max_lag: As in :func:`~torchnlts.chaos.zero_one_growth_rate`.

    Returns:
        ``K``, shape ``[]`` unbatched or ``[B]`` batched. For an even
        number of frequencies the median is the mean of the two middle
        values. ``NaN`` if any ``K_c`` is ``NaN``.

    Raises:
        ValueError: As in :func:`~torchnlts.chaos.zero_one_growth_rate`.
    """
    tensor, ndim = _ensure_batched(trajectory)
    growth_rate, _ = _growth_rate(tensor, frequencies, max_lag)  # [C, B]
    ordered = growth_rate.sort(dim=0).values
    count = ordered.shape[0]
    median = (ordered[(count - 1) // 2] + ordered[count // 2]) / 2
    median = torch.where(torch.isnan(growth_rate).any(dim=0), torch.full_like(median, float("nan")), median)
    return median if ndim == 3 else median.squeeze(-1)
