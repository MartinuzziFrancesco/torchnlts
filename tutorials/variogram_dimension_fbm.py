"""Reproduces, in reduced form, a finding from Gneiting, Sevcikova &
Percival (2012), "Estimators of fractal dimension: assessing the
roughness of time series and spatial data", Statistical Science 27(2),
247-277, Section 3.2: on clean Gaussian data the variogram estimator
(power p=2) has lower RMSE than the madogram (p=1). Their Fig. 15 shows
this for stationary Gaussian processes with powered-exponential
covariance; the text states that "this ranking is retained" for
fractional Brownian motion at sufficiently large sample sizes, which is
the case checked here. Both estimators use lags 1 and 2 only (their
eq. 18, L=2).

The paper still recommends the madogram as the general-purpose default,
for its robustness to outliers and model misspecification, which this
noise-free fBm simulation does not exercise.

9 Hurst values (0.1 to 0.9), 30 realizations each of a length-4096 fBm
(fractional Gaussian noise via exact Davies-Harte circulant embedding,
cumulatively summed) -- well short of the paper's own much larger
simulation grid; the RMSE ordering, not the exact numbers, is what's
being checked. Higuchi's dimension (a curve-length estimator, not part
of the paper's own comparison) is printed alongside as an additional
reference point, not asserted against any target.
"""

import torch

from torchnlts.dimension.higuchi import higuchi_dimension, higuchi_length
from torchnlts.dimension.variogram import power_variogram, variogram_dimension


def fractional_gaussian_noise(hurst: float, length: int, dtype=torch.float64) -> torch.Tensor:
    """fGn via Davies & Harte (1987) circulant embedding, ported from
    the reference implementation in the `fbm` package (crflynn/fbm,
    `_daviesharte`)."""

    def autocovariance(lag: int) -> float:
        return 0.5 * (abs(lag - 1) ** (2 * hurst) - 2 * abs(lag) ** (2 * hurst) + abs(lag + 1) ** (2 * hurst))

    row = [autocovariance(0)] + [autocovariance(index) for index in range(1, length)] + [0.0] + [autocovariance(index) for index in range(length - 1, 0, -1)]
    eigenvalues = torch.fft.fft(torch.tensor(row, dtype=dtype).to(torch.complex128)).real
    # Clamping only removes rounding-level negatives; a genuinely negative
    # eigenvalue means the embedding is invalid for this (hurst, length).
    assert (eigenvalues >= -1e-6).all(), "Davies-Harte embedding is not positive semi-definite for this (hurst, length)"
    eigenvalues = eigenvalues.clamp(min=0.0)

    gaussian_one = torch.randn(2 * length, dtype=dtype)
    gaussian_two = torch.randn(2 * length, dtype=dtype)
    weights = torch.zeros(2 * length, dtype=torch.complex128)
    weights[0] = torch.sqrt(eigenvalues[0] / (2 * length)) * gaussian_one[0]
    weights[length] = torch.sqrt(eigenvalues[length] / (2 * length)) * gaussian_two[0]
    middle = torch.arange(1, length)
    weights[middle] = torch.sqrt(eigenvalues[middle] / (4 * length)) * (gaussian_one[middle] + 1j * gaussian_two[middle])
    weights[2 * length - middle] = torch.sqrt(eigenvalues[2 * length - middle] / (4 * length)) * (gaussian_one[middle] - 1j * gaussian_two[middle])
    return torch.fft.fft(weights).real[:length]


torch.manual_seed(0)
hurst_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_realizations = 30
num_points = 4096
lag = torch.tensor([1.0, 2.0], dtype=torch.float64)
higuchi_strides = list(range(1, 11))
higuchi_stride_tensor = torch.tensor(higuchi_strides, dtype=torch.float64)

print(f"{'true D':>8}  {'madogram':>10}  {'variogram':>10}  {'higuchi':>10}")
squared_error = {"madogram": [], "variogram": [], "higuchi": []}
for hurst in hurst_values:
    true_dimension = 2.0 - hurst
    madogram_estimates, variogram_estimates, higuchi_estimates = [], [], []
    for _ in range(num_realizations):
        fbm = fractional_gaussian_noise(hurst, num_points).cumsum(0)
        madogram_estimates.append(variogram_dimension(lag, power_variogram(fbm, [1, 2], power=1.0), power=1.0).item())
        variogram_estimates.append(variogram_dimension(lag, power_variogram(fbm, [1, 2], power=2.0), power=2.0).item())
        higuchi_estimates.append(higuchi_dimension(higuchi_stride_tensor, higuchi_length(fbm, higuchi_strides)).item())
    mean_madogram = sum(madogram_estimates) / num_realizations
    mean_variogram = sum(variogram_estimates) / num_realizations
    mean_higuchi = sum(higuchi_estimates) / num_realizations
    print(f"{true_dimension:>8.2f}  {mean_madogram:>10.4f}  {mean_variogram:>10.4f}  {mean_higuchi:>10.4f}")
    squared_error["madogram"].extend((estimate - true_dimension) ** 2 for estimate in madogram_estimates)
    squared_error["variogram"].extend((estimate - true_dimension) ** 2 for estimate in variogram_estimates)
    squared_error["higuchi"].extend((estimate - true_dimension) ** 2 for estimate in higuchi_estimates)

rmse = {name: (sum(errors) / len(errors)) ** 0.5 for name, errors in squared_error.items()}
print(f"\nOverall RMSE across all H: madogram={rmse['madogram']:.4f}  variogram={rmse['variogram']:.4f}  higuchi={rmse['higuchi']:.4f}")

assert rmse["variogram"] < rmse["madogram"], "the paper's ranking on clean Gaussian data: variogram below madogram"
print("\nThe variogram (p=2) has lower RMSE than the madogram (p=1) on this clean fBm, the ranking the")
print("paper reports (its Fig. 15 and Section 3.2). The paper recommends the madogram for robustness to")
print("outliers, which this simulation does not test. Only the ranking is checked, not the paper's RMSE")
print("values. Higuchi is not in the paper's comparison and is printed for reference only.")
