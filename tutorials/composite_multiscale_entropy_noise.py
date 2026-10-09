"""Reproduces the qualitative claim of Wu, Wu, Lin, Wang & Lee (2013),
"Time series analysis using composite multiscale entropy" (Entropy
15(3), 1069-1084), Figs. 5-8: on short series, plain multiscale entropy (MSE) becomes erratic at large scales because the
coarse-grained series has few points, while composite multiscale entropy
(CMSE) averages the sample entropy over all ``s`` window offsets at
scale ``s`` and has a smaller spread across realizations.

White noise and 1/f noise are generated many times at a fixed short
length, both normalized to zero mean/unit variance with
``radius = 0.15``. The tutorial prints the mean curve of each noise and
the standard deviation across realizations of MSE versus CMSE. Expected:
white noise decays with scale, 1/f noise stays roughly flat, and CMSE's
standard deviation is below MSE's at the larger scales.
"""

import torch

from torchnlts.entropy import composite_multiscale_entropy, multiscale_entropy

torch.manual_seed(0)
num_points = 1500
num_realizations = 20
scales = [1, 2, 4, 6, 8, 10]
radius = 0.15


def normalize(series: torch.Tensor) -> torch.Tensor:
    return (series - series.mean()) / series.std()


def white_noise(num_points: int) -> torch.Tensor:
    return normalize(torch.randn(num_points, dtype=torch.float64))


def pink_noise(num_points: int) -> torch.Tensor:
    """1/f-power-spectrum noise via spectral filtering of white noise."""
    spectrum = torch.fft.rfft(torch.randn(num_points, dtype=torch.float64))
    freqs = torch.fft.rfftfreq(num_points).to(torch.float64)
    freqs[0] = freqs[1]  # avoid a division by zero at DC
    return normalize(torch.fft.irfft(spectrum / torch.sqrt(freqs), n=num_points))


curves = {}
for name, generate in (("white", white_noise), ("1/f", pink_noise)):
    series = [generate(num_points) for _ in range(num_realizations)]
    curves[name] = {
        "MSE": torch.stack([multiscale_entropy(row, scales, radius) for row in series]),
        "CMSE": torch.stack([composite_multiscale_entropy(row, scales, radius) for row in series]),
    }

print(
    f"{'scale':>5}  "
    + "  ".join(
        f"{name + ' ' + kind + ' mean':>14}  {name + ' ' + kind + ' std':>13}"
        for name in curves
        for kind in ("MSE", "CMSE")
    )
)
for idx, scale in enumerate(scales):
    cells = []
    for name in curves:
        for kind in ("MSE", "CMSE"):
            values = curves[name][kind][:, idx]
            cells.append(f"{values.mean():>14.4f}  {values.std():>13.4f}")
    print(f"{scale:>5}  " + "  ".join(cells))

large = slice(len(scales) // 2, None)
for name in curves:
    mse_std = curves[name]["MSE"].std(dim=0)[large].mean()
    cmse_std = curves[name]["CMSE"].std(dim=0)[large].mean()
    print(f"\n{name}: mean std over scales {scales[large]}: MSE={mse_std:.4f}, CMSE={cmse_std:.4f}")
    assert cmse_std < mse_std, f"CMSE should vary less across realizations than MSE for {name} noise"

white_cmse = curves["white"]["CMSE"].mean(dim=0)
pink_cmse = curves["1/f"]["CMSE"].mean(dim=0)
assert white_cmse[0] > pink_cmse[0], "at scale 1 white noise should be the more entropic"
assert white_cmse[-1] < white_cmse[0] - 0.5, "white-noise CMSE should decay with scale"
assert (pink_cmse[-1] - pink_cmse[0]).abs() < 0.5 * (white_cmse[0] - white_cmse[-1]), (
    "1/f CMSE should stay flatter than white"
)
print("\nAll checks passed: CMSE has lower spread than MSE at large scales; white noise decays, 1/f stays flat.")
