# torchnlts

Nonlinear time series analysis in PyTorch. Recurrence analysis, fractal
dimensions, entropies, Lyapunov exponents and causality measures are written
as tensor operations, so they run on CPU or GPU and can be differentiated
where the underlying quantity allows it.

## Contents

Functions and classes are importable from their subpackage, for example
`from torchnlts.entropy import sample_entropy`.

- `torchnlts.embedding`: delay embedding, fractional delay embedding, false
  nearest neighbors (`hard_fnn`), Cao's E1 and E2, and delay selection
  (`autocorrelation`, `auto_mutual_information`, `estimate_delay`).
- `torchnlts.neighbors`: pairwise distance, k-nearest-neighbor and radius
  search, Theiler window masks, trajectory kernels.
- `torchnlts.recurrence`: hard and soft recurrence, cross-recurrence and
  joint-recurrence matrices; recurrence quantification: recurrence rate,
  determinism, diagonal line statistics, laminarity, trapping time,
  vertical line statistics (`vl_max`, `vl_entropy`), recurrence time
  entropy, `ratio`, `divergence` and sliding-window `windowed_rqa`. The
  recurrence matrix threshold can be a fixed `epsilon` (optionally relative
  to a distance `scale`), a target `recurrence_rate` or a `local_rate`.
- `torchnlts.dimension`: correlation sum and dimension, generalized (D_q)
  dimension, Higuchi, Katz, Petrosian and variogram estimators.
- `torchnlts.entropy`: sample, approximate, permutation (including weighted
  and amplitude-aware variants), dispersion, fuzzy, Renyi and Tsallis
  entropies, multiscale and composite multiscale entropy, Lempel-Ziv
  complexity, mutual information (KSG1 and KSG2 estimators). Sample,
  approximate and multiscale entropy have soft variants.
- `torchnlts.fluctuations`: detrended fluctuation analysis.
- `torchnlts.chaos`: the 0-1 test for chaos.
- `torchnlts.lyapunov`: Rosenstein and Kantz divergence curves, Eckmann
  spectrum from data, Lyapunov spectrum from a known map or flow, step
  Jacobian, Kaplan-Yorke dimension.
- `torchnlts.causality`: convergent cross mapping, transfer entropy.
- `torchnlts.surrogate`: phase-randomized, AAFT, IAAFT, random shuffle and
  random shift surrogates; `surrogate_test` for hypothesis tests.
- `torchnlts.sciml`: loss modules built on the above (`LyapunovLoss`,
  `CorrelationDimensionLoss`, `RecurrenceLoss`, `AttractorLoss`,
  `SinkhornLoss`) for training models of dynamical systems.
- `torchnlts.scaling`: scaling-region detection for the slope estimators
  (`linear_regions`, `largest_linear_region`, `scaling_slope`).
- `torchnlts.systems`: trajectories of reference systems from
  [dysts](https://github.com/williamgilpin/dysts).
- `torchnlts.convert`: `as_tensor`, `to_numpy`, `to_pandas`. Functions also
  accept numpy arrays and pandas objects directly.

## Installation

Requires Python 3.10 or later and PyTorch 2.2 or later. From a checkout:

```
pip install -e .
```

Optional extras: `sde` (torchsde), `pandas`, and `dev` (pytest and the
reference packages EntropyHub, antropy and nolds used in the tests).

## Example

Correlation dimension of the Lorenz attractor, and the gradient of a soft
sample entropy with respect to the input series:

```python
import torch
from torchnlts.systems import trajectory
from torchnlts.dimension import correlation_dimension, correlation_sum
from torchnlts.entropy import soft_sample_entropy

state = trajectory("Lorenz", 5000).to(torch.float64)
state = (state - state.mean(dim=0)) / state.std(dim=0)

radius = torch.logspace(-1.5, -0.3, 12, dtype=torch.float64)
sums = correlation_sum(state, radius, theiler_window=10)
print(correlation_dimension(radius, sums))  # about 2.10; literature value is near 2.05

signal = state[:1000, 0].clone().requires_grad_()
entropy = soft_sample_entropy(signal, temperature=0.2, embedding_dim=2)
entropy.backward()
print(entropy.item(), signal.grad.shape)  # 0.157, torch.Size([1000])
```

The Lorenz estimate depends on the radius range chosen. Scripts reproducing
published results, one per method, are in `tutorials/`.

## Gradient classes

Each function has a gradient class: exact, smooth (a differentiable
relaxation of a hard threshold), piecewise, implicit, or none. Hard-threshold
functions such as `sample_entropy` have no gradient; their `soft_*`
counterparts do. `docs/primitives.rst` lists the class of every function,
the paper it follows, and the Julia function it was checked against where one
exists (DynamicalSystems.jl packages such as DelayEmbeddings.jl,
RecurrenceAnalysis.jl and FractalDimensions.jl). Entropy functions are also
compared with EntropyHub, antropy and nolds.

## Documentation and tests

```
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```

```
pip install -e ".[dev]"
pytest
```

Tests that compare against Julia call `julia` with the environments in
`juliaenv/` and `juliaenv_causality/`, so they need a Julia installation.

## License

MIT. See `LICENSE`.
