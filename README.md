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
  search, Theiler window masks, trajectory kernels, complexity-invariant
  distance.
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
  approximate and multiscale entropy also have soft variants.
- `torchnlts.fluctuations`: detrended fluctuation analysis.
- `torchnlts.chaos`: the 0-1 test for chaos.
- `torchnlts.lyapunov`: Rosenstein and Kantz divergence curves, Eckmann
  spectrum from data, Lyapunov spectrum from a known map or flow, step
  Jacobian, Kaplan-Yorke dimension.
- `torchnlts.causality`: convergent cross mapping, transfer entropy.
- `torchnlts.matrixprofile`: matrix profile (self-join and AB-join), motifs,
  discords and FLUSS regime boundaries.
- `torchnlts.surrogate`: phase-randomized, AAFT, IAAFT, random shuffle and
  random shift surrogates; `surrogate_test` for hypothesis tests.
- `torchnlts.sciml`: loss modules built on the above (`LyapunovLoss`,
  `CorrelationDimensionLoss`, `RecurrenceLoss`, `AttractorLoss`,
  `SinkhornLoss`) for training models of dynamical systems.
- `torchnlts.scaling`: scaling-region detection for the slope estimators
  (`linear_regions`, `largest_linear_region`, `scaling_slope`).
- `torchnlts.convert`: `as_tensor`, `to_numpy`, `to_pandas`. Functions also
  accept numpy arrays and pandas objects directly.

## Installation

Requires Python 3.10 or later and PyTorch 2.5 or later.

```shell
pip install torchnlts
```

Optional extras: `pandas`, `tutorials` (dysts and torchdiffeq, used by the
scripts in `tutorials/`) and `test` (pytest, dysts, and the reference
packages EntropyHub, antropy, nolds, stumpy and tsfresh), for example
`pip install "torchnlts[pandas]"`.

From a checkout:

```shell
pip install -e .
```

## Example

Correlation dimension of the Lorenz attractor, and the gradient of a soft
sample entropy with respect to the input series. The trajectory comes from
[dysts](https://github.com/williamgilpin/dysts) (`pip install "torchnlts[tutorials]"`):

```python
import torch
from dysts.flows import Lorenz
from torchnlts.dimension import correlation_dimension, correlation_sum
from torchnlts.entropy import soft_sample_entropy

state = torch.as_tensor(Lorenz().make_trajectory(5000)).to(torch.float64)
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
the paper it follows, and the reference implementation it was checked
against where one exists.

## Development

Formatting and linting use [ruff](https://github.com/astral-sh/ruff), pinned in
the `dev` dependency group (installing a group needs pip 25.1 or later) and
configured in `pyproject.toml`:

```shell
pip install --group dev
ruff format .
ruff check .
```

Tests:

```shell
pip install -e ".[test]"
pytest
```

The comparisons against Julia call `julia` with the environments in
`juliaenv/` and `juliaenv_causality/`. They are skipped when `julia` is not on
`PATH` (`pytest -rs` lists them). With `julia` on `PATH`, both environments must
be instantiated, otherwise the comparisons fail with Julia's error message:

```shell
julia --project=juliaenv -e 'import Pkg; Pkg.instantiate()'
julia --project=juliaenv_causality -e 'import Pkg; Pkg.instantiate()'
```

Setting `TORCHNLTS_REQUIRE_JULIA=1` turns a missing Julia into a failure
instead of a skip, as in the Julia CI job. One DFA comparison also needs a
development checkout of FluctuationAnalysis.jl in
`~/.julia/dev/FluctuationAnalysis` and is skipped without it.

Documentation:

```shell
pip install -r docs/requirements.txt
sphinx-build -W docs docs/_build/html
```

## License

MIT. See `LICENSE`.

## Citation

If you use torchnlts in your work, please cite it with the metadata in
`CITATION.cff` (GitHub shows it under "Cite this repository") or the BibTeX
entry in `CITATION.bib`:

```bibtex
@software{martinuzzi2026torchnlts,
  author = {Martinuzzi, Francesco},
  title = {torchnlts: Nonlinear time series analysis in PyTorch},
  year = {2026},
  url = {https://github.com/MartinuzziFrancesco/torchnlts}
}
```

## Acknowledgments

This work builds on previous software and publications. We checked the
correctness of our implementations against the results of these packages:

- Julia, from [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl):
  [DelayEmbeddings.jl](https://github.com/JuliaDynamics/DelayEmbeddings.jl),
  [RecurrenceAnalysis.jl](https://github.com/JuliaDynamics/RecurrenceAnalysis.jl),
  [FractalDimensions.jl](https://github.com/JuliaDynamics/FractalDimensions.jl),
  [ChaosTools.jl](https://github.com/JuliaDynamics/ChaosTools.jl),
  [ComplexityMeasures.jl](https://github.com/JuliaDynamics/ComplexityMeasures.jl),
  [TimeseriesSurrogates.jl](https://github.com/JuliaDynamics/TimeseriesSurrogates.jl)
  and [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl).
- Python: [EntropyHub](https://github.com/MattWillFlood/EntropyHub),
  [antropy](https://github.com/raphaelvallat/antropy),
  [nolds](https://github.com/CSchoel/nolds),
  [stumpy](https://github.com/stumpy-dev/stumpy) and
  [tsfresh](https://github.com/blue-yonder/tsfresh).

The tutorials and tests generate reference system trajectories with
[dysts](https://github.com/williamgilpin/dysts).

The design also draws on [TISEAN](https://www.pks.mpg.de/tisean/).
