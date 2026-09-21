# Tutorials

Hand-holding walkthroughs of library functionality, each reproducing a
specific published result as a runnable check that the implementation
matches the literature (not just the unit tests). One script per
result; add one whenever a new module lands a result worth reproducing.

For harder technical showcases (advanced, composed usage rather than
guided walkthroughs), see `../examples/`.

- `delay_embedding_lorenz.py`: Takens (1981) -- a scalar observable's
  delay embedding is diffeomorphic to the original attractor.
- `numpy_pandas_interop.py`: using torchnlts functions directly on
  numpy/pandas data, and converting results back with
  `torchnlts.convert`.
- `embedding_fnn_cao.py`: Kennel, Brown & Abarbanel (1992) + Cao (1997)
  -- choosing a delay embedding's dimension from a scalar observable
  alone.
- `recurrence_rqa.py`: Marwan, Romano, Thiel & Kurths (2007) --
  determinism separates deterministic signals (periodic or chaotic)
  from stochastic noise.
- `correlation_dimension_lorenz.py`: Grassberger & Procaccia (1983) --
  the Lorenz attractor's fractal correlation dimension, estimated from
  a trajectory alone.
- `entropy_measures.py`: Bandt & Pompe (2002) + Richman & Moorman
  (2000) -- permutation and sample entropy, same deterministic-vs-random
  distinction as `recurrence_rqa.py` from a different angle.
- `mutual_information_gaussians.py`: Kraskov, Stogbauer & Grassberger
  (2004) -- their KSG1 estimator checked against correlated Gaussians'
  analytically known mutual information.
- `lyapunov_from_data.py`: Rosenstein, Collins & De Luca (1993) + Kantz
  (1994) -- largest Lyapunov exponent from an observed time series,
  reproducing the logistic map's known ln(2).
- `lyapunov_spectrum_henon.py`: Benettin, Galgani, Giorgilli & Strelcyn
  (1980) + Geist, Parlitz & Lauterborn (1990) -- the full Lyapunov
  spectrum of a differentiable map, via `torch.func`.
- `surrogate_data_testing.py`: Theiler, Eubank, Longtin, Galdrikian &
  Farmer (1992) + Schreiber & Schmitz (1996) -- the surrogate data
  hypothesis-testing methodology, using IAAFT surrogates.
- `convergent_cross_mapping.py`: Sugihara, May, Ye, Hsieh, Deyle,
  Fogarty & Munch (2012) -- detecting which of two coupled variables
  drives the other.
- `transfer_entropy_coupled_maps.py`: Schreiber (2000) -- the same
  causal-direction question as `convergent_cross_mapping.py`, via an
  information-theoretic method instead.
