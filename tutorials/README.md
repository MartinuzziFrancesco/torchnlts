# Tutorials

Hand-holding walkthroughs of library functionality, using published methods on
explicit benchmarks as runnable checks alongside the unit tests. Some
reproduce known numerical values; others illustrate a method on synthetic
data rather than replicate a paper's exact experiment. One script per
benchmark; add one whenever a new module lands a result worth reproducing.

- `delay_embedding_lorenz.py`: Takens (1981) -- a scalar observable's
  delay embedding is diffeomorphic to the original attractor.
- `numpy_pandas_interop.py`: using torchnlts functions directly on
  numpy/pandas data, and converting results back with
  `torchnlts.convert`.
- `embedding_fnn_cao.py`: Kennel, Brown & Abarbanel (1992) + Cao (1997)
  -- choosing a delay embedding's dimension from a scalar observable
  alone.
- `recurrence_rqa.py`: Marwan, Romano, Thiel & Kurths (2007) --
  compare DET for sine, Lorenz x, and white noise at matched
  recurrence rate; this comparison is not a general determinism test.
- `correlation_dimension_lorenz.py`: Grassberger & Procaccia (1983) --
  the Lorenz attractor's fractal correlation dimension, estimated from
  a trajectory alone.
- `entropy_measures.py`: Bandt & Pompe (2002) + Richman & Moorman
  (2000) -- compare permutation and sample entropy for sine, Lorenz x, and
  white noise at fixed sampling and embedding settings.
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
  hypothesis-testing methodology, using IAAFT surrogates and a
  lower-tail rank test against a statically transformed Gaussian-linear null.
- `convergent_cross_mapping.py`: Sugihara, May, Ye, Hsieh, Deyle,
  Fogarty & Munch (2012) -- cross-map skill versus prefix-library size
  in a synthetic unidirectionally coupled map pair.
- `transfer_entropy_coupled_maps.py`: Schreiber (2000) -- directional
  predictive information in the same synthetic maps as
  `convergent_cross_mapping.py`, estimated with Frenzel-Pompe (2007)
  nearest-neighbor conditional mutual information.
- `fprm_missing_dynamics.py`: Wu, Gao, Tang, An, Sun, Aihara, An &
  Kurths (2026) -- imputing a missing variable from a fully observed
  one via delay embedding + Gaussian process regression (FPRM),
  reproducing their ecology-map and Lorenz r=90%-missing results in
  the same range.
- `sinkhorn_vs_mmd_invariant_matching.py`: Jiang, Lu, Orlova & Willett
  (2023) -- training a differentiable Lorenz vector field with a
  whole-attractor distributional loss added to MSE, comparing
  `torchnlts.sciml.losses.SinkhornLoss` (their entropic-OT loss,
  new in this tutorial) against `AttractorLoss` (MMD, already in the
  library) on Lyapunov-exponent and correlation-dimension recovery.
- `multiscale_entropy_noise.py`: Costa, Goldberger & Peng (2002) --
  single-scale sample entropy ranks white noise as more "complex" than
  1/f noise, and multiscale entropy (`torchnlts.entropy.multiscale`)
  reverses that ranking by the largest coarse-graining scale swept.
- `lempel_ziv_logistic_map.py`: Lempel & Ziv (1976) + Kaspar & Schuster
  (1987) -- Lempel-Ziv complexity across the logistic map's
  period-doubling route to chaos: near-minimal at periodic parameters
  (including a periodic window inside the chaotic band), far higher at
  full chaos, and close to pure random noise there.
- `dispersion_vs_permutation_entropy.py`: Rostaghi & Azami (2016) --
  permutation entropy is provably invariant under any strictly
  increasing transform of a trajectory (checked exactly), while
  dispersion entropy (`torchnlts.entropy.dispersion`), sensitive to
  actual amplitude values rather than just rank order, is not.
- `generalized_entropy_lorenz.py`: Rényi (1961) + Tsallis (1988) --
  applying `torchnlts.entropy.generalized`'s order-tunable entropies to
  the Lorenz x-coordinate's real ordinal-pattern distribution (reusing
  `permutation_entropy`'s own probability estimator): both recover its
  exact Shannon value at order 1, and Rényi entropy is non-increasing
  in order across the full sweep, as Rényi's own theorem requires.
