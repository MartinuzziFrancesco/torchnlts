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
- `delay_selection_lorenz.py`: Fraser & Swinney (1986) -- choosing a delay
  embedding's delay from the first minimum of the self mutual information of
  Lorenz x, compared with autocorrelation criteria.
- `recurrence_rqa.py`: Marwan, Romano, Thiel & Kurths (2007) --
  compare DET for sine, Lorenz x, and white noise at matched
  recurrence rate; this comparison is not a general determinism test.
- `rqa_laminarity_logistic_map.py`: Marwan, Romano, Thiel & Kurths (2007) --
  laminarity, trapping time and recurrence time entropy across the logistic
  map's r range: LAM is zero on periodic orbits and peaks after band merging.
- `windowed_rqa_transition.py`: Marwan, Romano, Thiel & Kurths (2007) --
  windowed DET and DIV on a logistic map whose parameter jumps from periodic
  (r = 3.5) to chaotic (r = 3.9).
- `correlation_dimension_lorenz.py`: Grassberger & Procaccia (1983) --
  the Lorenz attractor's fractal correlation dimension, estimated from
  a trajectory alone, with the scaling region fixed by hand and detected
  automatically (`torchnlts.scaling`).
- `entropy_measures.py`: Bandt & Pompe (2002) + Richman & Moorman
  (2000) -- compare permutation and sample entropy for sine, Lorenz x, and
  white noise at fixed sampling and embedding settings.
- `mutual_information_gaussians.py`: Kraskov, Stogbauer & Grassberger
  (2004) -- their KSG1 and KSG2 estimators checked against correlated Gaussians'
  analytically known mutual information.
- `lyapunov_from_data.py`: Rosenstein, Collins & De Luca (1993) + Kantz
  (1994) -- largest Lyapunov exponent from an observed time series,
  reproducing the logistic map's known ln(2).
- `lyapunov_spectrum_henon.py`: Benettin, Galgani, Giorgilli & Strelcyn
  (1980) + Geist, Parlitz & Lauterborn (1990) -- the full Lyapunov
  spectrum of a differentiable map, via `torch.func`, and the
  Kaplan-Yorke dimension (Kaplan & Yorke, 1979) computed from it.
- `eckmann_spectrum_henon.py`: Eckmann, Kamphorst, Ruelle & Ciliberto
  (1986) -- the Henon map's Lyapunov spectrum from its x-series alone,
  checked against the known exponents and `lyapunov_spectrum`, with the
  spurious exponents that a larger matrix dimension adds.
- `surrogate_data_testing.py`: Theiler, Eubank, Longtin, Galdrikian &
  Farmer (1992) + Schreiber & Schmitz (1996) -- the surrogate data
  hypothesis-testing methodology, using IAAFT surrogates and a
  lower-tail rank test (`surrogate_test`) against a statically transformed
  Gaussian-linear null.
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
  `torchnlts.sciml.SinkhornLoss` (their entropic-OT loss,
  new in this tutorial) against `AttractorLoss` (MMD, already in the
  library) on Lyapunov-exponent and correlation-dimension recovery.
- `multiscale_entropy_noise.py`: Costa, Goldberger & Peng (2002) --
  single-scale sample entropy ranks white noise as more "complex" than
  1/f noise, and multiscale entropy (`torchnlts.entropy.multiscale`)
  reverses that ranking by the largest coarse-graining scale swept.
- `composite_multiscale_entropy_noise.py`: Wu, Wu, Lin, Wang & Lee (2013) --
  white vs 1/f noise over many realizations; composite multiscale
  entropy has a smaller spread across realizations than plain multiscale
  entropy at large scales.
- `weighted_permutation_entropy_spikes.py`: Fadlallah et al. (2013) +
  Azami & Escudero (2016) -- ordinary permutation entropy barely changes
  when sparse large-amplitude bursts are added to white noise, while the
  variance-weighted and amplitude-aware variants
  (`permutation_entropy(..., weighting=...)`) drop clearly.
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
- `fuzzy_vs_sample_entropy.py`: Chen, Wang, Xie & Yu (2007) -- sample
  entropy's hard threshold diverges to infinity at small
  radius/short trajectories (`torchnlts.entropy.sample_entropy`,
  zero matching long templates), while fuzzy entropy
  (`torchnlts.entropy.fuzzy`), whose continuous membership function
  degrades far more gracefully, stays finite and smoothly increasing
  across the same sweep.
- `generalized_dimension_cascade_and_lorenz.py`: Russell, Hanson & Ott
  (1980) -- `torchnlts.dimension.generalized`'s box-counting D_q
  spectrum is non-increasing in order on a strongly multifractal
  p-weighted binary cascade (Rényi's own theorem), with D0/D1/D2
  falling out as special cases of one method. Also compares
  box-counting D2 against `torchnlts.dimension.correlation`'s existing
  pairwise-distance D2 on the Lorenz attractor, illustrating (not
  tightly matching) box-counting's well-known need for far more data
  to resolve a higher-dimensional attractor -- the historical reason
  Grassberger & Procaccia (1983) introduced the correlation sum.
- `higuchi_dimension_signals.py`: Higuchi (1988) -- `torchnlts.dimension.higuchi`'s
  no-embedding-needed fractal dimension D_H orders a sine wave, Lorenz
  x, and white noise from close to 1 (smooth) to close to 2 (maximally
  rough), the same three signals `entropy_measures.py` compares with
  embedding-based measures instead.
- `katz_weierstrass_esteller.py`: Katz (1988), following Esteller et al.
  (2001) -- `torchnlts.dimension.katz` on Weierstrass cosine functions of
  known dimension: Katz increases with the true dimension but is biased
  low and compressed toward 1, while Higuchi tracks it (qualitative match).
- `petrosian_dimension_signals.py`: Petrosian (1995) --
  `torchnlts.dimension.petrosian`'s sign-change-based dimension on a sine
  wave, Lorenz x, and white noise, and why its ordering depends on sampling.
- `variogram_dimension_fbm.py`: Gneiting, Ševčíková & Percival (2012) --
  `torchnlts.dimension.variogram`'s madogram and variogram estimates on
  fractional Brownian motion of known dimension, reproducing their Fig. 15
  ranking (variogram has the lower error on clean Gaussian data; the
  madogram is their default for robustness to outliers).
- `dfa_noise_scaling.py`: Peng et al. (1994) / Kantelhardt et al. (2001) --
  `torchnlts.fluctuations.dfa`'s exponent on white, 1/f, and Brownian noise,
  near the asymptotic 0.5 / 1.0 / 1.5.
- `zero_one_test_logistic_map.py`: Gottwald & Melbourne (2009) --
  `torchnlts.chaos.zero_one`'s 0-1 test on the logistic map: K near 0 in the
  periodic windows and near 1 for chaos, agreeing with the sign of the
  Lyapunov exponent (Fig. 4), and K growing slowly with N just above the
  edge of chaos (K = 0.029 at N = 2000, as they report 0.027).
