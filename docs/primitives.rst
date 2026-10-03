Primitives
==========

This page lists every public torchnlts primitive with its gradient
class (*exact*, *smooth*, *piecewise*,
*implicit*, or *none*) and the reference(s) it's validated against.
Updated in the same commit that adds or changes a primitive.

Numerical range limits. Outside these, results or gradients can be
``inf``, ``NaN`` or inaccurate even when the exact answer is representable:

- Magnitudes within a few orders of the dtype's largest finite value
  (about ``1e38`` for float32, ``1e308`` for float64): intermediate sums,
  squares and differences can overflow.
- Magnitude ratios beyond about ``1e30`` within one series: the small
  values are lost relative to the large ones.
- Distance-based primitives (everything built on
  :func:`~torchnlts.neighbors.distance.pairwise_distance`: neighbor
  searches, recurrence, correlation sums, entropies, Lyapunov estimators,
  ...) raise coordinate differences to the power ``norm_order``. In
  float32 with ``norm_order=2``, distances more than about ``1e19`` times
  smaller than the trajectory's largest magnitude underflow to zero; use
  float64 for such data. Larger ``norm_order`` narrows this range to about
  ``2**(-126 / norm_order)`` in float32 (``2**(-1022 / norm_order)`` in
  float64); ``norm_order=float('inf')`` (max-norm) and ``norm_order=1``
  are not affected.
- Absolute accuracy is about the working dtype's epsilon times the
  series' largest magnitude. A result far smaller than that and resolved
  only by cancellation (e.g. a DFA residual of ``0.5`` next to a ``1e20``
  spike in float64) is not accurate. Katz, DFA, the variogram and the 0-1 test always
  compute in float64.

Non-finite input. Higuchi, Katz, Petrosian, DFA, the variogram, the 0-1 test and
Rosenstein return ``NaN`` for any batch row containing ``NaN`` or ``inf``
instead of raising, so they work under ``torch.vmap``.
:func:`~torchnlts.neighbors.distance.pairwise_distance` returns ``NaN`` for
every pair involving a point with a ``NaN``/``inf`` coordinate. Other rows
(or pairs) and their gradients are unaffected, and a non-finite row
receives a zero gradient. Exceptions: Petrosian has no gradient at all, and
a result that is ``NaN`` for another reason (e.g. a series too short for
the requested sweep) is not connected to the input's graph.

- Clicking the **primitive name** links to its API documentation.
- Clicking the **reference** links to the defining paper.
- The **Julia ground truth** column links the DynamicalSystems.jl
  sibling package function it's cross-validated against, where one
  exists.

.. list-table::
   :header-rows: 1
   :widths: 25 15 30 30

   * - Primitive
     - Gradient class
     - Reference
     - Julia ground truth

   * - :doc:`pairwise_distance <generated/torchnlts.neighbors.distance.pairwise_distance>`
     - exact
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__
     - `Distances.jl <https://github.com/JuliaStats/Distances.jl>`__

   * - :doc:`theiler_window_mask <generated/torchnlts.neighbors.theiler.theiler_window_mask>` / :doc:`apply_theiler_window <generated/torchnlts.neighbors.theiler.apply_theiler_window>`
     - exact
     - `Theiler (1986) <https://doi.org/10.1103/PhysRevA.34.2427>`__
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``Theiler``)

   * - :doc:`knn_search <generated/torchnlts.neighbors.knn.knn_search>`
     - piecewise
     - Kantz & Schreiber (2004), *Nonlinear Time Series Analysis*, Ch. 3
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``bulksearch`` + ``NeighborNumber``)

   * - :doc:`radius_neighbors <generated/torchnlts.neighbors.radius.radius_neighbors>`
     - none
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``bulksearch`` + ``WithinRange``)

   * - :doc:`trajectory_kernel <generated/torchnlts.neighbors.kernels.trajectory_kernel>`
     - smooth
     - `Judd (1992) <https://doi.org/10.1016/0167-2789(92)90028-B>`__ (gaussian); `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__ (exponential)
     - none (see module docstring)

   * - :doc:`delay_embedding <generated/torchnlts.embedding.delay.delay_embedding>`
     - exact
     - Takens (1981), *Detecting Strange Attractors in Turbulence*
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``embed``)

   * - :doc:`fractional_delay_embedding <generated/torchnlts.embedding.fractional_delay.fractional_delay_embedding>`
     - piecewise
     - `Laakso et al. (1996) <https://doi.org/10.1109/79.482137>`__ (linear-interpolation fractional delay)
     - none (see module docstring; consistency-checked against ``delay_embedding``)

   * - :doc:`hard_fnn <generated/torchnlts.embedding.fnn.hard_fnn>`
     - none
     - `Kennel, Brown & Abarbanel (1992) <https://doi.org/10.1103/PhysRevA.45.3403>`__
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``delay_fnn``)

   * - :doc:`cao_e1 <generated/torchnlts.embedding.cao.cao_e1>`
     - none
     - `Cao (1997) <https://doi.org/10.1016/S0167-2789(97)00118-8>`__, eq. 1-3
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``delay_afnn``)

   * - :doc:`cao_e2 <generated/torchnlts.embedding.cao.cao_e2>`
     - none
     - `Cao (1997) <https://doi.org/10.1016/S0167-2789(97)00118-8>`__, eq. 5
     - none (see module docstring: ``stochastic_indicator`` has indexing quirks that don't match eq. 5 cleanly)

   * - :doc:`hard_recurrence_matrix <generated/torchnlts.recurrence.matrix.hard_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 1
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``RecurrenceMatrix``)

   * - :doc:`soft_recurrence_matrix <generated/torchnlts.recurrence.matrix.soft_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 1 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency-checked against ``hard_recurrence_matrix``'s epsilon=0 limit)

   * - :doc:`hard_cross_recurrence_matrix <generated/torchnlts.recurrence.cross_recurrence.hard_cross_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 4.1
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``CrossRecurrenceMatrix``)

   * - :doc:`soft_cross_recurrence_matrix <generated/torchnlts.recurrence.cross_recurrence.soft_cross_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 4.1 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency-checked against ``hard_cross_recurrence_matrix``'s epsilon=0 limit)

   * - :doc:`hard_joint_recurrence_matrix <generated/torchnlts.recurrence.joint_recurrence.hard_joint_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 4.2
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``JointRecurrenceMatrix``'s definition, via element-wise AND of two ``RecurrenceMatrix``; see module docstring)

   * - :doc:`soft_joint_recurrence_matrix <generated/torchnlts.recurrence.joint_recurrence.soft_joint_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 4.2 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency-checked against ``hard_joint_recurrence_matrix``'s epsilon=0 limit)

   * - :doc:`recurrence_rate <generated/torchnlts.recurrence.rqa.recurrence_rate>`
     - exact
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 11
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``recurrencerate``)

   * - :doc:`determinism <generated/torchnlts.recurrence.rqa.determinism>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 13
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``determinism``)

   * - :doc:`dl_average <generated/torchnlts.recurrence.rqa.dl_average>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 12
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_average``)

   * - :doc:`dl_max <generated/torchnlts.recurrence.rqa.dl_max>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 2.1 (no separate numbered equation: the max of the same distribution ``dl_average``/eq. 12 draws on)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_max``)

   * - :doc:`dl_entropy <generated/torchnlts.recurrence.rqa.dl_entropy>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 14
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_entropy``)

   * - :doc:`correlation_sum <generated/torchnlts.dimension.correlation.correlation_sum>`
     - none
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``correlationsum``)

   * - :doc:`soft_correlation_sum <generated/torchnlts.dimension.correlation.soft_correlation_sum>`
     - smooth
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency-checked against ``correlation_sum``'s radius=0 limit)

   * - :doc:`correlation_dimension <generated/torchnlts.dimension.correlation.correlation_dimension>`
     - exact
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1 and surrounding discussion
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``slopefit`` with ``LinearRegression``)

   * - :doc:`box_occupation_probabilities <generated/torchnlts.dimension.generalized.box_occupation_probabilities>`
     - none
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``probabilities(ValueHistogram(ε), X)``; exact match confirmed, including box occupancy on a multivariate point set -- an earlier version of the cross-check test itself scrambled Julia's column-major ``reshape`` against torch's row-major flatten, giving a spurious-looking mismatch traced to every single point's box index before concluding it wasn't a real deviation)

   * - :doc:`box_counting_entropy <generated/torchnlts.dimension.generalized.box_counting_entropy>`
     - none
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``entropy(Renyi(;q,base=ℯ), ValueHistogram(ε), X)``; exact match across ``q`` in ``{0, 1, 2, 3.5}``)

   * - :doc:`generalized_dimension <generated/torchnlts.dimension.generalized.generalized_dimension>`
     - exact
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - none directly (this codebase doesn't replicate FractalDimensions.jl's ``estimate_boxsizes``/``slopefit`` automatic scaling-region detection, same "caller picks the sizes" convention as ``correlation_dimension``); validated instead against a p-weighted binary cascade's known strongly non-increasing-in-``q`` D_q spectrum (Rényi's own theorem) and a uniform interval's known D_q=1 for every ``q``

   * - :doc:`higuchi_length <generated/torchnlts.dimension.higuchi.higuchi_length>`
     - piecewise
     - `Higuchi (1988) <https://doi.org/10.1016/0167-2789(88)90081-4>`__
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``higuchi_length``; exact match confirmed, including its ``k**2`` (not ``k``) normalization, verified against Higuchi's own equation via a secondary source, not just carried over from Julia uncritically)

   * - :doc:`higuchi_dimension <generated/torchnlts.dimension.higuchi.higuchi_dimension>`
     - exact
     - `Higuchi (1988) <https://doi.org/10.1016/0167-2789(88)90081-4>`__
     - none directly (this codebase doesn't replicate FractalDimensions.jl's ``higuchi_dim``'s own automatic linear-region selection, same "caller picks the sweep" convention as ``correlation_dimension``/``generalized_dimension`` -- confirmed both functions can genuinely disagree, e.g. a sine wave over strides 1-20 gives 1.274 here against Julia's own region-restricted 1.052); ``higuchi_length`` itself is validated against Julia's own ``higuchi_length`` above, and this module's OLS-slope formula is validated directly on hand-derived power-law examples

   * - :doc:`katz_dimension <generated/torchnlts.dimension.katz.katz_dimension>`
     - piecewise
     - `Katz (1988) <https://doi.org/10.1016/0010-4825(88)90041-8>`__
     - `antropy <https://github.com/raphaelvallat/antropy>`__ (``katz_fd``; exact match with ``amplitude_only=True``, antropy's amplitude-differences-only variant; the default measures distances in the (index, amplitude) plane); no Julia equivalent

   * - :doc:`petrosian_dimension <generated/torchnlts.dimension.petrosian.petrosian_dimension>`
     - none
     - `Petrosian (1995) <https://doi.org/10.1109/CBMS.1995.465426>`__
     - `antropy <https://github.com/raphaelvallat/antropy>`__ (``petrosian_fd``; exact match, including its zero/negative-zero sign convention); no Julia equivalent

   * - :doc:`power_variogram <generated/torchnlts.dimension.variogram.power_variogram>`
     - exact (``power=2``), piecewise (``power<=1``)
     - `Gneiting, Ševčíková & Percival (2012) <https://doi.org/10.1214/11-STS370>`__, eq. 16
     - `fractaldim <https://github.com/cran/fractaldim>`__ (``fd.Variation``; normalization checked by reading its R source, R not available to run); hand-derived example; no Python/Julia equivalent

   * - :doc:`variogram_dimension <generated/torchnlts.dimension.variogram.variogram_dimension>`
     - exact
     - `Gneiting, Ševčíková & Percival (2012) <https://doi.org/10.1214/11-STS370>`__, eqs. 17-18
     - `fractaldim <https://github.com/cran/fractaldim>`__ (``fd.estim.variogram``/``fd.estim.madogram``, source read); validated on a straight line (D=1), Brownian motion (D near 1.5) and exact Davies-Harte fBm (D near 2 - H for H in 0.2-0.8)

   * - :doc:`dfa_fluctuation <generated/torchnlts.fluctuations.dfa.dfa_fluctuation>`
     - exact
     - `Peng et al. (1994) <https://doi.org/10.1103/PhysRevE.49.1685>`__; order-n detrending from `Kantelhardt et al. (2001) <https://doi.org/10.1016/S0378-4371(01)00144-3>`__
     - FluctuationAnalysis.jl (``dfa`` with ``bidirectional=false``; fluctuation values match to 1e-12 at orders 1-2); hand-derived example

   * - :doc:`dfa_exponent <generated/torchnlts.fluctuations.dfa.dfa_exponent>`
     - exact
     - `Peng et al. (1994) <https://doi.org/10.1103/PhysRevE.49.1685>`__
     - FluctuationAnalysis.jl (``scaling_exponent``, to 1e-12); `nolds <https://github.com/CSchoel/nolds>`__ (``dfa`` with ``overlap=False, fit_trend='poly', fit_exp='poly'``; exponent to 1e-9 at orders 0-3) and `antropy <https://github.com/raphaelvallat/antropy>`__ (``detrended_fluctuation``, order 1; exponent to 1e-6); also checked on white noise (alpha near 0.5) and Brownian motion (alpha near 1.5)

   * - :doc:`sample_entropy <generated/torchnlts.entropy.sample.sample_entropy>`
     - none
     - `Richman & Moorman (2000) <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__, eq. 1-3
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``SampleEntropy``; only loosely comparable -- its B(r) has a confirmed boundary-handling deviation from the paper, neutralized rather than replicated in one test and locked in as a regression in another; it also returns ``NaN`` rather than ``inf`` when A(r)=0)

   * - :doc:`soft_sample_entropy <generated/torchnlts.entropy.sample.soft_sample_entropy>`
     - smooth (except at exact Chebyshev-distance ties, see docstring)
     - `Richman & Moorman (2000) <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__, eq. 1-3 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency built on ``soft_correlation_sum``, already validated against its own radius=0 hard limit)

   * - :doc:`approximate_entropy <generated/torchnlts.entropy.approximate.approximate_entropy>`
     - none
     - `Pincus (1991) <https://doi.org/10.1073/pnas.88.6.2297>`__, eq. 1-4
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``ApproximateEntropy``; matches at ``embedding_dim >= 2`` only -- confirmed to special-case ``m=1`` to a different, non-Pincus definition, locked in as a regression)

   * - :doc:`soft_approximate_entropy <generated/torchnlts.entropy.approximate.soft_approximate_entropy>`
     - smooth (except at exact Chebyshev-distance ties, see docstring)
     - `Pincus (1991) <https://doi.org/10.1073/pnas.88.6.2297>`__, eq. 1-4 (kernel-relaxed via ``trajectory_kernel``)
     - none (see module docstring; consistency-checked against ``approximate_entropy``'s radius=0 hard limit)

   * - :doc:`ordinal_pattern_probabilities <generated/torchnlts.entropy.permutation.ordinal_pattern_probabilities>`
     - none
     - `Bandt & Pompe (2002) <https://doi.org/10.1103/PhysRevLett.88.174102>`__, Sec. II-III
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``OrdinalPatterns``; validated via ``permutation_entropy``'s own tests plus a direct composition check)

   * - :doc:`permutation_entropy <generated/torchnlts.entropy.permutation.permutation_entropy>`
     - none
     - `Bandt & Pompe (2002) <https://doi.org/10.1103/PhysRevLett.88.174102>`__, Sec. II-III
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``OrdinalPatterns`` + ``Shannon``)

   * - :doc:`mutual_information <generated/torchnlts.entropy.mutual_information.mutual_information>`
     - none
     - `Kraskov, Stogbauer & Grassberger (2004) <https://doi.org/10.1103/PhysRevE.69.066138>`__, eq. 6-8 (KSG1)
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``KSG1``; confirmed inclusive-vs-strict marginal-counting deviation from the paper, locked in as a regression rather than reconciled -- see module docstring)

   * - :doc:`coarse_grain <generated/torchnlts.entropy.multiscale.coarse_grain>`
     - exact
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2 (their eq. 1 is Shannon entropy, not this)
     - none (validated directly via hand-derived examples; its only consumer, ``multiscale_entropy``, is cross-checked below)

   * - :doc:`multiscale_entropy <generated/torchnlts.entropy.multiscale.multiscale_entropy>`
     - none (inherits ``sample_entropy``'s)
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``multiscale(RegularDownsampling, SampleEntropy)``; inherits ``sample_entropy``'s own confirmed boundary-handling deviation, neutralized the same way at every scale)

   * - :doc:`soft_multiscale_entropy <generated/torchnlts.entropy.multiscale.soft_multiscale_entropy>`
     - smooth (inherits ``soft_sample_entropy``'s, including its Chebyshev-tie caveat)
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2 (kernel-relaxed via ``soft_sample_entropy``)
     - none (no direct test of this function's own hard-limit behavior -- inherits ``soft_sample_entropy``'s own correctness story instead, which traces to ``soft_correlation_sum``'s radius=0 hard-limit validation, see that row above; gradcheck-verified for both temperature and trajectory)

   * - :doc:`lempel_ziv_complexity <generated/torchnlts.entropy.lempel_ziv.lempel_ziv_complexity>`
     - none (sequential incremental-parsing algorithm, no soft variant -- see module docstring)
     - `Lempel & Ziv (1976) <https://doi.org/10.1109/TIT.1976.1055501>`__ (complexity c(n)); `Kaspar & Schuster (1987) <https://doi.org/10.1103/PhysRevA.36.842>`__ (the algorithm actually implemented -- not linear time despite the reputation, see module docstring)
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``LempelZiv76``; a direct port of the same algorithm, exact match confirmed -- no known deviation, unlike this subpackage's other Julia comparisons)

   * - :doc:`dispersion_entropy <generated/torchnlts.entropy.dispersion.dispersion_entropy>`
     - none (Gaussian-CDF classification + histogram-based Shannon entropy, no soft variant -- same reasoning as ``permutation_entropy``); fully ``torch.vmap``-compatible, confirmed directly
     - `Rostaghi & Azami (2016) <https://doi.org/10.1109/LSP.2016.2542880>`__, eqs. as restated in `Azami, Rostaghi, Abasolo & Escudero (2017) <https://doi.org/10.1109/TBME.2017.2679136>`__, eqs. 2-4
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Dispersion`` + ``Shannon``; exact float64-precision match confirmed on ordinary input, including at an exact tie in the class-rounding step -- but a confirmed deviation for an exactly-constant trajectory, where Julia returns ``-0.0`` rather than this function's ``NaN``, locked in as a regression test)

   * - :doc:`renyi_entropy <generated/torchnlts.entropy.generalized.renyi_entropy>`
     - smooth, except at ``alpha=0`` (no gradient path) and, for every ``alpha``, only on a fixed support (see module docstring for both)
     - `Rényi (1961) <https://projecteuclid.org/euclid.bsmsp/1200512181>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Renyi``; exact match across the full parameter range tested, ``alpha`` in ``{0, 0.5, 1, 2, 5, inf}``, including near its own ``alpha ≈ 1`` tolerance window, not only exactly at it -- see module docstring for the precise tolerance conventions, which differ between ``alpha ≈ 0`` and ``alpha ≈ 1``)

   * - :doc:`tsallis_entropy <generated/torchnlts.entropy.generalized.tsallis_entropy>`
     - smooth, only on a fixed support (see module docstring)
     - `Tsallis (1988) <https://doi.org/10.1007/BF01016429>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Tsallis``; exact match including negative ``q`` -- but a confirmed deviation across Julia's own ``q ≈ 1`` tolerance window (not only exactly at ``q=1``) with ``k!=1``, where Julia's own special case silently drops ``k`` throughout that window unlike its own general-formula branch just outside it, locked in as a regression test)

   * - :doc:`fuzzy_entropy <generated/torchnlts.entropy.fuzzy.fuzzy_entropy>`
     - smooth, except a systematic (not measure-zero) Chebyshev-distance tie for any flattened length-2 template, and (outside the always-zero univariate ``embedding_dim=1`` short-template case, handled exactly) an inherent singularity at exactly coincident templates for ``0 < fuzzy_power < 1`` -- see module docstring
     - `Chen, Wang, Xie & Yu (2007) <https://doi.org/10.1109/TNSRE.2007.897025>`__, eq. 2-10
     - `EntropyHub <https://github.com/MattWillFlood/EntropyHub>`__ (``FuzzEn``, ``Fx='default'``; exact match confirmed, including a hand re-derivation of the paper's own formula from first principles, not just against EntropyHub's source -- no Julia comparison available, ``ComplexityMeasures.jl`` has no fuzzy entropy estimator)

   * - :doc:`rosenstein_divergence <generated/torchnlts.lyapunov.rosenstein.rosenstein_divergence>`
     - piecewise
     - `Rosenstein, Collins & De Luca (1993) <https://doi.org/10.1016/0167-2789(93)90009-P>`__
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunov_from_data`` with ``ntype=NeighborNumber(1)``, ``distance=Euclidean()``; matches to float64 precision, including its reference-set-fixed-by-max(k) convention, confirmed by reading its source)

   * - :doc:`rosenstein_lyapunov <generated/torchnlts.lyapunov.rosenstein.rosenstein_lyapunov>`
     - exact
     - `Rosenstein, Collins & De Luca (1993) <https://doi.org/10.1016/0167-2789(93)90009-P>`__, eq. 12-13 and surrounding discussion
     - none (closed-form OLS slope; validated against the logistic map's analytically known ln(2) Lyapunov exponent instead, see module docstring)

   * - :doc:`kantz_divergence <generated/torchnlts.lyapunov.kantz.kantz_divergence>`
     - piecewise
     - `Kantz (1994) <https://doi.org/10.1016/0375-9601(94)90991-1>`__, eq. 1-2
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunov_from_data`` with ``ntype=WithinRange``; confirmed mean-of-logs-vs-log-of-mean deviation from the paper, locked in as a regression rather than reconciled -- see module docstring). Its Lyapunov-exponent slope reuses ``rosenstein_lyapunov`` directly.

   * - :doc:`step_jacobian <generated/torchnlts.lyapunov.jacobian.step_jacobian>`
     - exact
     - none (a direct ``torch.func.jacrev``/``vmap`` application, not a published algorithm of its own)
     - none

   * - :doc:`lyapunov_spectrum <generated/torchnlts.lyapunov.spectrum.lyapunov_spectrum>`
     - exact
     - `Benettin, Galgani, Giorgilli & Strelcyn (1980) <https://doi.org/10.1007/BF02128236>`__; Geist, Parlitz & Lauterborn (1990), *Prog. Theor. Phys.* 83(5), 875-893 (QR "H2" variant)
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunovspectrum``, run live via subprocess in the test suite, not hardcoded; matches ``PredefinedDynamicalSystems.henon()`` to float64 precision, including its ``Ttr`` transient convention; confirmed to share this method's non-generic non-descending-order edge case on an unsorted diagonal linear map rather than diverge from it -- see module docstring)

   * - :doc:`zero_one_growth_rate <generated/torchnlts.chaos.zero_one.zero_one_growth_rate>`
     - smooth
     - `Gottwald & Melbourne (2009) <https://doi.org/10.1137/080718851>`__ (correlation method on the modified mean square displacement); first version: `Gottwald & Melbourne (2004) <https://doi.org/10.1098/rspa.2003.1183>`__
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``testchaos01(x, c, N0)``, run live via subprocess in the test suite; exact match per frequency when ChaosTools is given ``[x[2:N]; x[1]]``, because it starts its sums at ``p(1) = 0`` and never uses the last sample while this module follows the paper's window convention -- see module docstring)

   * - :doc:`zero_one_test <generated/torchnlts.chaos.zero_one.zero_one_test>`
     - piecewise
     - `Gottwald & Melbourne (2009) <https://doi.org/10.1137/080718851>`__ (median of ``K_c`` over frequencies)
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``testchaos01(x, cs, N0)`` returns only ``median(K_c) > 0.5``, so the per-frequency ``K_c`` comparison above is the ground truth; the median here averages the two middle values for an even count, as Julia's ``median`` does)

   * - :doc:`eckmann_spectrum <generated/torchnlts.lyapunov.eckmann.eckmann_spectrum>`
     - piecewise
     - `Eckmann, Kamphorst, Ruelle & Ciliberto (1986) <https://doi.org/10.1103/PhysRevA.34.4971>`__
     - `nolds <https://github.com/CSchoel/nolds>`__ (``lyap_e``, run live in the test suite; matches to float64 precision for matrix step 1 on well-conditioned fits. For matrix step > 1 nolds composes the fitted maps at every consecutive point where the paper's Sec. 2c composes them at stride ``m``, so this follows the paper instead, checked against an independent NumPy implementation; degenerate/rank-deficient rows are ``NaN``, ties beyond ``num_neighbors`` are dropped -- see module docstring); no Julia equivalent

   * - :doc:`phase_randomized_surrogate <generated/torchnlts.surrogate.phase.phase_randomized_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``RandomFourier``, run live via subprocess; confirmed to share this algorithm's literal-recipe quirk of not preserving the Nyquist bin's magnitude for even-length input, rather than diverge from it -- see module docstring)

   * - :doc:`aaft_surrogate <generated/torchnlts.surrogate.aaft.aaft_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``AAFT``; confirmed live, not just by reading source, to skip the paper's Gaussian-rank-matching step and reduce to ``RandomFourier`` on the raw data plus rank-restoration -- locked in as a disagreement rather than reconciled, same playbook as ``kantz_divergence``'s Julia deviation -- see module docstring)

   * - :doc:`iaaft_surrogate <generated/torchnlts.surrogate.iaaft.iaaft_surrogate>`
     - none
     - `Schreiber & Schmitz (1996) <https://doi.org/10.1103/PhysRevLett.77.635>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``IAAFT``, run live via subprocess; confirmed to share this method's seed-dependent convergence-quality spread on the same input, rather than a fixed always-tiny error -- deliberate, documented differences in initial condition and iteration-count/early-stopping convention, see module docstring)

   * - :doc:`cross_map_estimate <generated/torchnlts.causality.ccm.cross_map_estimate>`
     - piecewise
     - `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``ConvergentCrossMapping``/``predict``, run live via subprocess in the ``juliaenv_causality`` project; matches to float64 precision (~1e-16 max per-point difference) once a confirmed off-by-one in its own ``w`` Theiler-window parameter is accounted for (``theiler_window = w + 1``, not ``w`` directly) -- see module docstring)

   * - :doc:`ccm_skill <generated/torchnlts.causality.ccm.ccm_skill>`
     - exact
     - `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__
     - none (a direct Pearson correlation, not a published algorithm of its own; validated via ``cross_map_estimate``'s own Julia cross-check, which compares the resulting correlation too)

   * - :doc:`conditional_mutual_information <generated/torchnlts.causality.transfer_entropy.conditional_mutual_information>`
     - none
     - `Frenzel & Pompe (2007) <https://doi.org/10.1103/PhysRevLett.99.204101>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``FPVP``; confirmed inclusive-vs-strict marginal-counting deviation from the paper, same underlying ``Neighborhood.jl`` primitive as ``mutual_information``'s own confirmed ``KSG1`` deviation -- locked in as a regression rather than reconciled, see module docstring)

   * - :doc:`transfer_entropy <generated/torchnlts.causality.transfer_entropy.transfer_entropy>`
     - none
     - `Schreiber (2000) <https://doi.org/10.1103/PhysRevLett.85.461>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``TEShannon`` decomposed via ``CMIDecomposition``/``FPVP``, run live via subprocess; confirmed to match once the same (inclusive) marginal-counting convention is used on both sides, isolating the source/target-past and target-future alignment as independently verified correct -- see module docstring)

   * - :doc:`LyapunovLoss <generated/torchnlts.sciml.losses.LyapunovLoss>`
     - exact
     - none (composes ``lyapunov_spectrum``, already dual-validated -- see module docstring)
     - none (validated by direct comparison against calling ``lyapunov_spectrum`` outside the loss, not independently re-validated against Julia)

   * - :doc:`CorrelationDimensionLoss <generated/torchnlts.sciml.losses.CorrelationDimensionLoss>`
     - smooth
     - none (composes ``soft_correlation_sum`` + ``correlation_dimension``, already dual-validated -- see module docstring)
     - none (validated by direct comparison against calling those functions outside the loss)

   * - :doc:`RecurrenceLoss <generated/torchnlts.sciml.losses.RecurrenceLoss>`
     - smooth
     - none (composes ``soft_recurrence_matrix`` + ``recurrence_rate``, already dual-validated -- see module docstring)
     - none (validated by direct comparison against calling those functions outside the loss)

   * - :doc:`AttractorLoss <generated/torchnlts.sciml.losses.AttractorLoss>`
     - smooth
     - `Gretton et al. (2012) <https://jmlr.org/papers/v13/gretton12a.html>`__ (Maximum Mean Discrepancy)
     - none (composes ``trajectory_kernel``, already dual-validated; the MMD combination itself is validated by direct comparison against an independent computation, not against a published MMD reference implementation -- see module docstring)
