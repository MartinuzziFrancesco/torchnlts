Primitives
==========

This page lists every public torchnlts primitive with its gradient
class (one of *exact*, *smooth*, *piecewise*,
*implicit*, or *none*) and the reference(s) it's validated against.
Updated in the same commit that adds or changes a primitive.

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

   * - :doc:`permutation_entropy <generated/torchnlts.entropy.permutation.permutation_entropy>`
     - none
     - `Bandt & Pompe (2002) <https://doi.org/10.1103/PhysRevLett.88.174102>`__, Sec. II-III
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``OrdinalPatterns`` + ``Shannon``)

   * - :doc:`mutual_information <generated/torchnlts.entropy.mutual_information.mutual_information>`
     - none
     - `Kraskov, Stogbauer & Grassberger (2004) <https://doi.org/10.1103/PhysRevE.69.066138>`__, eq. 6-8 (KSG1)
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``KSG1``; confirmed inclusive-vs-strict marginal-counting deviation from the paper, locked in as a regression rather than reconciled -- see module docstring)

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
