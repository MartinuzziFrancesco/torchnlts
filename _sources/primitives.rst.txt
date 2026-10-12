Primitives
==========

This page lists every public torchnlts primitive with its gradient
class (*exact*, *smooth*, *piecewise*,
*implicit*, or *none*) and the reference(s) it's validated against.
Updated in the same commit that adds or changes a primitive.

Conventions. Trajectories are ``[T]``, ``[T, D]`` or ``[B, T, D]`` tensors, numpy arrays or pandas objects.
Integer input is computed in the default floating dtype; float16 and bfloat16 are computed in float32 and, unless a
docstring says otherwise, cast back; complex input raises ``ValueError``. Exceptions:
:func:`~torchnlts.embedding.delay_embedding`, :func:`~torchnlts.surrogate.random_shuffle_surrogate` and
:func:`~torchnlts.surrogate.random_shift_surrogate` only index their input, so they keep any dtype, integer and complex
included; :func:`~torchnlts.lyapunov.lyapunov_spectrum` needs a float32 or float64 state (QR has no half-precision
kernel) and :func:`~torchnlts.lyapunov.step_jacobian` a real floating one. Parameters with a default value are
keyword-only. Scalar integer parameters such as ``embedding_dim``, ``delay`` and ``theiler_window`` must be integers: a
float (even ``2.0``) or a string raises ``ValueError``, and so does ``None`` unless it is the parameter's default, where
it selects the default behavior (``max_lag`` of :func:`~torchnlts.chaos.zero_one_growth_rate`, ``num_exponents`` of
:func:`~torchnlts.lyapunov.lyapunov_spectrum`, ``library_size`` of :func:`~torchnlts.causality.cross_map_estimate`,
``exclusion_zone`` of :func:`~torchnlts.matrixprofile.matrix_profile`, ``num_neighbors`` of
:func:`~torchnlts.lyapunov.eckmann_spectrum`). The swept integer parameters (``k``
of :func:`~torchnlts.dimension.higuchi_length`, ``lag`` of :func:`~torchnlts.dimension.power_variogram`,
``window_size`` of :func:`~torchnlts.fluctuations.dfa_fluctuation`, ``k_values`` of the Rosenstein and Kantz
estimators) take a number or a 1-D sequence of integer-valued numbers: ``[4.0, 8.0]`` is accepted, ``[4.5]`` raises
``ValueError``; the ``scales`` of the multiscale entropies are integers. A result evaluated over a swept parameter
(radii, scales, window sizes, lags, ...) has the sweep on its first axis: ``[K]`` for one trajectory, ``[K, B]`` for a
batch (so
:func:`~torchnlts.embedding.autocorrelation` and :func:`~torchnlts.embedding.auto_mutual_information` give
``[max_delay + 1]`` or ``[max_delay + 1, B]``). :func:`~torchnlts.causality.cross_map_estimate` returns one estimate
per embedded point with the rank of its estimated series (``[N]``, ``[N, D]`` or ``[B, N, D]``), and
:func:`~torchnlts.causality.ccm_skill` reduces the time axis (``[]``, ``[D]`` or ``[B, D]``).

Numerical range limits. Outside these, results or gradients can be
``inf``, ``NaN`` or inaccurate even when the exact answer is representable:

- Magnitudes within a few orders of the dtype's largest finite value
  (about ``1e38`` for float32, ``1e308`` for float64): intermediate sums,
  squares and differences can overflow.
- Magnitude ratios beyond about ``1e30`` within one series: the small
  values are lost relative to the large ones.
- Distance-based primitives (everything built on
  :func:`~torchnlts.neighbors.pairwise_distance`: neighbor
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
  spike in float64) is not accurate. Katz, DFA, the variogram and the 0-1 test compute in float64, or in
  float32 on Apple MPS (which has no float64), where the limits above are those of float32.

Non-finite input. Higuchi, Katz, Petrosian, DFA, the variogram, the 0-1 test,
:func:`~torchnlts.scaling.scaling_slope`, Rosenstein, Kantz, Eckmann, the correlation sums (and so sample entropy,
multiscale and composite multiscale entropy), approximate, fuzzy,
permutation and dispersion entropy, mutual information (both estimators), conditional mutual information, transfer
entropy, :func:`~torchnlts.embedding.hard_fnn`, :func:`~torchnlts.embedding.cao_e1`,
:func:`~torchnlts.embedding.cao_e2`, :func:`~torchnlts.embedding.autocorrelation` and
:func:`~torchnlts.embedding.auto_mutual_information` return ``NaN`` for any batch row
containing ``NaN`` or ``inf`` instead of raising, so they work under
``torch.vmap``. :func:`~torchnlts.embedding.estimate_delay` returns ``-1`` for such a row. Lempel-Ziv complexity
marks such a row too, with ``-1`` for the integer count (``NaN`` with ``normalize=True``), but it runs a Python loop
per row and does not work under ``torch.vmap``. The entropy functions check the whole row, including samples that
enter no embedded window (a large ``delay``) or that ``coarse_grain`` drops; ``mutual_information`` ignores samples
beyond the common length of its two inputs.
:func:`~torchnlts.causality.cross_map_estimate` marks single estimates instead: an estimate is ``NaN`` when its
embedded point has a non-finite coordinate, when fewer than ``embedding_dim + 1`` neighbors are left by the Theiler
window and ``library_size``, or when a neighbor's target is non-finite.
:func:`~torchnlts.causality.ccm_skill` is ``NaN`` for each ``[B, D]`` entry whose estimate or actual is constant or
non-finite. :func:`~torchnlts.lyapunov.lyapunov_spectrum` and :func:`~torchnlts.lyapunov.step_jacobian` give ``NaN``
for a state, or a row of a batched state, containing ``NaN`` or ``inf``; ``step_fn`` is evaluated at the zero state in
its place. The ``torchnlts.sciml`` losses give a ``NaN`` loss for a
batch row whose predicted or target trajectory (``LyapunovLoss``: initial state) contains ``NaN`` or ``inf``, so the
``"mean"`` and ``"sum"`` reductions are ``NaN`` too.
:func:`~torchnlts.surrogate.surrogate_test` returns a
``NaN`` p-value for a row whose trajectory has a non-finite entry or whose
data or any surrogate statistic is non-finite (``NaN`` or ``inf``); the shuffle and shift surrogates move non-finite values like any
other.
:func:`~torchnlts.neighbors.pairwise_distance` returns ``NaN`` for
every pair involving a point with a ``NaN``/``inf`` coordinate. Other rows
(or pairs, or estimates) and their gradients are unaffected, and a non-finite row
receives a zero gradient. Exceptions: Petrosian has no gradient at all, and
a result that is ``NaN`` for another reason (e.g. a series too short for
the requested sweep) is not connected to the input's graph. The hard
recurrence matrices with ``recurrence_rate``, ``local_rate`` or ``scale``
compute their thresholds over the finite distances of each batch row (a row
with none gives an all-False matrix) and work under ``torch.vmap``.

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

   * - :doc:`pairwise_distance <generated/torchnlts.neighbors.pairwise_distance>`
     - exact
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__
     - `Distances.jl <https://github.com/JuliaStats/Distances.jl>`__

   * - :doc:`theiler_window_mask <generated/torchnlts.neighbors.theiler_window_mask>` / :doc:`apply_theiler_window <generated/torchnlts.neighbors.apply_theiler_window>`
     - exact
     - `Theiler (1986) <https://doi.org/10.1103/PhysRevA.34.2427>`__
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``Theiler``)

   * - :doc:`knn_search <generated/torchnlts.neighbors.knn_search>`
     - piecewise
     - Kantz & Schreiber (2004), *Nonlinear Time Series Analysis*, Ch. 3
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``bulksearch`` + ``NeighborNumber``)

   * - :doc:`radius_neighbors <generated/torchnlts.neighbors.radius_neighbors>`
     - none
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__
     - `Neighborhood.jl <https://github.com/JuliaDynamics/Neighborhood.jl>`__ (``bulksearch`` + ``WithinRange``)

   * - :doc:`trajectory_kernel <generated/torchnlts.neighbors.trajectory_kernel>`
     - smooth
     - `Judd (1992) <https://doi.org/10.1016/0167-2789(92)90028-B>`__ (gaussian); `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__ (exponential)
     - none (no Julia package implements these kernels; hand-derived values from Judd (1992) and Sugihara et al. (2012), and a gradcheck)

   * - :doc:`complexity_estimate <generated/torchnlts.neighbors.complexity_estimate>`
     - exact (zero subgradient at a constant series)
     - `Batista et al. (2014) <https://doi.org/10.1007/s10618-013-0312-3>`__
     - none in Julia; `tsfresh <https://github.com/blue-yonder/tsfresh>`__ (``cid_ce``, with and without ``normalize``)

   * - :doc:`complexity_invariant_distance <generated/torchnlts.neighbors.complexity_invariant_distance>`
     - piecewise (``max``/``min`` of the two complexity estimates)
     - `Batista et al. (2014) <https://doi.org/10.1007/s10618-013-0312-3>`__
     - none (hand-derived values and Euclidean distance times the ratio of tsfresh ``cid_ce`` values). Both estimates zero gives a correction factor of 1, one zero gives ``inf``; the paper leaves both cases undefined.

   * - :doc:`delay_embedding <generated/torchnlts.embedding.delay_embedding>`
     - exact
     - Takens (1981), *Detecting Strange Attractors in Turbulence*
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``embed``)

   * - :doc:`fractional_delay_embedding <generated/torchnlts.embedding.fractional_delay_embedding>`
     - piecewise
     - `Laakso et al. (1996) <https://doi.org/10.1109/79.482137>`__ (linear-interpolation fractional delay)
     - none (DelayEmbeddings.jl's ``embed`` takes integer lags only; at integer lags the result equals ``delay_embedding`` exactly, which is checked)

   * - :doc:`hard_fnn <generated/torchnlts.embedding.hard_fnn>`
     - none
     - `Kennel, Brown & Abarbanel (1992) <https://doi.org/10.1103/PhysRevA.45.3403>`__
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``delay_fnn``)

   * - :doc:`cao_e1 <generated/torchnlts.embedding.cao_e1>`
     - none
     - `Cao (1997) <https://doi.org/10.1016/S0167-2789(97)00118-8>`__, eq. 1-3
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``delay_afnn``)

   * - :doc:`cao_e2 <generated/torchnlts.embedding.cao_e2>`
     - none
     - `Cao (1997) <https://doi.org/10.1016/S0167-2789(97)00118-8>`__, eq. 4-5
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``stochastic_indicator``; agrees to about 0.5% at 500 points and 5% at 100, the gap shrinking as 1/N, because Julia divides each E* by the embedded length instead of the number of summed terms and leaves the last ``delay + 1`` points out of its neighbor search)

   * - :doc:`autocorrelation <generated/torchnlts.embedding.autocorrelation>`
     - exact
     - `Rosenstein, Collins & De Luca (1994) <https://doi.org/10.1016/0167-2789(94)90134-1>`__ (delay criteria)
     - `StatsBase.jl <https://github.com/JuliaStats/StatsBase.jl>`__ (``autocor``, ``demean=true``)

   * - :doc:`auto_mutual_information <generated/torchnlts.embedding.auto_mutual_information>`
     - none
     - `Fraser & Swinney (1986) <https://doi.org/10.1103/PhysRevA.33.1134>`__
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``selfmutualinfo``, ``nbins``; exact at delay 0 and for a series periodic over the pair window, otherwise within ~0.01 nats at 200 points and ~0.001 at 500, because the delayed marginal is counted from the delayed samples)

   * - :doc:`estimate_delay <generated/torchnlts.embedding.estimate_delay>`
     - none
     - `Fraser & Swinney (1986) <https://doi.org/10.1103/PhysRevA.33.1134>`__ (``mi_min``)
     - `DelayEmbeddings.jl <https://github.com/JuliaDynamics/DelayEmbeddings.jl>`__ (``estimate_delay``: ``mi_min``, ``ac_zero``, ``ac_min``; ``ac_1/e`` has no Julia counterpart)

   * - :doc:`hard_recurrence_matrix <generated/torchnlts.recurrence.hard_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 1
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``RecurrenceMatrix``; ``recurrence_rate=`` matches ``GlobalRecurrenceRate`` and ``scale=`` matches ``RecurrenceThresholdScaled`` with ``maximum``/``mean``/``std`` exactly; ``local_rate=`` gives each column about that fraction of neighbours (quantile-derived; ties can add recurrences, and rate 1 saturates where Julia rejects it), while Julia's ``LocalRecurrenceRate`` keeps only the upper triangle and mirrors it, so Julia's matrix is the mirrored upper triangle of this one)

   * - :doc:`soft_recurrence_matrix <generated/torchnlts.recurrence.soft_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 1 (kernel-relaxed via ``trajectory_kernel``)
     - none (consistency-checked against ``hard_recurrence_matrix``'s epsilon=0 limit as the temperature goes to 0)

   * - :doc:`hard_cross_recurrence_matrix <generated/torchnlts.recurrence.hard_cross_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.3
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``CrossRecurrenceMatrix``, including ``GlobalRecurrenceRate``, ``LocalRecurrenceRate`` and ``RecurrenceThresholdScaled``; exact)

   * - :doc:`soft_cross_recurrence_matrix <generated/torchnlts.recurrence.soft_cross_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.3 (kernel-relaxed via ``trajectory_kernel``)
     - none (consistency-checked against ``hard_cross_recurrence_matrix``'s epsilon=0 limit as the temperature goes to 0)

   * - :doc:`hard_joint_recurrence_matrix <generated/torchnlts.recurrence.hard_joint_recurrence_matrix>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.4
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``JointRecurrenceMatrix``'s definition, via element-wise AND of two ``RecurrenceMatrix`` because the two-argument ``JointRecurrenceMatrix(R1, R2)`` constructor errors in the tested version; rate-based and scaled thresholds are computed per trajectory after truncation to the common length, as in Julia)

   * - :doc:`soft_joint_recurrence_matrix <generated/torchnlts.recurrence.soft_joint_recurrence_matrix>`
     - smooth
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.4 (kernel-relaxed via ``trajectory_kernel``)
     - none (consistency-checked against ``hard_joint_recurrence_matrix``'s epsilon=0 limit as the temperature goes to 0)

   * - :doc:`recurrence_rate <generated/torchnlts.recurrence.recurrence_rate>`
     - exact
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 11
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``recurrencerate``)

   * - :doc:`determinism <generated/torchnlts.recurrence.determinism>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 13
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``determinism``)

   * - :doc:`dl_average <generated/torchnlts.recurrence.dl_average>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 12
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_average``)

   * - :doc:`dl_max <generated/torchnlts.recurrence.dl_max>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 2.1 (no separate numbered equation: the max of the same distribution ``dl_average``/eq. 12 draws on)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_max``)

   * - :doc:`dl_entropy <generated/torchnlts.recurrence.dl_entropy>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 14
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``dl_entropy``)

   * - :doc:`laminarity <generated/torchnlts.recurrence.laminarity>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 62 (LAM)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``rqa`` ``:LAM``; standalone ``laminarity`` ignores ``theiler`` in its numerator for ``theiler != 1``)

   * - :doc:`trapping_time <generated/torchnlts.recurrence.trapping_time>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 63 (TT)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``trappingtime``)

   * - :doc:`vl_max <generated/torchnlts.recurrence.vl_max>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, eq. 64 (Vmax)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``vl_max``)

   * - :doc:`vl_entropy <generated/torchnlts.recurrence.vl_entropy>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.5.3 (vertical lines; the Shannon entropy of their length histogram is the vertical analogue of ``dl_entropy``)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``vl_entropy``)

   * - :doc:`rt_entropy <generated/torchnlts.recurrence.rt_entropy>`
     - none
     - `Little et al. (2007) <https://doi.org/10.1186/1475-925X-6-23>`__ (RPDE; the concept -- Little uses first returns after leaving a neighbourhood, this uses RecurrenceAnalysis.jl's gaps plus floored half-line lengths)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``rt_entropy``)

   * - :doc:`ratio <generated/torchnlts.recurrence.ratio>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.5.2 (DET / RR)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (no ``ratio``; ``determinism`` / ``recurrencerate``)

   * - :doc:`divergence <generated/torchnlts.recurrence.divergence>`
     - none
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.5.2 (1 / Lmax)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``divergence``, ``rqa`` ``:DIV``; ``inf`` when Lmax = 0)

   * - :doc:`windowed_rqa <generated/torchnlts.recurrence.windowed_rqa>`
     - inherits ``measure``'s
     - `Marwan et al. (2007) <https://doi.org/10.1016/j.physrep.2006.11.001>`__, Sec. 3.5 (RQA in windows along the main diagonal)
     - `RecurrenceAnalysis.jl <https://github.com/JuliaDynamics/RecurrenceAnalysis.jl>`__ (``windowed``)

   * - :doc:`correlation_sum <generated/torchnlts.dimension.correlation_sum>`
     - none
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``correlationsum``)

   * - :doc:`soft_correlation_sum <generated/torchnlts.dimension.soft_correlation_sum>`
     - smooth
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1 (kernel-relaxed via ``trajectory_kernel``)
     - none (consistency-checked against ``correlation_sum``'s radius=0 limit as the temperature goes to 0)

   * - :doc:`correlation_dimension <generated/torchnlts.dimension.correlation_dimension>`
     - exact
     - `Grassberger & Procaccia (1983) <https://doi.org/10.1016/0167-2789(83)90298-1>`__, eq. 1 and surrounding discussion
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``slopefit`` with ``LinearRegression``)

   * - :doc:`box_occupation_probabilities <generated/torchnlts.dimension.box_occupation_probabilities>`
     - none
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``probabilities(ValueHistogram(ε), X)``; exact match, including box occupancy on a multivariate point set)

   * - :doc:`box_counting_entropy <generated/torchnlts.dimension.box_counting_entropy>`
     - none
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``entropy(Renyi(;q,base=ℯ), ValueHistogram(ε), X)``; exact match across ``q`` in ``{0, 1, 2, 3.5}``)

   * - :doc:`generalized_dimension <generated/torchnlts.dimension.generalized_dimension>`
     - exact
     - `Russell, Hanson & Ott (1980) <https://doi.org/10.1103/PhysRevLett.45.1175>`__
     - none directly (FractalDimensions.jl's ``generalized_dim`` picks the box sizes and the scaling region automatically, while here the caller supplies the sizes and every point is fitted, as in ``correlation_dimension``); validated instead against a p-weighted binary cascade's known strongly non-increasing-in-``q`` D_q spectrum (Rényi's own theorem) and a uniform interval's known D_q=1 for every ``q``

   * - :doc:`higuchi_length <generated/torchnlts.dimension.higuchi_length>`
     - piecewise
     - `Higuchi (1988) <https://doi.org/10.1016/0167-2789(88)90081-4>`__
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``higuchi_length``; exact match, including its ``k**2`` (not ``k``) normalization, which is Higuchi's own formula)

   * - :doc:`higuchi_dimension <generated/torchnlts.dimension.higuchi_dimension>`
     - exact
     - `Higuchi (1988) <https://doi.org/10.1016/0167-2789(88)90081-4>`__
     - none directly (FractalDimensions.jl's ``higuchi_dim`` selects a linear region of the sweep automatically, while here every given ``k`` is fitted, as in ``correlation_dimension``; the two can differ, e.g. a sine wave over strides 1-20 gives 1.274 here against Julia's 1.052); ``higuchi_length`` itself is validated against Julia's above, and the OLS-slope formula against hand-derived power-law examples

   * - :doc:`katz_dimension <generated/torchnlts.dimension.katz_dimension>`
     - piecewise
     - `Katz (1988) <https://doi.org/10.1016/0010-4825(88)90041-8>`__
     - `antropy <https://github.com/raphaelvallat/antropy>`__ (``katz_fd``; exact match with ``amplitude_only=True``, antropy's amplitude-differences-only variant; the default measures distances in the (index, amplitude) plane); no Julia equivalent

   * - :doc:`petrosian_dimension <generated/torchnlts.dimension.petrosian_dimension>`
     - none
     - `Petrosian (1995) <https://doi.org/10.1109/CBMS.1995.465426>`__
     - `antropy <https://github.com/raphaelvallat/antropy>`__ (``petrosian_fd``; exact match, including its zero/negative-zero sign convention); no Julia equivalent

   * - :doc:`power_variogram <generated/torchnlts.dimension.power_variogram>`
     - exact (``power=2``), piecewise (``power<=1``)
     - `Gneiting, Ševčíková & Percival (2012) <https://doi.org/10.1214/11-STS370>`__, eq. 16
     - `fractaldim <https://github.com/cran/fractaldim>`__ (``fd.Variation``; the normalization follows its R source, which was not run); hand-derived example; no Python/Julia equivalent

   * - :doc:`variogram_dimension <generated/torchnlts.dimension.variogram_dimension>`
     - exact
     - `Gneiting, Ševčíková & Percival (2012) <https://doi.org/10.1214/11-STS370>`__, eqs. 17-18
     - `fractaldim <https://github.com/cran/fractaldim>`__ (``fd.estim.variogram``/``fd.estim.madogram``, formulas from the R source, which was not run); validated on a straight line (D=1), Brownian motion (D near 1.5) and exact Davies-Harte fBm (D near 2 - H for H in 0.2-0.8)

   * - :doc:`dfa_fluctuation <generated/torchnlts.fluctuations.dfa_fluctuation>`
     - exact
     - `Peng et al. (1994) <https://doi.org/10.1103/PhysRevE.49.1685>`__; order-n detrending from `Kantelhardt et al. (2001) <https://doi.org/10.1016/S0378-4371(01)00144-3>`__
     - FluctuationAnalysis.jl (``dfa`` with ``bidirectional=false``; fluctuation values match to 1e-12 at orders 1-2); hand-derived example

   * - :doc:`dfa_exponent <generated/torchnlts.fluctuations.dfa_exponent>`
     - exact
     - `Peng et al. (1994) <https://doi.org/10.1103/PhysRevE.49.1685>`__
     - FluctuationAnalysis.jl (``scaling_exponent``, to 1e-12); `nolds <https://github.com/CSchoel/nolds>`__ (``dfa`` with ``overlap=False, fit_trend='poly', fit_exp='poly'``; exponent to 1e-9 at orders 0-3) and `antropy <https://github.com/raphaelvallat/antropy>`__ (``detrended_fluctuation``, order 1; exponent to 1e-6); also checked on white noise (alpha near 0.5) and Brownian motion (alpha near 1.5)

   * - :doc:`sample_entropy <generated/torchnlts.entropy.sample_entropy>`
     - none
     - `Richman & Moorman (2000) <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__, eq. 1-3
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``SampleEntropy``; its B(r) counts one extra neighbour taken from the untruncated embedding at ``delay=1``, so the values agree only when no such match exists, which the test arranges with far outliers at the end of the series; it also returns ``NaN`` rather than ``inf`` when A(r)=0)

   * - :doc:`soft_sample_entropy <generated/torchnlts.entropy.soft_sample_entropy>`
     - smooth (except at exact Chebyshev-distance ties, see docstring)
     - `Richman & Moorman (2000) <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__, eq. 1-3 (kernel-relaxed via ``trajectory_kernel``)
     - none (built on ``soft_correlation_sum``, which is validated against its radius=0 hard limit)

   * - :doc:`approximate_entropy <generated/torchnlts.entropy.approximate_entropy>`
     - none
     - `Pincus (1991) <https://doi.org/10.1073/pnas.88.6.2297>`__, eq. 1-4
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``ApproximateEntropy``; matches at ``embedding_dim >= 2``. At ``embedding_dim=1`` that package returns ``phi(1)`` alone, a non-Pincus definition, and a test pins the disagreement)

   * - :doc:`soft_approximate_entropy <generated/torchnlts.entropy.soft_approximate_entropy>`
     - smooth (except at exact Chebyshev-distance ties, see docstring)
     - `Pincus (1991) <https://doi.org/10.1073/pnas.88.6.2297>`__, eq. 1-4 (kernel-relaxed via ``trajectory_kernel``)
     - none (checked against ``approximate_entropy``'s radius=0 hard limit)

   * - :doc:`ordinal_pattern_probabilities <generated/torchnlts.entropy.ordinal_pattern_probabilities>`
     - none; piecewise smooth with ``weighting="variance"``/``"amplitude"`` (pattern assignment piecewise constant, weights smooth; jumps at ties, kinks at zero differences)
     - `Bandt & Pompe (2002) <https://doi.org/10.1103/PhysRevLett.88.174102>`__, Sec. II-III
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``OrdinalPatterns``; validated via ``permutation_entropy``'s own tests plus a direct composition check)

   * - :doc:`permutation_entropy <generated/torchnlts.entropy.permutation_entropy>`
     - none; piecewise smooth with ``weighting="variance"``/``"amplitude"`` (pattern assignment piecewise constant, weights smooth; jumps at ties, kinks at zero differences)
     - `Bandt & Pompe (2002) <https://doi.org/10.1103/PhysRevLett.88.174102>`__, Sec. II-III; `Fadlallah et al. (2013) <https://doi.org/10.1103/PhysRevE.87.022911>`__ (variance weights); `Azami & Escudero (2016) <https://doi.org/10.1016/j.cmpb.2016.02.008>`__ (amplitude-aware weights)
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``OrdinalPatterns``, ``WeightedOrdinalPatterns``, ``AmplitudeAwareOrdinalPatterns`` + ``Shannon``; weighted variants match to 1e-9)

   * - :doc:`mutual_information <generated/torchnlts.entropy.mutual_information>`
     - none
     - `Kraskov, Stogbauer & Grassberger (2004) <https://doi.org/10.1103/PhysRevE.69.066138>`__, eq. 6-8 (KSG1) and the second estimator I^(2) (``estimator="ksg2"``)
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``KSG1``: counts marginal neighbours inclusively where this function follows the paper's strict count; ``KSG2``: agrees at ``k=1`` for data at ordinary scales, differs for ``k>1`` (uses only the k-th neighbour's marginal distance and ``1`` in place of ``1/k``) and at any ``k`` for data near 1e-16 (adds ``10*eps()`` to each marginal radius). Tests pin these differences; see the module docstring)

   * - :doc:`coarse_grain <generated/torchnlts.entropy.coarse_grain>`
     - exact
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2 (their eq. 1 is Shannon entropy, not this)
     - none (hand-derived examples; ``multiscale_entropy`` and ``composite_multiscale_entropy``, which use it, are cross-checked below)

   * - :doc:`multiscale_entropy <generated/torchnlts.entropy.multiscale_entropy>`
     - none (inherits ``sample_entropy``'s)
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``multiscale(RegularDownsampling, SampleEntropy)``; inherits ``sample_entropy``'s boundary-handling deviation, neutralized the same way at every scale)

   * - :doc:`soft_multiscale_entropy <generated/torchnlts.entropy.soft_multiscale_entropy>`
     - smooth (inherits ``soft_sample_entropy``'s, including its Chebyshev-tie caveat)
     - `Costa, Goldberger & Peng (2002) <https://doi.org/10.1103/PhysRevLett.89.068102>`__, the unnumbered coarse-graining formula on p. 2 (kernel-relaxed via ``soft_sample_entropy``)
     - none (built on ``soft_sample_entropy`` and ``coarse_grain``; gradcheck on both temperature and trajectory)

   * - :doc:`composite_multiscale_entropy <generated/torchnlts.entropy.composite_multiscale_entropy>`
     - none (inherits ``sample_entropy``'s)
     - `Wu, Wu, Lin, Wang & Lee (2013) <https://doi.org/10.3390/e15031069>`__, sample entropy averaged over all coarse-graining offsets (``coarse_grain(..., offset=)``)
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``multiscale(CompositeDownsampling, SampleEntropy)``; matches to 2e-16. Julia truncates every offset series to the common length ``(N - s + 1) // s``, this function keeps each offset's natural length ``(N - offset) // s``; the test uses a length where both agree. Inherits ``sample_entropy``'s boundary-handling deviation, neutralized with far outliers at the end of the series)

   * - :doc:`soft_composite_multiscale_entropy <generated/torchnlts.entropy.soft_composite_multiscale_entropy>`
     - smooth (inherits ``soft_sample_entropy``'s, including its Chebyshev-tie caveat)
     - `Wu, Wu, Lin, Wang & Lee (2013) <https://doi.org/10.3390/e15031069>`__ (kernel-relaxed via ``soft_sample_entropy``)
     - none (composition of ``coarse_grain`` offsets and ``soft_sample_entropy``; gradcheck on both temperature and trajectory)

   * - :doc:`lempel_ziv_complexity <generated/torchnlts.entropy.lempel_ziv_complexity>`
     - none (sequential incremental-parsing algorithm, no soft variant)
     - `Lempel & Ziv (1976) <https://doi.org/10.1109/TIT.1976.1055501>`__ (complexity c(n)); `Kaspar & Schuster (1987) <https://doi.org/10.1103/PhysRevA.36.842>`__ (the algorithm implemented; its comparison-based parsing is closer to quadratic than linear in the length)
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``LempelZiv76``; the same algorithm, identical counts)

   * - :doc:`dispersion_entropy <generated/torchnlts.entropy.dispersion_entropy>`
     - none (Gaussian-CDF classification + histogram-based Shannon entropy, no soft variant -- same reasoning as ``permutation_entropy``); works under ``torch.vmap``
     - `Rostaghi & Azami (2016) <https://doi.org/10.1109/LSP.2016.2542880>`__, eqs. as restated in `Azami, Rostaghi, Abasolo & Escudero (2017) <https://doi.org/10.1109/TBME.2017.2679136>`__, eqs. 2-4
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Dispersion`` + ``Shannon``; exact float64-precision match on ordinary input, including an exact tie in the class-rounding step. For an exactly constant trajectory Julia returns ``-0.0`` where this function returns ``NaN``; a test pins the difference)

   * - :doc:`renyi_entropy <generated/torchnlts.entropy.renyi_entropy>`
     - smooth, except at ``alpha=0`` (no gradient path) and, for every ``alpha``, only on a fixed support (an exactly zero probability gets gradient 0 by convention)
     - `Rényi (1961) <https://projecteuclid.org/euclid.bsmsp/1200512181>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Renyi``; exact match for ``alpha`` in ``{0, 0.5, 1, 2, 5, inf}``, including near its ``alpha ≈ 1`` tolerance window, not only exactly at 1. The tolerance conventions at ``alpha ≈ 0`` and ``alpha ≈ 1`` differ; see the module docstring)

   * - :doc:`tsallis_entropy <generated/torchnlts.entropy.tsallis_entropy>`
     - smooth, only on a fixed support (an exactly zero probability gets gradient 0 by convention)
     - `Tsallis (1988) <https://doi.org/10.1007/BF01016429>`__
     - `ComplexityMeasures.jl <https://github.com/JuliaDynamics/ComplexityMeasures.jl>`__ (``Tsallis``; exact match including negative ``q``. Inside Julia's ``q ≈ 1`` tolerance window with ``k!=1``, Julia drops ``k``, unlike its general-formula branch just outside the window; this function returns ``k`` times Shannon entropy there. A test pins the difference)

   * - :doc:`fuzzy_entropy <generated/torchnlts.entropy.fuzzy_entropy>`
     - smooth, except a systematic (not measure-zero) Chebyshev-distance tie for any flattened length-2 template, and (outside the always-zero univariate ``embedding_dim=1`` short-template case, handled exactly) an inherent singularity at exactly coincident templates for ``0 < fuzzy_power < 1`` -- see docstring
     - `Chen, Wang, Xie & Yu (2007) <https://doi.org/10.1109/TNSRE.2007.897025>`__, eq. 2-10
     - `EntropyHub <https://github.com/MattWillFlood/EntropyHub>`__ (``FuzzEn``, ``Fx='default'``; exact match, and a test re-derives the paper's formula by hand. No Julia comparison exists: ``ComplexityMeasures.jl`` has no fuzzy entropy estimator)

   * - :doc:`rosenstein_divergence <generated/torchnlts.lyapunov.rosenstein_divergence>`
     - piecewise
     - `Rosenstein, Collins & De Luca (1993) <https://doi.org/10.1016/0167-2789(93)90009-P>`__
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunov_from_data`` with ``ntype=NeighborNumber(1)``, ``distance=Euclidean()``; matches to float64 precision, including its reference set fixed by the largest ``k``)

   * - :doc:`rosenstein_lyapunov <generated/torchnlts.lyapunov.rosenstein_lyapunov>`
     - exact
     - `Rosenstein, Collins & De Luca (1993) <https://doi.org/10.1016/0167-2789(93)90009-P>`__, eq. 12-13 and surrounding discussion
     - none (closed-form OLS slope; checked against the logistic map's exact exponent ln 2)

   * - :doc:`kantz_divergence <generated/torchnlts.lyapunov.kantz_divergence>`
     - piecewise
     - `Kantz (1994) <https://doi.org/10.1016/0375-9601(94)90991-1>`__, eq. 1-2
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunov_from_data`` with ``ntype=WithinRange``; it averages the log of each neighbor distance where the paper takes the log of the mean, so the two differ -- see module docstring). The slope is taken with ``rosenstein_lyapunov``.

   * - :doc:`step_jacobian <generated/torchnlts.lyapunov.step_jacobian>`
     - exact
     - none (a direct ``torch.func.jacrev``/``vmap`` application, not a published algorithm of its own)
     - none

   * - :doc:`lyapunov_spectrum <generated/torchnlts.lyapunov.lyapunov_spectrum>`
     - exact
     - `Benettin, Galgani, Giorgilli & Strelcyn (1980) <https://doi.org/10.1007/BF02128236>`__; Geist, Parlitz & Lauterborn (1990), *Prog. Theor. Phys.* 83(5), 875-893 (QR "H2" variant)
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``lyapunovspectrum`` on ``PredefinedDynamicalSystems.henon()``; matches to float64 precision, including its ``Ttr`` transient convention; shares the non-descending order for a linear map with an unsorted diagonal -- see module docstring)

   * - :doc:`zero_one_growth_rate <generated/torchnlts.chaos.zero_one_growth_rate>`
     - smooth
     - `Gottwald & Melbourne (2009) <https://doi.org/10.1137/080718851>`__ (correlation method on the modified mean square displacement); first version: `Gottwald & Melbourne (2004) <https://doi.org/10.1098/rspa.2003.1183>`__
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``testchaos01(x, c, N0)``, run via subprocess; exact match per frequency when ChaosTools is given ``[x[2:N]; x[1]]``, because it starts its sums at ``p(1) = 0`` and never uses the last sample while this module follows the paper's window convention -- see module docstring)

   * - :doc:`zero_one_test <generated/torchnlts.chaos.zero_one_test>`
     - piecewise
     - `Gottwald & Melbourne (2009) <https://doi.org/10.1137/080718851>`__ (median of ``K_c`` over frequencies)
     - `ChaosTools.jl <https://github.com/JuliaDynamics/ChaosTools.jl>`__ (``testchaos01(x, cs, N0)`` returns only ``median(K_c) > 0.5``, so the per-frequency ``K_c`` comparison above is the ground truth; the median here averages the two middle values for an even count, as Julia's ``median`` does)

   * - :doc:`eckmann_spectrum <generated/torchnlts.lyapunov.eckmann_spectrum>`
     - piecewise
     - `Eckmann, Kamphorst, Ruelle & Ciliberto (1986) <https://doi.org/10.1103/PhysRevA.34.4971>`__
     - `nolds <https://github.com/CSchoel/nolds>`__ (``lyap_e``; matches to float64 precision for matrix step 1 on well-conditioned fits. For matrix step > 1 nolds composes the fitted maps at every consecutive point where the paper's Sec. 2c composes them at stride ``m``, so this follows the paper, checked against an independent NumPy implementation; degenerate or rank-deficient rows are ``NaN`` and ties beyond ``num_neighbors`` are dropped -- see module docstring); no Julia equivalent

   * - :doc:`kaplan_yorke_dimension <generated/torchnlts.lyapunov.kaplan_yorke_dimension>`
     - piecewise (continuous, with kinks where a partial sum crosses zero or sorted exponents tie; jumps where the largest exponent crosses zero)
     - `Kaplan & Yorke (1979) <https://doi.org/10.1007/BFb0064319>`__; Frederickson, Kaplan, Yorke & Yorke (1983), *J. Diff. Eq.* 49, 185-207
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``kaplanyorke_dim``; matches to float64 precision, including zero and all-negative edge cases); literature Hénon (1.258) and Lorenz (2.06) values

   * - :doc:`phase_randomized_surrogate <generated/torchnlts.surrogate.phase_randomized_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``RandomFourier``, run via subprocess; both implementations share the literal recipe's property of not preserving the Nyquist magnitude for even-length input -- see module docstring)

   * - :doc:`aaft_surrogate <generated/torchnlts.surrogate.aaft_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``AAFT``, run via subprocess; Julia skips the paper's Gaussian rank-matching step and equals ``RandomFourier`` on the raw data followed by rank restoration, so the two differ for non-Gaussian data. This implementation follows the paper -- see module docstring)

   * - :doc:`iaaft_surrogate <generated/torchnlts.surrogate.iaaft_surrogate>`
     - none
     - `Schreiber & Schmitz (1996) <https://doi.org/10.1103/PhysRevLett.77.635>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``IAAFT``, run via subprocess; both show the same seed-dependent spread of the converged spectral error. Differences: a random-permutation start instead of resampling with replacement, and a fixed iteration count instead of early stopping -- see module docstring)

   * - :doc:`random_shuffle_surrogate <generated/torchnlts.surrogate.random_shuffle_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``RandomShuffle``; uniform permutation, checked by property since RNG streams differ)

   * - :doc:`random_shift_surrogate <generated/torchnlts.surrogate.random_shift_surrogate>`
     - none
     - `Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``CircShift``; Julia takes the shifts from the caller, here uniform on 0..T-1, the whole cyclic group including the identity, so the rank test is exact)

   * - :doc:`surrogate_test <generated/torchnlts.surrogate.surrogate_test>`
     - none (rank count)
     - `Theiler et al. (1992) <https://doi.org/10.1016/0167-2789(92)90102-S>`__; `Schreiber & Schmitz (2000) <https://doi.org/10.1016/S0167-2789(00)00043-9>`__
     - `TimeseriesSurrogates.jl <https://github.com/JuliaDynamics/TimeseriesSurrogates.jl>`__ (``pvalue(SurrogateTest(...))``, run via subprocess; Julia returns k/M, here (1+k)/(M+1) with the same k; two-sided 2(1+min)/(M+1) clipped to 1)

   * - :doc:`cross_map_estimate <generated/torchnlts.causality.cross_map_estimate>`
     - piecewise
     - `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``ConvergentCrossMapping``/``predict`` in the ``juliaenv_causality`` project; matches to float64 precision (~1e-16 per point) with ``theiler_window = w + 1`` for Julia's ``w``, whose Theiler window excludes ``|i - j| <= w + 1`` -- see module docstring)

   * - :doc:`ccm_skill <generated/torchnlts.causality.ccm_skill>`
     - exact
     - `Sugihara et al. (2012) <https://doi.org/10.1126/science.1227079>`__
     - none (a Pearson correlation, not a published algorithm of its own; the Julia comparison of ``cross_map_estimate`` includes the correlation)

   * - :doc:`conditional_mutual_information <generated/torchnlts.causality.conditional_mutual_information>`
     - none
     - `Frenzel & Pompe (2007) <https://doi.org/10.1103/PhysRevLett.99.204101>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``FPVP``; it counts marginal neighbors with ``<=`` where Frenzel & Pompe use ``<``, the same ``Neighborhood.jl`` primitive behind ``mutual_information``'s ``KSG1`` difference, so the estimates differ -- see module docstring)

   * - :doc:`transfer_entropy <generated/torchnlts.causality.transfer_entropy>`
     - none
     - `Schreiber (2000) <https://doi.org/10.1103/PhysRevLett.85.461>`__
     - `CausalityTools.jl <https://github.com/JuliaDynamics/CausalityTools.jl>`__ (``TEShannon`` through ``CMIDecomposition``/``FPVP``; matches to float64 precision when both count inclusively, which checks the alignment of the source past, target past and target future -- see module docstring)

   * - :doc:`LyapunovLoss <generated/torchnlts.sciml.LyapunovLoss>`
     - exact
     - none (wraps ``lyapunov_spectrum``)
     - none (compared with a direct call of ``lyapunov_spectrum``; the gradient inherits its exploding-variance caveat for chaotic maps)

   * - :doc:`CorrelationDimensionLoss <generated/torchnlts.sciml.CorrelationDimensionLoss>`
     - smooth
     - none (wraps ``soft_correlation_sum`` and ``correlation_dimension``)
     - none (compared with a direct call of those functions)

   * - :doc:`RecurrenceLoss <generated/torchnlts.sciml.RecurrenceLoss>`
     - smooth
     - none (wraps ``soft_recurrence_matrix`` and ``recurrence_rate``)
     - none (compared with a direct call of those functions)

   * - :doc:`AttractorLoss <generated/torchnlts.sciml.AttractorLoss>`
     - smooth
     - `Gretton et al. (2012) <https://jmlr.org/papers/v13/gretton12a.html>`__ (Maximum Mean Discrepancy)
     - none (composes ``trajectory_kernel``; the biased MMD estimator is compared with an independent computation, not with a published MMD implementation)

   * - :doc:`SinkhornLoss <generated/torchnlts.sciml.SinkhornLoss>`
     - smooth
     - `Feydy et al. (2019) <https://proceedings.mlr.press/v89/feydy19a.html>`__ (Sinkhorn divergence); `Jiang, Lu, Orlova & Willett (2023) <https://arxiv.org/abs/2306.01187>`__ (applied to chaotic attractors)
     - none (entropic optimal transport is not implemented elsewhere in the library). As the regularization shrinks, the cost converges to the exact optimal transport cost between two equal-size uniform point clouds, computed with ``scipy.optimize.linear_sum_assignment``; the regularized objective also matches a brute-force computation from the explicit transport plan.

   * - :doc:`linear_regions <generated/torchnlts.scaling.linear_regions>` / :doc:`largest_linear_region <generated/torchnlts.scaling.largest_linear_region>`
     - none (integer labels and indices)
     - none (sequential ``dxi = 1`` linear-region scan of FractalDimensions.jl)
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``linear_regions``, ``linear_region`` with ``LargestLinearRegion``; regions and largest region match exactly on hand-built curves, a saturating curve and a Lorenz correlation-sum curve). Not implemented: ``dxi`` stride, ``:overlap`` method.

   * - :doc:`scaling_slope <generated/torchnlts.scaling.scaling_slope>`
     - exact (the region choice is piecewise constant and carries no gradient)
     - none (sequential ``dxi = 1`` linear-region scan of FractalDimensions.jl)
     - `FractalDimensions.jl <https://github.com/JuliaDynamics/FractalDimensions.jl>`__ (``slopefit`` with ``LargestLinearRegion``; the slope matches exactly on hand-built curves, a saturating curve and a Lorenz correlation-sum curve)

   * - :doc:`matrix_profile <generated/torchnlts.matrixprofile.matrix_profile>`
     - piecewise (the index has none)
     - `Yeh et al. (2016) <https://doi.org/10.1109/ICDM.2016.0179>`__, `Zhu et al. (2016) <https://doi.org/10.1109/ICDM.2016.0085>`__
     - none in Julia; `stumpy <https://github.com/stumpy-dev/stumpy>`__ (``stump``, self-join and AB-join; profile to 1e-6 in float64, index up to ties). Forms the full distance matrix, O(T^2) memory.

   * - :doc:`motifs <generated/torchnlts.matrixprofile.motifs>` / :doc:`discords <generated/torchnlts.matrixprofile.discords>`
     - none
     - `Yeh et al. (2016) <https://doi.org/10.1109/ICDM.2016.0179>`__
     - none in Julia; `stumpy <https://github.com/stumpy-dev/stumpy>`__ (``motifs`` with ``max_matches=2``); discords hand-derived

   * - :doc:`corrected_arc_curve <generated/torchnlts.matrixprofile.corrected_arc_curve>` / :doc:`regime_boundaries <generated/torchnlts.matrixprofile.regime_boundaries>`
     - none
     - `Gharghabi et al. (2017) <https://doi.org/10.1109/ICDM.2017.21>`__
     - none in Julia; `stumpy <https://github.com/stumpy-dev/stumpy>`__ (``fluss`` with ``custom_iac`` set to the paper's parabola ``2 j (n - j) / n``; stumpy's default idealized arc curve is a fitted beta density, so its default curve differs)
