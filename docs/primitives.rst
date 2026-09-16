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
