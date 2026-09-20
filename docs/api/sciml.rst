SciML Reference
=================

Invariant-matching training losses for differentiable dynamics models --
the ``torchnlts.sciml`` subpackage. Each loss composes primitives
already validated elsewhere in the package; see the module docstring for
why none needs its own fresh paper/Julia validation, and for
:class:`~torchnlts.sciml.losses.LyapunovLoss`'s inherited
gradient-stability caveat.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.sciml.losses.LyapunovLoss
   torchnlts.sciml.losses.CorrelationDimensionLoss
   torchnlts.sciml.losses.RecurrenceLoss
   torchnlts.sciml.losses.AttractorLoss
