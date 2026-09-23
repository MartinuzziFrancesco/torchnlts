SciML Reference
=================

Invariant-matching training losses for differentiable dynamics models --
the ``torchnlts.sciml`` subpackage. Most of these losses compose
primitives already validated elsewhere in the package; see the module
docstring for why those don't need their own fresh paper/Julia
validation, for :class:`~torchnlts.sciml.losses.LyapunovLoss`'s
inherited gradient-stability caveat, and for why
:class:`~torchnlts.sciml.losses.SinkhornLoss` -- introducing entropic
optimal transport, not reused from elsewhere in this codebase -- is
validated differently, against an independent ground truth.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.sciml.losses.LyapunovLoss
   torchnlts.sciml.losses.CorrelationDimensionLoss
   torchnlts.sciml.losses.RecurrenceLoss
   torchnlts.sciml.losses.AttractorLoss
   torchnlts.sciml.losses.SinkhornLoss
