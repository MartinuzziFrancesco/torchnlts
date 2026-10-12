SciML Reference
=================

Invariant-matching training losses for differentiable dynamics models --
the ``torchnlts.sciml`` subpackage. The Lyapunov, correlation-dimension and
recurrence losses match a statistic of a model rollout to a target value;
:class:`~torchnlts.sciml.AttractorLoss` (maximum mean discrepancy) and
:class:`~torchnlts.sciml.SinkhornLoss` (optimal transport) compare the
state-space point clouds of two trajectories. They are added to an MSE term.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.sciml.LyapunovLoss
   torchnlts.sciml.CorrelationDimensionLoss
   torchnlts.sciml.RecurrenceLoss
   torchnlts.sciml.AttractorLoss
   torchnlts.sciml.SinkhornLoss
