Surrogate Reference
====================

Surrogate data generation for nonlinearity testing -- the
``torchnlts.surrogate`` subpackage. Analysis-only, non-differentiable
utilities: every function here
draws fresh randomness on each call and is not meant to be
backpropagated through.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.surrogate.phase.phase_randomized_surrogate
   torchnlts.surrogate.aaft.aaft_surrogate
   torchnlts.surrogate.iaaft.iaaft_surrogate
