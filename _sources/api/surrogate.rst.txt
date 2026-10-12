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

   torchnlts.surrogate.phase_randomized_surrogate
   torchnlts.surrogate.aaft_surrogate
   torchnlts.surrogate.iaaft_surrogate
   torchnlts.surrogate.random_shuffle_surrogate
   torchnlts.surrogate.random_shift_surrogate
   torchnlts.surrogate.surrogate_test
   torchnlts.surrogate.SurrogateTestResult
