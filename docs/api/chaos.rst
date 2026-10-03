Chaos Detection Reference
=========================

The Gottwald-Melbourne 0-1 test for chaos -- the ``torchnlts.chaos``
subpackage.

The test maps a scalar series to a planar random-walk-like path for each
caller-supplied frequency ``c`` in ``(0, pi)`` and measures how fast its
modified mean square displacement grows (``K_c``); the median over
frequencies is near 0 for regular dynamics and near 1 for chaos. The
frequencies are explicit arguments (the paper draws about 100 at random from
``(pi/5, 4*pi/5)``), so the functions are deterministic. Continuous-time
data must not be oversampled.

Gradient note: ``K_c`` is a correlation coefficient of differentiable
quantities, so its gradient is smooth wherever the displacement is not
constant; the median ``K`` is piecewise. Constant series and series too
short for ``max_lag`` return ``NaN``.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.chaos.zero_one.zero_one_growth_rate
   torchnlts.chaos.zero_one.zero_one_test
