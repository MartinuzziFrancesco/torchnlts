Lyapunov Reference
===================

Largest Lyapunov exponent estimation and the full Lyapunov spectrum -- the
``torchnlts.lyapunov`` subpackage.

Gradient notes. Rosenstein selects a nearest neighbor per point and Kantz a radius neighborhood whose evolved
distances are averaged before the log; both average over reference points and fit a slope. Neither rolls out a
dynamics model, so neither has the exploding-variance gradients of products of state Jacobians (Metz et al.,
`Gradients are Not All You Need <https://arxiv.org/abs/2111.05803>`_). They are still not uniformly well-conditioned:
log-distance gradients grow as a separation approaches zero, exact duplicates give ``-inf`` and undefined gradients,
neighbor membership changes at discrete boundaries, and the slope fit is ill-conditioned when the fitted steps vary
little (undefined at zero variance). Distances are not clamped or regularized. If ``trajectory`` is itself the rollout
of a model (e.g. :class:`~torchnlts.sciml.LyapunovLoss`), backpropagating through that rollout carries the
exploding-variance risk whatever is computed on top of it; truncated backpropagation or gradient clipping has to be
applied to the rollout.

:func:`~torchnlts.lyapunov.lyapunov_spectrum` differentiates through ``num_steps`` chained Jacobian and QR steps and
so carries that risk itself; it applies no mitigation (see its docstring). Its spectrum is generically, not
unconditionally, sorted descending: for a diagonal linear map with unsorted entries ChaosTools.jl returns the same
non-descending order.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.lyapunov.rosenstein_divergence
   torchnlts.lyapunov.rosenstein_lyapunov
   torchnlts.lyapunov.kantz_divergence
   torchnlts.lyapunov.step_jacobian
   torchnlts.lyapunov.lyapunov_spectrum
   torchnlts.lyapunov.eckmann_spectrum
   torchnlts.lyapunov.kaplan_yorke_dimension
