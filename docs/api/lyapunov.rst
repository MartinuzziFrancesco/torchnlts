Lyapunov Reference
===================

Largest Lyapunov exponent estimation -- the ``torchnlts.lyapunov`` subpackage.

Gradient stability note: these estimators select nearest neighbors,
compute log-distances between evolved points, and fit a slope. They do
not internally roll out a dynamics model. That local computation does
not itself have the chaos-induced exploding-variance gradient structure Metz et al.
("Gradients are Not All You Need", 2021/2022) describe (products of
state Jacobians accumulated over many steps of a repeated map).
See `Metz et al. <https://arxiv.org/abs/2111.05803>`_.

The local operations are not uniformly well-conditioned: log-distance
gradients grow as separation approaches zero, exact duplicates can
produce ``-inf`` and undefined gradients, and nearest-neighbor changes
are discrete boundaries. The slope fit is also ill-conditioned when
the fitted steps have little variation, and undefined when their
variance is zero. Distances are not clamped or regularized.

This does *not* mean gradients through these functions are always
well-behaved: if ``trajectory`` is itself the output of rolling out a
model over many steps (e.g. computing a Lyapunov-based loss for training
a differentiable dynamics model, the planned
``sciml.losses.LyapunovLoss``), backpropagating through *that* rollout
inherits the same exploding-variance risk regardless of what's computed
on top of it -- these functions neither cause nor fix that, and offer no
built-in mitigation for it. Any mitigation (truncated/windowed BPTT,
gradient clipping, or a documented fallback to a black-box estimator)
has to be applied to the rollout itself. The planned ``torch.func``-based
Jacobian/QR Lyapunov *spectrum* estimator for differentiable dynamics
models (not yet implemented) will face this directly, since computing it
generally requires differentiating through the model's own rollout, and
will document its chosen mitigation in its own docstring when it's built.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.lyapunov.rosenstein.rosenstein_divergence
   torchnlts.lyapunov.rosenstein.rosenstein_lyapunov
