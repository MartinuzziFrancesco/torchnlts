Lyapunov Reference
===================

Largest Lyapunov exponent estimation and the full Lyapunov spectrum -- the
``torchnlts.lyapunov`` subpackage.

Gradient stability note: Rosenstein selects a nearest neighbor per point;
Kantz selects a radius neighborhood and averages its evolved distances
before taking the log. Both average over reference points and use a slope
fit to estimate the exponent. They do
not internally roll out a dynamics model. That local computation does
not itself have the chaos-induced exploding-variance gradient structure Metz et al.
("Gradients are Not All You Need", 2021/2022) describe (products of
state Jacobians accumulated over many steps of a repeated map).
See `Metz et al. <https://arxiv.org/abs/2111.05803>`_.

The local operations are not uniformly well-conditioned: log-distance
gradients grow as separation approaches zero, exact duplicates can
produce ``-inf`` and undefined gradients, and neighbor-membership changes
are discrete boundaries. The slope fit is also ill-conditioned when
the fitted steps have little variation, and undefined when their
variance is zero. Distances are not clamped or regularized.

This does *not* mean gradients through these functions are always
well-behaved: if ``trajectory`` is itself the output of rolling out a
model over many steps (e.g. computing a Lyapunov-based loss for training
a differentiable dynamics model, as in
``sciml.losses.LyapunovLoss``), backpropagating through *that* rollout
inherits the same exploding-variance risk regardless of what's computed
on top of it -- these functions neither cause nor fix that, and offer no
built-in mitigation for it. Any mitigation (truncated/windowed BPTT,
gradient clipping, or a documented fallback to a black-box estimator)
has to be applied to the rollout itself.

``lyapunov_spectrum`` faces this directly rather than only inheriting it:
it computes the full Lyapunov spectrum of a differentiable ``step_fn``
(a discrete map, or one fixed-step continuous-flow integration) via the
Benettin/QR ("H2") method, using ``torch.func.jacrev``/``vmap`` for the
per-step Jacobian. Differentiating its output w.r.t. anything upstream of
``initial_state`` or ``step_fn`` means backpropagating through
``num_steps`` chained Jacobian-and-QR steps -- the same
products-of-state-Jacobians structure Metz et al. describe. It applies no
mitigation of its own and exposes no hook to add one inside its own loop;
see its own docstring for what's actually actionable (keeping
``num_steps`` modest, clipping gradients in the surrounding optimizer
step, or detaching state in an outer rollout the caller controls,
upstream of calling this function at all).

Its returned spectrum is also only *generically* sorted descending, not
unconditionally: see its own docstring for a confirmed non-generic
counterexample (a diagonal linear map with unsorted diagonal entries)
where ChaosTools.jl's ``lyapunovspectrum`` returns the same
non-descending order -- a shared limitation of the Benettin/QR method's
sensitivity to the initial deviation vectors, not a divergence between
this implementation and its reference.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.lyapunov.rosenstein.rosenstein_divergence
   torchnlts.lyapunov.rosenstein.rosenstein_lyapunov
   torchnlts.lyapunov.kantz.kantz_divergence
   torchnlts.lyapunov.jacobian.step_jacobian
   torchnlts.lyapunov.spectrum.lyapunov_spectrum
