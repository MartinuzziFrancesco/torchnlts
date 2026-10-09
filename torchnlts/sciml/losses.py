"""Invariant-matching training losses for differentiable dynamics models.

Each loss in this module composes primitives already implemented and
validated elsewhere in the library -- none of them introduces a new
*statistic* of its own, so none needs a fresh paper/Julia validation of
the underlying quantity being matched. That's not the whole correctness
story for all five, though: three (``LyapunovLoss``,
``CorrelationDimensionLoss``, ``RecurrenceLoss``) really are thin
wiring -- one primitive call, squared error against a target -- and
this module's tests check exactly that wiring by calling the same
primitive directly and comparing. ``AttractorLoss`` additionally
combines three kernel evaluations into an MMD estimate, genuinely new
combination logic that a single reused primitive's own prior validation
doesn't cover on its own -- see its own docstring for a real
correctness constraint this combination introduces (a kernel-validity
requirement on ``norm_order``) that was caught by direct
counterexample, not by inspection. ``SinkhornLoss`` is the one true
new primitive in this module (entropic optimal transport itself isn't
implemented anywhere else in this codebase), so unlike the other four
it's validated against an independent ground truth: as its entropic
regularization shrinks, its cost must converge to the exact (regularization-free)
optimal transport cost between two equal-size point clouds, computed via
``scipy.optimize.linear_sum_assignment`` (the exact discrete-OT solution
when both marginals are uniform and equinumerous -- an optimal transport
plan between equal-size uniform distributions is a permutation). The shared
motivation: training a differentiable dynamics model on pointwise MSE alone
can fit short-term trajectories well while still producing a qualitatively
wrong long-term/statistical behavior (wrong Lyapunov exponent, wrong fractal
dimension, wrong recurrence structure, wrong attractor shape) -- these
losses let a training loop penalize that directly, added to (not
instead of) an ordinary MSE term.

All four are ``torch.nn.Module`` subclasses (configuration -- sweep
values, kernel bandwidth, embedding parameters -- lives on the module,
set once at construction; ``forward`` takes only the data that changes
every training step), matching ``torch.nn.MSELoss``'s own convention,
and all support the same ``reduction in {"mean", "sum", "none"}`` as
that class.

- :class:`~torchnlts.sciml.LyapunovLoss` -- matches a differentiable model's own rollout
  Lyapunov spectrum (:func:`torchnlts.lyapunov.lyapunov_spectrum`)
  against a target. Directly inherits that function's own documented
  gradient-stability caveat (Metz et al., "Gradients are Not All You
  Need"): differentiating this loss backpropagates through
  ``num_steps`` chained Jacobian/QR steps, which can have
  exploding-variance gradients for chaotic ``step_fn``. This loss adds
  no mitigation of its own -- see ``lyapunov_spectrum``'s own docstring
  for what's actionable (keep ``num_steps`` modest, clip gradients in
  the optimizer, or detach state in an outer rollout).
- :class:`~torchnlts.sciml.CorrelationDimensionLoss` -- matches a predicted trajectory's
  fractal (correlation) dimension
  (:func:`torchnlts.dimension.soft_correlation_sum` +
  :func:`torchnlts.dimension.correlation_dimension`)
  against a target. Uses the *soft* correlation sum unconditionally
  (the hard variant's Heaviside threshold has no useful gradient, see
  its own "none" gradient class) -- this is the one place in this
  module a hard/soft choice isn't left to the caller, since a
  non-differentiable loss can't train anything.
- :class:`~torchnlts.sciml.RecurrenceLoss` -- matches a predicted trajectory's recurrence
  rate (:func:`torchnlts.recurrence.soft_recurrence_matrix` +
  :func:`torchnlts.recurrence.recurrence_rate`) against a target,
  same soft-only reasoning as above.
- :class:`~torchnlts.sciml.AttractorLoss` -- unlike the other three (which each match one
  fixed target value -- a scalar for ``CorrelationDimensionLoss`` and
  ``RecurrenceLoss``, a whole spectrum for ``LyapunovLoss``), this
  compares two whole trajectories' state-space point clouds directly,
  via the (biased,
  V-statistic) squared Maximum Mean Discrepancy [Gretton2012]_:
  ``MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]`` for points
  ``x ~ P`` (predicted), ``y ~ Q`` (target), built entirely from
  :func:`torchnlts.neighbors.trajectory_kernel` (the same
  kernel already used for soft recurrence and soft correlation sum)
  applied to the predicted/predicted, target/target, and
  predicted/target pairwise distance matrices. No temporal
  (Theiler-window) exclusion of nearby-in-time point pairs is applied
  to the same-trajectory terms -- the simpler, standard biased MMD
  estimator, not the temporally-debiased version a genuinely rigorous
  treatment might want. Limitation: biased V-statistic MMD, no Theiler
  exclusion on the P,P/Q,Q terms; add if a workload shows the omission
  measurably biasing training (the self-similarity terms' diagonal
  contribution is a model-independent constant either way, so this
  mainly risks nearby-in-time-but-distinct points inflating apparent
  self-similarity, not a gradient sign error).
- :class:`~torchnlts.sciml.SinkhornLoss` -- same purpose as ``AttractorLoss`` (whole-
  attractor point-cloud matching) but the two-sample distance is a
  debiased Sinkhorn divergence [Feydy2019]_ (entropy-regularized
  optimal transport) instead of MMD. Jiang, Lu, Orlova & Willett
  [Jiang2023]_ report Sinkhorn/OT beating MMD specifically for matching
  chaotic attractors' invariant measures during training -- this class
  makes that comparison directly runnable against ``AttractorLoss``
  already in this module, same trajectories, same kernel-free cost
  (squared Euclidean).

.. [Gretton2012] Gretton, A., Borgwardt, K.M., Rasch, M.J., Schoelkopf,
   B. & Smola, A. (2012), "A Kernel Two-Sample Test", Journal of Machine
   Learning Research 13, 723-773.
.. [Feydy2019] Feydy, J. et al. (2019), "Interpolating between Optimal
   Transport and MMD using Sinkhorn Divergences", AISTATS 2019,
   PMLR 89, 2681-2690.
.. [Jiang2023] Jiang, R., Lu, P.Y., Orlova, E. & Willett, R. (2023),
   "Training neural operators to preserve invariant measures of chaotic
   attractors", NeurIPS 2023.
"""

from __future__ import annotations

import torch

from torchnlts.convert import as_tensor
from torchnlts.dimension.correlation import correlation_dimension, soft_correlation_sum
from torchnlts.lyapunov.spectrum import lyapunov_spectrum
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.kernels import trajectory_kernel
from torchnlts.recurrence.matrix import soft_recurrence_matrix
from torchnlts.recurrence.rqa import recurrence_rate


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def _sinkhorn_cost(cost: torch.Tensor, reg: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Entropy-regularized optimal transport *objective value* between two
    uniform empirical distributions, given a pairwise cost matrix:
    ``<P,C> + reg * KL(P | a (x) b)`` at the entropic-optimal plan ``P``
    (Feydy et al. 2019, eq. 1) -- not just the transport term ``<P,C>``
    alone, which is a different, smaller quantity once ``reg > 0``. Dual
    potentials found via log-domain Sinkhorn iteration (Peyre & Cuturi,
    "Computational Optimal Transport", 2019, sec. 4.4 -- the numerically
    stable reformulation of Cuturi (2013)'s original scaling-vector
    algorithm); the regularized objective is then read off the converged
    potentials directly via strong duality (``<f,a> + <g,b>``, adjusted
    for this function's own log-domain potentials already absorbing the
    uniform marginals' log terms -- see this module's tests for the
    independent brute-force check confirming the adjustment), which is
    both the mathematically correct quantity and cheaper than
    materializing ``P`` and its KL term explicitly.

    Args:
        cost: Shape ``[..., num_source, num_target]``.
        reg: Entropic regularization strength (bandwidth), scalar.
        num_iters: Fixed number of Sinkhorn iterations (unrolled and
            differentiated through directly -- the standard approach for
            a differentiable-OT training loss, no implicit-function
            shortcut).

    Returns:
        Shape ``[...]``.
    """
    num_source, num_target = cost.shape[-2], cost.shape[-1]
    log_num_source = torch.log(torch.tensor(float(num_source), dtype=cost.dtype, device=cost.device))
    log_num_target = torch.log(torch.tensor(float(num_target), dtype=cost.dtype, device=cost.device))
    source_potential = torch.zeros(cost.shape[:-1], dtype=cost.dtype, device=cost.device)
    target_potential = torch.zeros(cost.shape[:-2] + cost.shape[-1:], dtype=cost.dtype, device=cost.device)
    for _ in range(num_iters):
        source_potential = reg * (-log_num_source - torch.logsumexp((target_potential.unsqueeze(-2) - cost) / reg, dim=-1))
        target_potential = reg * (-log_num_target - torch.logsumexp((source_potential.unsqueeze(-1) - cost) / reg, dim=-2))
    return source_potential.mean(dim=-1) + target_potential.mean(dim=-1) + reg * (log_num_source + log_num_target)


class LyapunovLoss(torch.nn.Module):
    """Squared error between a differentiable model's rollout Lyapunov
    spectrum and a target spectrum.

    Gradient class: exact (see the module docstring's gradient-stability
    caveat, inherited from :func:`torchnlts.lyapunov.lyapunov_spectrum`).

    Args:
        target: Target Lyapunov exponent(s). Scalar, or shape
            ``[num_exponents]`` to target the full (or a partial)
            spectrum -- broadcasts against ``lyapunov_spectrum``'s own
            output.
        num_steps: Forwarded to ``lyapunov_spectrum``.
        num_exponents: Forwarded to ``lyapunov_spectrum``.
        dt: Forwarded to ``lyapunov_spectrum``.
        transient_steps: Forwarded to ``lyapunov_spectrum``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        target,
        num_steps: int,
        num_exponents: int | None = None,
        dt: float = 1.0,
        transient_steps: int = 0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.num_steps = num_steps
        self.num_exponents = num_exponents
        self.dt = dt
        self.transient_steps = transient_steps
        self.reduction = reduction

    def forward(self, step_fn, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_fn: Forwarded to ``lyapunov_spectrum`` as its own
                ``step_fn`` -- the differentiable dynamics model being
                trained (or evaluated), one discrete step.
            initial_state: Forwarded to ``lyapunov_spectrum``.

        Returns:
            Scalar loss (``reduction != "none"``), or with
            ``reduction="none"`` the unreduced squared error, shape
            ``[num_exponents]`` for an unbatched ``initial_state`` or
            ``[B, num_exponents]`` for a batched one (matching
            ``lyapunov_spectrum``'s own output shape).
        """
        predicted = lyapunov_spectrum(step_fn, initial_state, self.num_steps, self.num_exponents, self.dt, self.transient_steps)
        return _reduce((predicted - self.target.to(predicted.dtype)) ** 2, self.reduction)


class CorrelationDimensionLoss(torch.nn.Module):
    """Squared error between a predicted trajectory's correlation
    dimension and a target dimension.

    Gradient class: smooth (inherits :func:`torchnlts.dimension.soft_correlation_sum`'s
    own class; the OLS slope step itself is exact).

    Args:
        target: Target correlation dimension. Scalar, or ``[B]`` to
            match a batched trajectory.
        temperature: Kernel-bandwidth sweep, forwarded to
            ``soft_correlation_sum`` and (as the ``radius`` argument)
            ``correlation_dimension``. Shape ``[R]``, ``R >= 2``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to
            ``soft_correlation_sum``.
        theiler_window: Forwarded to ``soft_correlation_sum``.
        norm_order: Forwarded to ``soft_correlation_sum``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        target,
        temperature,
        kind: str = "gaussian",
        theiler_window: int = 0,
        norm_order: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.theiler_window = theiler_window
        self.norm_order = norm_order
        self.reduction = reduction

    def forward(self, predicted_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]``, or
                ``[B, T, D]``, forwarded to ``soft_correlation_sum``.

        Returns:
            Scalar loss (``reduction != "none"``), or with
            ``reduction="none"`` the unreduced squared error, shape
            ``[]`` for an unbatched trajectory or ``[B]`` for a batched
            one.
        """
        temperature = self.temperature.to(as_tensor(predicted_trajectory).dtype)
        correlation_sum = soft_correlation_sum(predicted_trajectory, temperature, self.kind, self.theiler_window, self.norm_order)
        dimension = correlation_dimension(temperature, correlation_sum)
        return _reduce((dimension - self.target.to(dimension.dtype)) ** 2, self.reduction)


class RecurrenceLoss(torch.nn.Module):
    """Squared error between a predicted trajectory's recurrence rate
    and a target rate.

    Gradient class: smooth (inherits :func:`torchnlts.recurrence.soft_recurrence_matrix`'s
    own class; :func:`torchnlts.recurrence.recurrence_rate` itself
    is exact).

    Args:
        target: Target recurrence rate, in ``[0, 1]``. Scalar, or
            ``[B]`` to match a batched trajectory.
        temperature: Kernel bandwidth, forwarded to
            ``soft_recurrence_matrix``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to
            ``soft_recurrence_matrix``.
        theiler_window: Forwarded to ``recurrence_rate``.
        norm_order: Forwarded to ``soft_recurrence_matrix``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        target,
        temperature,
        kind: str = "gaussian",
        theiler_window: int = 0,
        norm_order: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.theiler_window = theiler_window
        self.norm_order = norm_order
        self.reduction = reduction

    def forward(self, predicted_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]``, or
                ``[B, T, D]``, forwarded to ``soft_recurrence_matrix``.

        Returns:
            Scalar loss (``reduction != "none"``), or with
            ``reduction="none"`` the unreduced squared error, shape
            ``[]`` for an unbatched trajectory or ``[B]`` for a batched
            one.
        """
        temperature = self.temperature.to(as_tensor(predicted_trajectory).dtype)
        weights = soft_recurrence_matrix(predicted_trajectory, temperature, self.kind, self.norm_order)
        rate = recurrence_rate(weights, self.theiler_window)
        return _reduce((rate - self.target.to(rate.dtype)) ** 2, self.reduction)


class AttractorLoss(torch.nn.Module):
    """Squared Maximum Mean Discrepancy between a predicted trajectory's
    and a target trajectory's state-space point clouds -- a whole-
    attractor-shape loss, unlike this module's other, fixed-target
    invariant losses (see the module docstring).

    Gradient class: smooth (built entirely on
    :func:`torchnlts.neighbors.trajectory_kernel`'s own class).

    Unlike this module's other three losses, ``norm_order`` is not
    configurable here: it's fixed to Euclidean distance, not exposed as
    a parameter. MMD^2 is only guaranteed non-negative when ``k`` is a
    valid positive-semidefinite kernel (Gretton et al. 2012, Sec. 2.2),
    which both ``trajectory_kernel`` kinds are, but *only* when the
    distance they're applied to is Euclidean -- confirmed by direct
    counterexample that plugging in an L1 (Manhattan) distance instead
    can make the Gaussian-kind kernel non-PSD and this loss go negative,
    which would give a nonsensical training signal (there's no "more
    negative is worse" reading of a squared distance). Not a
    theoretical concern being preempted: reproduced directly with two
    4-point trajectories and ``norm_order=1``.

    Args:
        temperature: Kernel bandwidth, forwarded to ``trajectory_kernel``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to
            ``trajectory_kernel``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        temperature,
        kind: str = "gaussian",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.kind = kind
        self.reduction = reduction

    def forward(self, predicted_trajectory, target_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]``, or
                ``[B, T, D]``.
            target_trajectory: Second trajectory, same batched-ness,
                dtype, and feature dimensionality as
                ``predicted_trajectory`` (mismatched dtypes raise from
                ``pairwise_distance``'s underlying ``torch.cdist``
                rather than being auto-promoted, matching this
                codebase's usual trust-the-caller convention for
                dtype-matched inputs); its own time length may differ
                (see :func:`torchnlts.neighbors.pairwise_distance`'s
                own ``other`` argument, used directly for the cross
                term below).

        Returns:
            Scalar loss (``reduction != "none"``), shape ``[]`` for
            unbatched trajectories or ``[B]`` for batched ones with
            ``reduction="none"``.
        """
        predicted_distance = pairwise_distance(predicted_trajectory, norm_order=2.0)
        temperature = self.temperature.to(predicted_distance.dtype)
        pp = trajectory_kernel(predicted_distance, temperature, self.kind).mean(dim=(-2, -1))
        qq = trajectory_kernel(pairwise_distance(target_trajectory, norm_order=2.0), temperature, self.kind).mean(dim=(-2, -1))
        pq = trajectory_kernel(
            pairwise_distance(predicted_trajectory, other=target_trajectory, norm_order=2.0), temperature, self.kind
        ).mean(dim=(-2, -1))
        mmd_squared = pp + qq - 2 * pq
        return _reduce(mmd_squared, self.reduction)


class SinkhornLoss(torch.nn.Module):
    """Debiased Sinkhorn divergence between a predicted trajectory's and
    a target trajectory's state-space point clouds -- a whole-attractor
    -shape loss like :class:`~torchnlts.sciml.AttractorLoss`, but comparing point clouds
    via entropy-regularized optimal transport (Jiang, Lu, Orlova &
    Willett, 2023) instead of MMD.

    Gradient class: smooth (:func:`_sinkhorn_cost` is a fixed, finite
    number of ``exp``/``logsumexp`` iterations, unrolled and
    differentiated through directly).

    Args:
        reg: Entropic regularization strength. Smaller values approach
            the exact (unregularized) optimal transport cost more
            closely but need more ``num_iters`` to converge and can lose
            numerical precision; must be positive (enforced -- ``reg<=0``
            makes ``_sinkhorn_cost`` divide by zero/flip sign silently,
            not raise on its own).
        num_iters: Number of Sinkhorn iterations; must be at least 1
            (enforced -- ``num_iters<=0`` would otherwise silently skip
            the dual-potential updates and return a meaningless value
            from the zero-initialized potentials).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(self, reg, num_iters: int = 50, reduction: str = "mean") -> None:
        super().__init__()
        reg_tensor = torch.as_tensor(reg)
        if (reg_tensor <= 0).any():
            raise ValueError(f"reg must be positive, got {reg!r}")
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, got {num_iters!r}")
        self.register_buffer("reg", reg_tensor)
        self.num_iters = num_iters
        self.reduction = reduction

    def forward(self, predicted_trajectory, target_trajectory) -> torch.Tensor:
        """
        Args:
            predicted_trajectory: Shape ``[T]``, ``[T, D]``, or
                ``[B, T, D]``.
            target_trajectory: Second trajectory, same convention as
                :class:`~torchnlts.sciml.AttractorLoss`'s own ``target_trajectory``.

        Returns:
            Scalar loss (``reduction != "none"``), shape ``[]`` for
            unbatched trajectories or ``[B]`` for batched ones with
            ``reduction="none"``.
        """
        reg = self.reg.to(as_tensor(predicted_trajectory).dtype)
        predicted_cost = 0.5 * pairwise_distance(predicted_trajectory, norm_order=2.0) ** 2
        target_cost = 0.5 * pairwise_distance(target_trajectory, norm_order=2.0) ** 2
        cross_cost = 0.5 * pairwise_distance(predicted_trajectory, other=target_trajectory, norm_order=2.0) ** 2
        pp = _sinkhorn_cost(predicted_cost, reg, self.num_iters)
        qq = _sinkhorn_cost(target_cost, reg, self.num_iters)
        pq = _sinkhorn_cost(cross_cost, reg, self.num_iters)
        divergence = pq - 0.5 * (pp + qq)
        return _reduce(divergence, self.reduction)
