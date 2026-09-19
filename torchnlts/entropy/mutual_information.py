"""Kraskov-Stogbauer-Grassberger (KSG1) mutual information estimator.

Reference: Kraskov, A., Stogbauer, H. & Grassberger, P. (2004),
"Estimating mutual information", Phys. Rev. E 69, 066138, eq. 6-8 (their
first estimator, I^(1)): for paired samples (x_i, y_i), find each
point's distance to its k-th nearest neighbor in the joint space under
the Chebyshev (max) norm -- excluding temporally correlated points via
a Theiler window -- then count, independently in each marginal space
(no Theiler exclusion there), how many *other* points lie strictly
closer than that same radius (eq. 6-7's ``n_x(i)``/``n_y(i)``, j != i).
Averaging digamma(n_x(i) + 1) + digamma(n_y(i) + 1) over all points
gives a bias-corrected Shannon mutual information estimate, in nats
(eq. 8). For positive radii, including self in the strict count supplies
the formula's ``+ 1``. At zero radius, self must still contribute that
``+ 1`` explicitly, even though no distance is strictly less than zero.

Gradient class: none (k-th-nearest-neighbor selection and the
threshold counts are discrete; :func:`torch.digamma` is itself smooth,
but that doesn't rescue the discrete steps feeding it).

Like :mod:`torchnlts.recurrence.joint_recurrence`, ``x`` and ``y`` need
not share a feature space, only a time axis -- unequal lengths are
truncated to the shorter one.

Known, confirmed deviation from CausalityTools.jl's ``KSG1``: that
estimator counts marginal neighbors with an *inclusive* (``<=``) radius
query (`Neighborhood.jl`'s ``inrangecount``, confirmed by direct
probing -- not a documented convention in its own docstring), rather
than eq. 6-7's strict ``<``. This isn't a rare floating-point tie: by
construction, the point achieving the joint radius always sits exactly
at that radius in at least one marginal, so the two conventions differ
on essentially every point, not just at coincidental ties (see
``tests/test_mutual_information.py``'s locked-in regression comparing
both conventions on the same input). This module implements the
paper's own strict definition; see that test for the confirmed,
quantified disagreement rather than an attempt to reconcile the two.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, apply_theiler_window


def mutual_information(
    trajectory,
    other,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """KSG1 Shannon mutual information estimate between two paired trajectories.

    Gradient class: none (see module docstring).

    Reference: Kraskov, Stogbauer & Grassberger (2004), eq. 6-8. See the
    module docstring for a confirmed, deliberate deviation from
    CausalityTools.jl's ``KSG1`` (strict vs. that package's inclusive
    marginal-counting radius) -- Julia ground truth here is a locked
    disagreement test with hand-verified expected values on both sides,
    not a numerical match, and its Theiler-window handling (applied only
    to the joint-space k-th-neighbor search, never to marginal counting)
    is separately confirmed to match CausalityTools.jl's ``KSG1`` source.

    ``norm_order`` sets a single Minkowski order shared by both marginals
    *and* the joint metric (the joint radius is always the elementwise
    max of the two marginal distances, which equals Chebyshev distance on
    their concatenation only when both marginals also use the default
    ``norm_order=inf``). This differs from CausalityTools.jl's ``KSG1``,
    whose joint metric is hardcoded to Chebyshev independent of its
    separately configurable ``metric_marginals`` -- this function instead
    keeps the single shared ``norm_order`` convention used throughout
    this codebase, at the cost of that independence.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame. A ``NaN`` anywhere in
            a given time point NaNs that whole point's contribution (see
            Returns) rather than raising -- a raise would need a
            data-dependent Python ``if`` on the tensor's contents, which
            rejects ``torch.vmap``'s ``BatchedTensor`` inputs outright.
        other: Second trajectory, same batched-ness as ``trajectory`` (see
            :func:`torchnlts.recurrence.joint_recurrence.hard_joint_recurrence_matrix`);
            its time length and feature dimensionality may otherwise differ.
        num_neighbors: k, the neighbor rank whose joint-space distance sets
            each point's marginal counting radius.
        theiler_window: Theiler window, applied only when searching for
            each point's k-th joint-space neighbor (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`) --
            not applied to the marginal counting step, matching
            CausalityTools.jl's KSG1. Must be ``>= 0`` (self-matches must
            be excluded unconditionally, or the "nearest neighbor" is
            always the point itself at distance 0).
        norm_order: Shared Minkowski norm order for the marginal distance
            matrices and, through them, the joint metric (``inf`` for the
            paper's and CausalityTools.jl's default Chebyshev choice).

    Returns:
        Scalar (unbatched trajectories) or ``[B]`` (batched) mutual
        information estimate, in nats. Can be negative or exceed the
        true value -- this is a finite-sample estimator, not a
        probability, and is only asymptotically unbiased (and, like any
        continuous-data estimator, has no special handling for exact
        duplicate points: a radius of exactly 0 is well-defined by the
        formula above but not statistically meaningful). ``NaN`` for any
        batch element where some point has fewer than ``num_neighbors``
        eligible neighbors after Theiler exclusion (including when
        ``num_neighbors >= length``) -- the honest answer for an
        undefined k-th-neighbor radius rather than a silently-wrong
        finite number from treating a missing neighbor as infinitely far
        -- or where some point contains ``NaN`` (see Args).

    Raises:
        ValueError: If ``num_neighbors < 1``, ``theiler_window < 0``, or
            ``trajectory``/``other`` disagree on batched-ness or batch
            size (see :func:`torchnlts._shape._validate_matching_batch`).
    """
    if num_neighbors < 1:
        raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
    _require_self_exclusion(theiler_window, "KSG mutual information")

    x, x_ndim = _ensure_batched(trajectory)
    y, y_ndim = _ensure_batched(other)
    _validate_matching_batch(x, y, x_ndim, y_ndim, check_feature_dim=False)
    length = min(x.shape[-2], y.shape[-2])
    x, y = x[..., :length, :], y[..., :length, :]
    invalid_point = torch.isnan(x).any(dim=-1) | torch.isnan(y).any(dim=-1)

    dist_x = pairwise_distance(x, norm_order=norm_order)
    dist_y = pairwise_distance(y, norm_order=norm_order)
    joint_distance = apply_theiler_window(torch.maximum(dist_x, dist_y), theiler_window)
    if num_neighbors > length:
        joint_distance = torch.nn.functional.pad(joint_distance, (0, num_neighbors - length), value=float("inf"))
    radius = joint_distance.topk(num_neighbors, dim=-1, largest=False).values[..., -1]

    # Include self unconditionally to implement n_x + 1 and n_y + 1,
    # including when duplicate joint points make the radius zero.
    self_mask = torch.eye(length, dtype=torch.bool, device=dist_x.device)
    count_x = ((dist_x < radius.unsqueeze(-1)) | self_mask).sum(dim=-1).to(dist_x.dtype)
    count_y = ((dist_y < radius.unsqueeze(-1)) | self_mask).sum(dim=-1).to(dist_y.dtype)
    per_point = torch.digamma(count_x) + torch.digamma(count_y)
    # A tensor-valued `if torch.isnan(...).any(): raise` would reject
    # vmap's BatchedTensor inputs outright (data-dependent Python control
    # flow); masking into NaN here instead is a plain tensor op, so NaN
    # input degrades to the same documented NaN output as an insufficient
    # neighbor count, under vmap or not.
    valid = torch.isfinite(radius) & ~invalid_point
    per_point = torch.where(valid, per_point, torch.full_like(per_point, float("nan")))
    digamma_k = torch.digamma(torch.tensor(float(num_neighbors), dtype=dist_x.dtype, device=dist_x.device))
    digamma_n = torch.digamma(torch.tensor(float(length), dtype=dist_x.dtype, device=dist_x.device))
    mi = digamma_k + digamma_n - per_point.mean(dim=-1)
    return _restore_batch(mi, x_ndim)
