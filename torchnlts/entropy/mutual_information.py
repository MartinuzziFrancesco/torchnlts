"""Kraskov-Stogbauer-Grassberger (KSG) mutual information estimators.

Reference: Kraskov, Stogbauer & Grassberger (2004), Phys. Rev. E 69, 066138.

``estimator="ksg1"`` (eq. 6-8, their I^(1)): for each point, take the distance to its k-th nearest neighbor in the
joint space (the larger of the two marginal distances), excluding temporally correlated points with a Theiler window.
Then count, in each marginal space and without Theiler exclusion, the other points (j != i) strictly closer than that
radius (``n_x(i)``, ``n_y(i)``). Averaging ``digamma(n_x(i) + 1) + digamma(n_y(i) + 1)`` gives the estimate
``digamma(k) + digamma(N) - mean(...)`` in nats (eq. 8). The point itself is counted (the formula's ``+ 1``), also at
zero radius, where no distance is strictly smaller.

``estimator="ksg2"`` (eq. 9, their I^(2)): per point, the marginal radii are the extents of its ``k`` nearest joint
neighbors in each marginal (``eps_x(i) / 2``, ``eps_y(i) / 2``), the other points within those radii are counted
inclusively (``<=``, as the paper specifies), and the estimate is ``psi(k) - 1/k - <psi(n_x) + psi(n_y)> + psi(N)``.
Ties among equidistant joint neighbors are broken by :func:`torch.topk`'s order.

Like :mod:`torchnlts.recurrence.joint_recurrence`, ``trajectory`` and ``other`` need not share a feature space, only
a time axis; unequal lengths are truncated to the shorter one. One ``norm_order`` is shared by both marginals and the
joint metric (the elementwise max of the marginal distances, which is the Chebyshev distance of their concatenation
only for ``norm_order=inf``); CausalityTools.jl instead fixes the joint metric to Chebyshev.

Deviations from CausalityTools.jl, which this module does not reproduce (it follows the paper):

- ``KSG1`` counts marginal neighbors with an inclusive (``<=``) radius instead of eq. 6-7's strict ``<``. The two
  differ on essentially every point, because the point that sets the joint radius lies exactly at that radius in one
  marginal.
- ``KSG2`` agrees at ``k=1`` at ordinary data scales but differs for ``k>1`` (it uses only the ``k``-th neighbor's
  marginal distance and subtracts ``1`` for ``1/k``), and at any ``k`` it adds ``10*eps()`` to each marginal radius,
  which changes counts for data near 1e-16 (``[0, 1, 4, 10] * 1e-16`` gives -1/6 here and -13/6 there).

Gradient class: none (neighbor selection and threshold counts are discrete).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, apply_theiler_window


def mutual_information(
    trajectory,
    other,
    *,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
    estimator: str = "ksg1",
) -> torch.Tensor:
    """KSG Shannon mutual information estimate between two paired trajectories.

    Reference: Kraskov, Stogbauer & Grassberger (2004), eq. 6-8 (KSG1) or eq. 9 (KSG2). Unlike CausalityTools.jl,
    ``KSG1`` counts marginal neighbors strictly (the paper's ``<``; that package uses ``<=``), and ``KSG2`` uses the
    paper's ``1/k`` term and all ``k`` joint neighbors (that package uses only the k-th and ``1`` for ``k > 1``, and
    adds ``10*eps()`` to each marginal radius, which changes counts for data near 1e-16).

    Gradient class: none.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        other: Second trajectory with the same batched-ness (see
            :func:`torchnlts.recurrence.hard_joint_recurrence_matrix`); its time length and feature dimension may
            differ, and unequal lengths are truncated to the shorter.
        num_neighbors: k, the neighbor rank whose joint-space distance sets each point's marginal counting radius.
        theiler_window: Theiler window for the joint-space k-th neighbor search only, not for the marginal counts (as
            in CausalityTools.jl's ``KSG1``); must be ``>= 0``, since the point itself must always be excluded.
        norm_order: Minkowski norm order shared by both marginal distance matrices and, through them, the joint metric
            (``inf`` for the paper's Chebyshev choice).
        estimator: ``"ksg1"`` (default; strict marginal counts) or ``"ksg2"`` (inclusive counts within the marginal
            extents of the ``k`` joint neighbors). Marginal counts exclude only the point itself, never Theiler
            neighbors, for both.

    Returns:
        Scalar (unbatched trajectories) or ``[B]`` estimate in nats. As a finite-sample estimator it can be negative
        or exceed the true value, and it has no special handling of duplicate points (a radius of exactly 0 is
        defined by the formula but not statistically meaningful). ``NaN`` for a batch element where some point has
        fewer than ``num_neighbors`` eligible neighbors after Theiler exclusion (including ``num_neighbors >= T``)
        or contains ``NaN`` or ``inf`` (only the first ``min(T, T')`` points count).

    Raises:
        ValueError: If ``estimator`` is not ``"ksg1"`` or ``"ksg2"``, ``num_neighbors`` is not an integer ``>= 1``,
            ``theiler_window`` is not an integer ``>= 0``, or ``trajectory`` and ``other`` disagree on batched-ness or
            batch size.
    """
    if estimator not in ("ksg1", "ksg2"):
        raise ValueError(f"estimator must be 'ksg1' or 'ksg2', got {estimator!r}")
    num_neighbors = _as_int(num_neighbors, "num_neighbors")
    if num_neighbors < 1:
        raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
    theiler_window = _require_self_exclusion(theiler_window, "KSG mutual information")

    x, x_ndim = _ensure_batched(trajectory)
    y, y_ndim = _ensure_batched(other)
    _validate_matching_batch(x, y, x_ndim, y_ndim, check_feature_dim=False)
    length = min(x.shape[-2], y.shape[-2])
    x, y = x[..., :length, :], y[..., :length, :]
    invalid_point = ~torch.isfinite(x).all(dim=-1) | ~torch.isfinite(y).all(dim=-1)

    dist_x = pairwise_distance(x, norm_order=norm_order)
    dist_y = pairwise_distance(y, norm_order=norm_order)
    joint_distance = apply_theiler_window(torch.maximum(dist_x, dist_y), theiler_window)
    if num_neighbors > length:
        joint_distance = torch.nn.functional.pad(joint_distance, (0, num_neighbors - length), value=float("inf"))
    nearest = joint_distance.topk(num_neighbors, dim=-1, largest=False)
    radius = nearest.values[..., -1]
    digamma_k = torch.digamma(torch.tensor(float(num_neighbors), dtype=dist_x.dtype, device=dist_x.device))
    digamma_n = torch.digamma(torch.tensor(float(length), dtype=dist_x.dtype, device=dist_x.device))

    if estimator == "ksg2":
        # Padded neighbors have radius inf and are masked below; clamp their indices so the gather stays in range.
        index = nearest.indices.clamp(max=length - 1)
        per_point = torch.zeros_like(radius)
        for dist in (dist_x, dist_y):
            extent = dist.gather(-1, index).amax(dim=-1, keepdim=True)
            # <= includes self, which the paper's n excludes.
            per_point = per_point + torch.digamma((dist <= extent).sum(dim=-1).sub(1).to(dist.dtype))
        valid = torch.isfinite(radius) & ~invalid_point
        per_point = torch.where(valid, per_point, torch.full_like(per_point, float("nan")))
        mi = digamma_k - 1.0 / num_neighbors + digamma_n - per_point.mean(dim=-1)
        return _restore_batch(mi, x_ndim)

    # Include self unconditionally for n_x + 1 and n_y + 1, also when duplicates make the radius zero.
    self_mask = torch.eye(length, dtype=torch.bool, device=dist_x.device)
    count_x = ((dist_x < radius.unsqueeze(-1)) | self_mask).sum(dim=-1).to(dist_x.dtype)
    count_y = ((dist_y < radius.unsqueeze(-1)) | self_mask).sum(dim=-1).to(dist_y.dtype)
    per_point = torch.digamma(count_x) + torch.digamma(count_y)
    # Mask into NaN instead of raising, so vmap's BatchedTensor inputs work.
    valid = torch.isfinite(radius) & ~invalid_point
    per_point = torch.where(valid, per_point, torch.full_like(per_point, float("nan")))
    mi = digamma_k + digamma_n - per_point.mean(dim=-1)
    return _restore_batch(mi, x_ndim)
