"""Convergent cross mapping (CCM), from observed data.

Reference: Sugihara, G., May, R., Ye, H., Hsieh, C., Deyle, E., Fogarty,
M. & Munch, S. (2012), "Detecting Causality in Complex Ecosystems",
Science 338(6106), 496-500.

CCM tests whether one variable causally influences another using only
observed time series, no intervention. The intuition (and the classic
source of confusion about which direction the test actually points in):
if a driving variable causally influences a response variable, the
driver's dynamical signature gets left behind in the response's own
trajectory. So the response's own delay-embedded shadow manifold can be
used to *cross-map* (locally, nearest-neighbor-weighted predict) the
driver's values -- and under this simple unidirectional-forcing story,
the more of the response's history is available (the larger the
"library"), the better that cross-map prediction gets ("convergence"),
while a variable that does *not* drive the response leaves no such
signature and skill stays flat as the library grows.

Neither function in this module performs that convergence check itself
-- each call computes one estimate/skill value at one library size (see
:func:`~torchnlts.causality.ccm_skill`'s own docstring: the causality read is in how skill
changes *across several calls* at different ``library_size`` values, not
in any single returned number). Nor is "high/increasing skill" a
guarantee of causation on its own even across a full convergence sweep:
the paper's own follow-up literature documents that variables driven by
a shared external forcing, or synchronized through sufficiently strong
*bidirectional* coupling, can also show convergent cross-map skill in
both directions without one causing the other in the sense CCM is meant
to detect -- convergence is evidence consistent with the causal-forcing
story under CCM's assumptions, not proof of it, and ruling out those
confounds is the caller's responsibility, not something this module
checks.

This module keeps the direction unambiguous in its own terms rather than
in "X"/"Y causes" language: :func:`~torchnlts.causality.cross_map_estimate` takes
``manifold_series`` (embedded and searched for neighbors) and
``estimated_series`` (whose values are cross-map-predicted from those
neighbors' own indices). Under the assumptions above, high/increasing
skill (see :func:`~torchnlts.causality.ccm_skill`) when embedding ``manifold_series`` and
predicting ``estimated_series`` across a library-size sweep is evidence
that ``estimated_series`` drives ``manifold_series`` -- the
reversed-looking direction the paper itself notes trips people up.

Neighbor search and weighting matches CausalityTools.jl's
``ConvergentCrossMapping`` (its generic, no-estimator ``crossmap``/
``predict`` path, i.e. always using the full available library -- see
this module's ``library_size`` for the partial-library case it doesn't
cover): for each embedded point, find its ``embedding_dim + 1`` nearest
neighbors (Theiler-windowed) within the library, weight them by
``w_i = exp(-d_i / d_1) / sum_j exp(-d_j / d_1)`` (``d_1`` the smallest
neighbor distance), and combine ``estimated_series``'s own values at
those neighbors' time indices with those weights. Confirmed, by reading
its source, that Julia's implementation adds a tiny epsilon to every
distance whenever the nearest neighbor's distance is exactly ``0``
(otherwise ``d_i / d_1`` is a literal ``0/0`` for that neighbor) --
matched here rather than left to propagate as ``NaN``, unlike this
package's usual "don't silently clamp a genuine degeneracy" convention
elsewhere (Kantz/Rosenstein's unclamped ``log(0)``): ``0/0`` here isn't
a meaningful mathematical answer being hidden, it's a genuinely
undefined expression every real implementation of this formula must
disambiguate somehow, and this is the reference implementation's own
disambiguation.

Like :mod:`torchnlts.recurrence.joint_recurrence`, ``manifold_series``
and ``estimated_series`` need not share a feature space (only a time
axis to align by), and are truncated to the shorter one's length if they
differ.

Confirmed, by direct comparison against ``CausalityTools.jl`` on a
synthetic unidirectionally-coupled logistic-map pair (see this module's
tests), to match its ``ConvergentCrossMapping`` to float64 precision
(max per-point difference ~1e-16) for the default full-library case --
but only once its own ``w`` parameter is translated as
``theiler_window = w + 1``, not ``theiler_window = w`` directly. That
offset is a confirmed discrepancy in ``CausalityTools.jl`` itself, not a
convention chosen here: its own docstring says ``w = 0`` excludes only
the query point itself, but its implementation passes ``Theiler(w + 1)``
to ``Neighborhood.jl``, whose own documented semantics are
``|i - j| <= w`` excluded -- so ``w = 0`` there actually excludes
``|i - j| <= 1`` (self *and* its immediate temporal neighbors), not only
self as the docstring claims. This module's own ``theiler_window``
matches this package's own established convention throughout (see
:func:`torchnlts.neighbors.theiler_window_mask`:
``|i - j| <= theiler_window`` excluded, ``theiler_window = 0`` excludes
only self, exactly as documented) rather than replicating Julia's
inconsistency.
"""

from __future__ import annotations

import operator

import torch

from torchnlts._shape import _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, theiler_window_mask


def cross_map_estimate(
    manifold_series,
    estimated_series,
    embedding_dim: int,
    delay: int = 1,
    library_size: int | None = None,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Cross-map estimate of ``estimated_series`` from ``manifold_series``'s
    delay-embedded shadow manifold.

    Gradient class: piecewise (built on nearest-neighbor selection, same
    caveat as :func:`torchnlts.neighbors.knn_search`: the weighted
    combination differentiates normally once neighbors are picked, but
    the pick itself is a discrete, non-differentiable decision).

    Args:
        manifold_series: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame. Delay-embedded
            and searched for neighbors.
        estimated_series: Second series, same batched-ness as
            ``manifold_series`` (see :func:`torchnlts._shape._validate_matching_batch`),
            any feature dimensionality of its own. Truncated together with
            ``manifold_series`` to ``min(len(manifold_series), len(estimated_series))``
            if they differ in length.
        embedding_dim: Delay-embedding dimension of ``manifold_series``
            (Sugihara et al.'s ``E``). Uses ``embedding_dim + 1`` nearest
            neighbors per point, per the cited paper.
        delay: Embedding lag.
        library_size: Number of leading embedded points to search for
            neighbors within (a prefix of the manifold, not a random
            subsample -- see the module docstring). ``None`` (default)
            uses every embedded point. Must be in ``[1, N]`` where
            ``N = len(manifold_series) - (embedding_dim - 1) * delay``
            after truncation.
        theiler_window: Theiler window over embedded-point start indices
            (see :func:`torchnlts.neighbors.theiler_window_mask`).
            Must be ``>= 0`` (self-matches must be excluded
            unconditionally, matching Sugihara et al.'s own convention
            that a point is never its own neighbor).
        norm_order: Minkowski norm order for the neighbor search.

    Returns:
        Cross-map estimate of ``estimated_series``, shape ``[N, D]`` or
        ``[B, N, D]`` (``D`` from ``estimated_series``; unlike
        :func:`torchnlts.embedding.delay_embedding`'s own
        ``[T] -> [N, E]``-style restoration, an unbatched ``[T]`` input
        for either series still yields a 2D ``[N, D]`` result, not a
        flattened ``[N]`` one -- ``D`` is genuinely ``estimated_series``'s
        own feature dimensionality here, which may not be 1). Row ``n``
        estimates ``estimated_series`` at the same raw time index as
        embedded row ``n``'s *most recent* sample, i.e. raw index
        ``n + (embedding_dim - 1) * delay`` (matching Sugihara et al.'s
        "current + past" alignment, not this codebase's own
        ``delay_embedding`` row-start convention, which is why this
        function re-derives the alignment itself rather than assuming
        row ``n`` of ``estimated_series`` directly). ``NaN`` at any row
        with fewer than ``embedding_dim + 1`` valid (Theiler-window-
        surviving, library-restricted) neighbor candidates.

    Raises:
        ValueError: If ``manifold_series``/``estimated_series`` aren't
            both batched or both unbatched, or have mismatched batch
            size; if ``embedding_dim < 1``; if ``library_size`` is out
            of ``[1, N]``; or if ``theiler_window < 0``.
    """
    embedding_dim = operator.index(embedding_dim)
    delay = operator.index(delay)
    theiler_window = operator.index(theiler_window)
    _require_self_exclusion(theiler_window, "cross-map neighbor search")

    manifold_batched, manifold_ndim = _ensure_batched(manifold_series)
    estimated_batched, estimated_ndim = _ensure_batched(estimated_series)
    _validate_matching_batch(manifold_batched, estimated_batched, manifold_ndim, estimated_ndim, check_feature_dim=False)

    num_points = min(manifold_batched.shape[-2], estimated_batched.shape[-2])
    manifold_batched = manifold_batched[:, :num_points, :]
    estimated_batched = estimated_batched[:, :num_points, :]

    embedded = delay_embedding(manifold_batched, embedding_dim, delay)  # [B, N, E]
    num_embedded = embedded.shape[-2]
    current_time_offset = (embedding_dim - 1) * delay
    target = estimated_batched[:, current_time_offset : current_time_offset + num_embedded, :]  # [B, N, D]

    library_size = num_embedded if library_size is None else operator.index(library_size)
    if not (1 <= library_size <= num_embedded):
        raise ValueError(f"library_size must be in [1, {num_embedded}], got {library_size}")

    library = embedded[:, :library_size, :]
    distance = pairwise_distance(embedded, other=library, norm_order=norm_order)  # [B, N, L]
    mask = theiler_window_mask(num_embedded, theiler_window, device=embedded.device)[:, :library_size]
    distance = torch.where(mask, distance, torch.full_like(distance, float("inf")))

    num_neighbors = embedding_dim + 1
    if num_neighbors > library_size:
        distance = torch.nn.functional.pad(distance, (0, num_neighbors - library_size), value=float("inf"))
    neighbor_distance, neighbor_index = torch.topk(distance, num_neighbors, dim=-1, largest=False, sorted=True)
    # The farthest of the requested num_neighbors slots is still `inf`
    # exactly when fewer than num_neighbors valid (Theiler/library-
    # surviving) candidates existed for that row -- catch it before the
    # eps nudge below turns a *different* degenerate case's inf into
    # something finite-looking downstream.
    row_valid = torch.isfinite(neighbor_distance[..., -1])

    # Match CausalityTools.jl: a literal 0/0 in the weight formula below
    # (when the nearest neighbor is at distance exactly 0) is resolved by
    # nudging every distance in that row away from 0, not left as NaN --
    # see the module docstring for why this one case departs from this
    # package's usual "let a genuine degeneracy propagate" convention.
    nearest_is_zero = neighbor_distance[..., :1] <= 0
    epsilon = torch.finfo(neighbor_distance.dtype).eps * 10
    neighbor_distance = torch.where(nearest_is_zero, neighbor_distance + epsilon, neighbor_distance)

    weight = torch.exp(-neighbor_distance / neighbor_distance[..., :1])
    weight = weight / weight.sum(dim=-1, keepdim=True)

    safe_index = neighbor_index.clamp(min=0, max=num_embedded - 1)
    batch_size, num_query, num_target_features = target.shape
    flat_index = safe_index.reshape(batch_size, num_query * num_neighbors, 1).expand(-1, -1, num_target_features)
    gathered = torch.gather(target, 1, flat_index).reshape(batch_size, num_query, num_neighbors, num_target_features)

    estimate = (weight.unsqueeze(-1) * gathered).sum(dim=2)  # [B, N, D]
    estimate = torch.where(row_valid.unsqueeze(-1), estimate, torch.full_like(estimate, float("nan")))
    return _restore_batch(estimate, max(manifold_ndim, estimated_ndim))


def ccm_skill(estimate: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between a cross-map estimate and the actual
    values it targets -- the CCM causality signal itself (Sugihara et
    al. 2012): compute this once per ``library_size`` and look for it
    increasing ("converging") as the library grows to conclude causal
    influence, rather than reading any single value in isolation.

    Gradient class: exact.

    Args:
        estimate: Shape ``[N]``, ``[N, D]``, or ``[B, N, D]``, as
            returned by :func:`~torchnlts.causality.cross_map_estimate` (or any array of the
            same convention).
        actual: The corresponding true values, exactly matching
            ``estimate``'s shape.

    Returns:
        Pearson correlation, shape ``[1]`` for an unbatched, single-feature
        ``estimate`` up to ``[B, D]`` for a batched multi-feature one
        (``_ensure_batched`` always keeps a trailing feature dim, even
        for a bare ``[N]`` input, and only the leading batch dim gets
        squeezed back off -- not a true 0-dim scalar). ``NaN`` if
        ``estimate`` or ``actual`` has fewer
        than 2 points, or if either is constant (zero variance) along the
        time axis -- an exactly-zero denominator, not clamped.

    Raises:
        ValueError: If ``estimate`` and ``actual`` don't have identical
            shapes.
    """
    if estimate.shape != actual.shape:
        raise ValueError(f"expected estimate and actual to have matching shapes, got {tuple(estimate.shape)} and {tuple(actual.shape)}")
    estimate_batched, ndim = _ensure_batched(estimate)
    actual_batched, _ = _ensure_batched(actual)

    estimate_centered = estimate_batched - estimate_batched.mean(dim=-2, keepdim=True)
    actual_centered = actual_batched - actual_batched.mean(dim=-2, keepdim=True)
    numerator = (estimate_centered * actual_centered).sum(dim=-2)
    denominator = estimate_centered.norm(dim=-2) * actual_centered.norm(dim=-2)
    correlation = numerator / denominator

    return _restore_batch(correlation, ndim)
