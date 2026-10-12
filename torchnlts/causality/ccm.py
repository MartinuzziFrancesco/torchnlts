"""Convergent cross mapping (CCM), from observed data.

Reference: Sugihara, G., May, R., Ye, H., Hsieh, C., Deyle, E., Fogarty, M. & Munch, S. (2012), "Detecting
Causality in Complex Ecosystems", Science 338(6106), 496-500.

:func:`~torchnlts.causality.cross_map_estimate` embeds ``manifold_series`` and predicts ``estimated_series`` at each
embedded point from the values at its nearest neighbors' time indices; :func:`~torchnlts.causality.ccm_skill` is the
correlation of that estimate with the truth. Skill that rises ("converges") with the library size is evidence that
``estimated_series`` drives ``manifold_series``, because the driven variable's manifold carries the driver's
signature. One call is one library size: the convergence check is the caller's sweep over ``library_size``. Shared
external forcing and strong bidirectional coupling can also give convergent skill in both directions.

The weighting follows CausalityTools.jl's ``ConvergentCrossMapping``: the ``embedding_dim + 1`` nearest
Theiler-windowed library points get weights ``exp(-d_i / d_1) / sum_j exp(-d_j / d_1)``, ``d_1`` the smallest
distance. When ``d_1 = 0``, every distance is shifted by ``10 * eps`` of the working dtype (as in Julia) instead of
dividing 0 by 0. For the full library the estimate matches CausalityTools.jl to float64 precision, except that
Julia's Theiler parameter ``w`` excludes ``|i - j| <= w + 1`` (not only the point itself for ``w = 0``, as its
documentation says); here ``theiler_window`` excludes ``|i - j| <= theiler_window``, so pass ``w + 1`` to compare.

The two series need not share a feature space; they are truncated to the shorter length.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _output_dtype,
    _restore_shape_preserving,
    _validate_matching_batch,
    _work_dtype,
)
from torchnlts.convert import as_tensor
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, theiler_window_mask


def cross_map_estimate(
    manifold_series,
    estimated_series,
    embedding_dim: int,
    *,
    delay: int = 1,
    library_size: int | None = None,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Cross-map estimate of ``estimated_series`` from the delay-embedded manifold of ``manifold_series``.

    Reference: Sugihara et al. (2012), Science 338(6106), 496-500.

    Gradient class: piecewise (the neighbor choice is discrete; the weighted combination is smooth once it is fixed).

    Args:
        manifold_series: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas
            Series/DataFrame. Embedded and searched for neighbors.
        estimated_series: Same batching as ``manifold_series``, any feature dimension. Truncated together with
            ``manifold_series`` to the shorter length.
        embedding_dim: Embedding dimension ``E`` of ``manifold_series``; ``E + 1`` neighbors are used per point.
        delay: Embedding lag.
        library_size: Number of leading embedded points searched for neighbors, in ``[1, N]`` with
            ``N = T - (embedding_dim - 1) * delay``. Default: all.
        theiler_window: Embedded points with ``|i - j| <= theiler_window`` are not neighbors. Must be ``>= 0``, so
            a point is never its own neighbor.
        norm_order: Minkowski norm order of the neighbor search.

    Returns:
        Estimates with the rank of ``estimated_series``, shape ``[N]``, ``[N, D]`` or ``[B, N, D]``, in the promoted
        floating dtype of the inputs (the default dtype for integer input, float32 for half precision). Row ``n``
        estimates ``estimated_series`` at raw index ``n + (embedding_dim - 1) * delay``, the embedded point's most
        recent sample. ``NaN`` for a row with fewer than ``embedding_dim + 1`` neighbors left by the Theiler window and
        ``library_size``, or whose embedded point or neighbor targets are non-finite; other rows and their gradients
        are unaffected.

    Raises:
        ValueError: If a series is complex, the series are not both batched or both unbatched, their batch sizes
            differ, ``embedding_dim`` or ``delay`` is not a positive integer, ``theiler_window`` is negative or not an
            integer, or ``library_size`` is not an integer in ``[1, N]``.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    theiler_window = _require_self_exclusion(theiler_window, "cross-map neighbor search")

    manifold_batched, manifold_ndim = _ensure_batched(manifold_series)
    estimated_batched, estimated_ndim = _ensure_batched(estimated_series)
    estimated_batched = estimated_batched.to(_work_dtype(estimated_batched.dtype))
    _validate_matching_batch(
        manifold_batched, estimated_batched, manifold_ndim, estimated_ndim, check_feature_dim=False
    )

    num_points = min(manifold_batched.shape[-2], estimated_batched.shape[-2])
    manifold_batched = manifold_batched[:, :num_points, :]
    estimated_batched = estimated_batched[:, :num_points, :]

    embedded = delay_embedding(manifold_batched, embedding_dim, delay=delay)  # [B, N, E]
    num_embedded = embedded.shape[-2]
    current_time_offset = (embedding_dim - 1) * delay
    target = estimated_batched[:, current_time_offset : current_time_offset + num_embedded, :]  # [B, N, D]

    library_size = num_embedded if library_size is None else _as_int(library_size, "library_size")
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
    # A non-finite farthest slot means fewer than num_neighbors usable candidates.
    row_valid = torch.isfinite(neighbor_distance[..., -1])  # [B, N]
    # Placeholder distances: exp(-inf / d_1) has a NaN gradient w.r.t. d_1 (0 * inf).
    neighbor_distance = torch.where(row_valid.unsqueeze(-1), neighbor_distance, torch.ones_like(neighbor_distance))

    # Match CausalityTools.jl: shift rows with a zero nearest distance off 0 to avoid 0/0.
    nearest_is_zero = neighbor_distance[..., :1] <= 0
    epsilon = torch.finfo(neighbor_distance.dtype).eps * 10
    neighbor_distance = torch.where(nearest_is_zero, neighbor_distance + epsilon, neighbor_distance)

    weight = torch.exp(-neighbor_distance / neighbor_distance[..., :1])
    weight = weight / weight.sum(dim=-1, keepdim=True)

    safe_index = neighbor_index.clamp(min=0, max=num_embedded - 1)
    batch_size, num_query, num_target_features = target.shape
    flat_index = safe_index.reshape(batch_size, num_query * num_neighbors, 1).expand(-1, -1, num_target_features)
    gathered = torch.gather(target, 1, flat_index).reshape(batch_size, num_query, num_neighbors, num_target_features)
    # Zeroed so a non-finite target cannot reach backward as 0 * NaN; its row is NaN below.
    target_finite = torch.isfinite(gathered)
    gathered = torch.where(target_finite, gathered, torch.zeros_like(gathered))
    row_valid = row_valid & target_finite.all(dim=(-2, -1))

    estimate = (weight.unsqueeze(-1) * gathered).sum(dim=2)  # [B, N, D]
    estimate = torch.where(row_valid.unsqueeze(-1), estimate, torch.full_like(estimate, float("nan")))
    return _restore_shape_preserving(estimate, estimated_ndim)


def ccm_skill(estimate, actual) -> torch.Tensor:
    """Pearson correlation between a cross-map estimate and the values it targets.

    Compute it once per ``library_size``; skill that rises with the library size is the convergence CCM looks for.

    Reference: Sugihara et al. (2012), Science 338(6106), 496-500.

    Gradient class: exact.

    Args:
        estimate: Shape ``[N]``, ``[N, D]`` or ``[B, N, D]``, as returned by
            :func:`~torchnlts.causality.cross_map_estimate`. A tensor, numpy array, or pandas Series/DataFrame.
        actual: The true values, with the same shape as ``estimate``.

    Returns:
        The correlation over the time axis, shape ``[]``, ``[D]`` or ``[B, D]``, in the promoted floating dtype of
        the inputs (the default dtype for integer input). ``NaN`` for an entry whose estimate or actual has fewer
        than two points, is constant along time (zero variance, not clamped) or contains ``NaN``/``inf``; other
        entries and their gradients are unaffected.

    Raises:
        ValueError: If the shapes differ or an input is complex.
    """
    estimate, actual = as_tensor(estimate), as_tensor(actual)
    if estimate.shape != actual.shape:
        raise ValueError(
            f"expected estimate and actual to have matching shapes, got {tuple(estimate.shape)} and {tuple(actual.shape)}"
        )
    dtype = torch.promote_types(_output_dtype(estimate), _output_dtype(actual))
    estimate_batched, ndim = _ensure_batched(estimate.to(_work_dtype(dtype)))
    actual_batched, _ = _ensure_batched(actual.to(_work_dtype(dtype)))

    # Non-finite columns are zeroed and constant ones get a unit denominator, so neither reaches backward as NaN.
    finite = torch.isfinite(estimate_batched).all(dim=-2) & torch.isfinite(actual_batched).all(dim=-2)  # [B, D]
    estimate_batched = torch.where(finite.unsqueeze(-2), estimate_batched, torch.zeros_like(estimate_batched))
    actual_batched = torch.where(finite.unsqueeze(-2), actual_batched, torch.zeros_like(actual_batched))
    estimate_centered = estimate_batched - estimate_batched.mean(dim=-2, keepdim=True)
    actual_centered = actual_batched - actual_batched.mean(dim=-2, keepdim=True)
    estimate_norm = estimate_centered.norm(dim=-2)
    actual_norm = actual_centered.norm(dim=-2)
    valid = finite & (estimate_norm > 0) & (actual_norm > 0)
    numerator = (estimate_centered * actual_centered).sum(dim=-2)
    denominator = torch.where(valid, estimate_norm * actual_norm, torch.ones_like(numerator))
    skill = torch.where(valid, numerator / denominator, torch.full_like(numerator, float("nan")))
    return _restore_shape_preserving(skill, ndim).to(dtype)
