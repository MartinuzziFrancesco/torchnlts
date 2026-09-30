"""Transfer entropy, from observed data.

Reference: Schreiber, T. (2000), "Measuring Information Transfer", Phys.
Rev. Lett. 85(2), 461-464: transfer entropy from a source series to a
target series is the conditional mutual information between the
target's *future* and the source's *past*, conditioned on the target's
own past --

    TE(source -> target) = I(target_future ; source_past | target_past)

-- i.e. how much *additional* predictive information about the target's
next value the source's history carries, beyond what the target's own
history already provides. Unlike (unconditional) mutual information,
this is directional: ``transfer_entropy(source, target, ...)`` and
``transfer_entropy(target, source, ...)`` generally differ, and neither
equals ``mutual_information(source, target)``.

:func:`conditional_mutual_information` implements the K-nearest-neighbor
conditional mutual information estimator of Frenzel, S. & Pompe, B.
(2007), "Partial Mutual Information for Coupling Analysis of
Multivariate Time Series", Phys. Rev. Lett. 99, 204101 -- the natural,
consistent generalization of this codebase's own
:func:`torchnlts.entropy.mutual_information.mutual_information` (KSG1)
from two variables to three with conditioning, reusing the same
building blocks (:func:`torchnlts.neighbors.distance.pairwise_distance`,
the same strict marginal-counting convention, the same Theiler-window-
on-the-joint-search-only convention). Critically, this is *not* the
same as calling ``mutual_information`` twice and subtracting: that naive
decomposition (``I(A;B|C) = I(A;(B,C)) - I(A;C)``) uses a different,
independently-computed k-th-neighbor radius for each term, which breaks
the digamma bias-cancellation that makes the KSG-style estimator
accurate in the first place -- Frenzel & Pompe's estimator instead uses
a *single* k-th-neighbor radius, from the full three-way joint space,
shared across all three marginal counts.

Matches CausalityTools.jl's ``FPVP`` conditional-mutual-information
estimator (its own transfer entropy, ``TEShannon``, is defined as this
same conditional mutual information, matching the decomposition above
exactly). Known, confirmed deviation, same playbook as
:func:`torchnlts.entropy.mutual_information.mutual_information`'s own
CausalityTools.jl comparison: that package's ``FPVP`` counts marginal
neighbors with an *inclusive* (``<=``) radius query
(``Neighborhood.jl``'s ``inrangecount``, the same primitive
``mutual_information``'s own Julia comparison already found to be
inclusive there), not Frenzel & Pompe's strict ``<``. This module
implements the paper's strict definition.

:func:`transfer_entropy` uses a single shared ``embedding_dim`` for
*both* the source's and the target's own past (matching
CausalityTools.jl's own default, ``EmbeddingTE()``'s ``dS = dT = 1``) --
a deliberate simplification relative to Julia's fully general
``EmbeddingTE``, which supports independent source/target embedding
dimensions and arbitrary (non-uniform-lag) embedding vectors. Symmetric
dimensions keep the two embeddings' raw-time alignment identical (see
:func:`torchnlts.causality.ccm.cross_map_estimate`'s own docstring for
why this codebase's forward-offset ``delay_embedding`` needs this
re-derivation at all); independent dimensions would need two different
offsets reconciled against a shared "current time", which this module
doesn't implement. Limitation: single shared ``embedding_dim``; independent
source/target embedding dimensions would be the upgrade if asymmetric
embeddings matter for a real workload.
"""

from __future__ import annotations

import operator

import torch

from torchnlts._shape import _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, apply_theiler_window


def conditional_mutual_information(
    x,
    y,
    z,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Frenzel-Pompe KSG-style conditional mutual information estimate,
    ``I(x ; y | z)``.

    Gradient class: none (same reasoning as
    :func:`torchnlts.entropy.mutual_information.mutual_information`:
    k-th-nearest-neighbor selection and the marginal threshold counts
    are discrete).

    Args:
        x: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy
            array, or pandas Series/DataFrame. A ``NaN`` anywhere in a
            given time point NaNs that whole point's contribution (see
            Returns) rather than raising, matching
            ``mutual_information``'s own vmap-compatible convention.
        y: Second variable, same batched-ness as ``x``; own feature
            dimensionality.
        z: Conditioning variable, same batched-ness as ``x``; own
            feature dimensionality. All three are truncated together to
            the shortest one's length if they differ.
        num_neighbors: k, the neighbor rank whose joint-(x,y,z)-space
            distance sets each point's marginal counting radius.
        theiler_window: Theiler window, applied only when searching for
            each point's k-th joint-space neighbor, not to the marginal
            counting step (matching ``mutual_information``'s and
            CausalityTools.jl's ``FPVP``'s own convention). Must be
            ``>= 0``.
        norm_order: Shared Minkowski norm order for every marginal
            distance matrix and, through them, the joint metric
            (``inf`` for the paper's and CausalityTools.jl's default
            Chebyshev choice).

    Returns:
        Scalar (unbatched inputs) or ``[B]`` (batched) conditional
        mutual information estimate, in nats. Can be negative -- a
        finite-sample estimator, not a probability. ``NaN`` for any
        batch element where some point has fewer than ``num_neighbors``
        eligible neighbors in the joint space after Theiler exclusion,
        or where some point contains ``NaN`` in ``x``, ``y``, or ``z``.

    Raises:
        ValueError: If ``num_neighbors < 1``, ``theiler_window < 0``, or
            ``x``/``y``/``z`` disagree on batched-ness or batch size.
    """
    if num_neighbors < 1:
        raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
    _require_self_exclusion(theiler_window, "conditional mutual information")

    x_batched, x_ndim = _ensure_batched(x)
    y_batched, y_ndim = _ensure_batched(y)
    z_batched, z_ndim = _ensure_batched(z)
    _validate_matching_batch(x_batched, y_batched, x_ndim, y_ndim, check_feature_dim=False)
    _validate_matching_batch(x_batched, z_batched, x_ndim, z_ndim, check_feature_dim=False)

    length = min(x_batched.shape[-2], y_batched.shape[-2], z_batched.shape[-2])
    x_batched, y_batched, z_batched = x_batched[..., :length, :], y_batched[..., :length, :], z_batched[..., :length, :]
    invalid_point = torch.isnan(x_batched).any(dim=-1) | torch.isnan(y_batched).any(dim=-1) | torch.isnan(z_batched).any(dim=-1)

    dist_x = pairwise_distance(x_batched, norm_order=norm_order)
    dist_y = pairwise_distance(y_batched, norm_order=norm_order)
    dist_z = pairwise_distance(z_batched, norm_order=norm_order)
    joint_distance = apply_theiler_window(torch.maximum(torch.maximum(dist_x, dist_y), dist_z), theiler_window)
    if num_neighbors > length:
        joint_distance = torch.nn.functional.pad(joint_distance, (0, num_neighbors - length), value=float("inf"))
    radius = joint_distance.topk(num_neighbors, dim=-1, largest=False).values[..., -1]

    # Include self unconditionally to implement n_{xz} + 1 etc., same
    # reasoning as mutual_information's own count_x/count_y.
    self_mask = torch.eye(length, dtype=torch.bool, device=dist_x.device)
    radius_expanded = radius.unsqueeze(-1)
    count_xz = ((torch.maximum(dist_x, dist_z) < radius_expanded) | self_mask).sum(dim=-1).to(dist_x.dtype)
    count_yz = ((torch.maximum(dist_y, dist_z) < radius_expanded) | self_mask).sum(dim=-1).to(dist_x.dtype)
    count_z = ((dist_z < radius_expanded) | self_mask).sum(dim=-1).to(dist_x.dtype)
    per_point = torch.digamma(count_xz) + torch.digamma(count_yz) - torch.digamma(count_z)

    valid = torch.isfinite(radius) & ~invalid_point
    per_point = torch.where(valid, per_point, torch.full_like(per_point, float("nan")))
    digamma_k = torch.digamma(torch.tensor(float(num_neighbors), dtype=dist_x.dtype, device=dist_x.device))
    cmi = digamma_k - per_point.mean(dim=-1)
    return _restore_batch(cmi, x_ndim)


def transfer_entropy(
    source,
    target,
    embedding_dim: int = 1,
    delay: int = 1,
    prediction_lag: int = 1,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Transfer entropy from ``source`` to ``target``.

    Gradient class: none (see :func:`conditional_mutual_information`).

    Args:
        source: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor,
            numpy array, or pandas Series/DataFrame.
        target: Second series, same batched-ness as ``source``; own
            feature dimensionality. Truncated together with ``source``
            to the shorter one's length if they differ.
        embedding_dim: Delay-embedding dimension shared by both the
            source's and the target's own past (see the module
            docstring for why this is a single shared value, not
            independently configurable).
        delay: Embedding lag, shared by both past embeddings.
        prediction_lag: How many steps ahead the target is predicted
            from (Schreiber's/CausalityTools.jl's ``eta``). Must be
            ``>= 1``.
        num_neighbors: Forwarded to :func:`conditional_mutual_information`.
        theiler_window: Forwarded to :func:`conditional_mutual_information`.
        norm_order: Forwarded to :func:`conditional_mutual_information`.

    Returns:
        Scalar (unbatched) or ``[B]`` (batched) transfer entropy
        estimate, in nats. See :func:`conditional_mutual_information`
        for when this is ``NaN``.

    Raises:
        ValueError: If ``embedding_dim < 1``, ``delay < 1``, or
            ``prediction_lag < 1``, or if ``source``/``target`` disagree
            on batched-ness or batch size.
    """
    embedding_dim = operator.index(embedding_dim)
    delay = operator.index(delay)
    prediction_lag = operator.index(prediction_lag)
    if prediction_lag < 1:
        raise ValueError(f"prediction_lag must be >= 1, got {prediction_lag}")

    source_batched, source_ndim = _ensure_batched(source)
    target_batched, target_ndim = _ensure_batched(target)
    _validate_matching_batch(source_batched, target_batched, source_ndim, target_ndim, check_feature_dim=False)

    length = min(source_batched.shape[-2], target_batched.shape[-2])
    source_batched, target_batched = source_batched[:, :length, :], target_batched[:, :length, :]

    source_past = delay_embedding(source_batched, embedding_dim, delay)  # [B, N, embedding_dim * D_source]
    target_past = delay_embedding(target_batched, embedding_dim, delay)  # [B, N, embedding_dim * D_target]
    num_embedded = source_past.shape[-2]
    offset = (embedding_dim - 1) * delay

    num_points = num_embedded - prediction_lag
    if num_points < 1:
        raise ValueError(
            f"not enough points: embedding_dim={embedding_dim}, delay={delay}, prediction_lag={prediction_lag} "
            f"need trajectories of length >= {offset + prediction_lag + 1}, got {length}"
        )
    source_past = source_past[:, :num_points, :]
    target_past = target_past[:, :num_points, :]
    target_future = target_batched[:, offset + prediction_lag : offset + prediction_lag + num_points, :]

    # conditional_mutual_information can't tell these were originally
    # unbatched -- by the time it sees target_future/source_past/
    # target_past they're already forced into [B, T, D] form (sliced
    # from this function's own _ensure_batched calls above), so its own
    # internal restoration always treats them as genuinely batched.
    # Restore again here, against this function's own source/target ndim.
    cmi = conditional_mutual_information(target_future, source_past, target_past, num_neighbors, theiler_window, norm_order)
    return _restore_batch(cmi, max(source_ndim, target_ndim))
