"""Transfer entropy and the Frenzel-Pompe conditional mutual information estimator it is built on.

Reference: Schreiber, T. (2000), "Measuring Information Transfer", Phys. Rev. Lett. 85(2), 461-464:
``TE(source -> target) = I(target_future ; source_past | target_past)``, the information the source's past adds about
the target's next value beyond the target's own past. It is directional: ``transfer_entropy(source, target)`` and
``transfer_entropy(target, source)`` differ.

:func:`~torchnlts.causality.conditional_mutual_information` is the k-nearest-neighbor estimator of Frenzel & Pompe
(2007), Phys. Rev. Lett. 99, 204101, the three-variable analogue of :func:`torchnlts.entropy.mutual_information`
(KSG1). One k-th-neighbor radius from the joint (x, y, z) space is shared by the three marginal counts, so the
digamma terms cancel the bias; subtracting two ``mutual_information`` calls, each with its own radius, does not. As in
``mutual_information``, the Theiler window applies to the joint search only.

Differences from CausalityTools.jl (``FPVP`` conditional mutual information, ``TEShannon``):

- Marginal neighbors are counted with the paper's strict ``<``; ``FPVP`` counts ``<=`` (Neighborhood.jl's
  ``inrangecount``), which changes the estimate on essentially every point.
- :func:`~torchnlts.causality.transfer_entropy` embeds the source's and the target's past with one shared
  ``embedding_dim`` (CausalityTools.jl's default ``EmbeddingTE()`` has ``dS = dT = 1``) and does not offer its
  independent dimensions or non-uniform lags.

Limitation: a single shared ``embedding_dim``; independent source and target embedding dimensions are the upgrade
path.

Gradient class: none (neighbor ranks and threshold counts are discrete).
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _restore_batch, _validate_matching_batch
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, apply_theiler_window


def conditional_mutual_information(
    x,
    y,
    z,
    *,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Frenzel-Pompe k-nearest-neighbor estimate of the conditional mutual information ``I(x ; y | z)``, in nats.

    Reference: Frenzel & Pompe (2007), Phys. Rev. Lett. 99, 204101.

    Gradient class: none (neighbor ranks and threshold counts are discrete).

    Args:
        x: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        y: Same batching as ``x``, own feature dimension.
        z: Conditioning variable, same batching as ``x``, own feature dimension. The three are truncated to the
            shortest length.
        num_neighbors: ``k``, the rank of the joint-space neighbor whose distance is each point's counting radius.
        theiler_window: Joint-space neighbors with ``|i - j| <= theiler_window`` are excluded; the marginal counts
            are not windowed. Must be ``>= 0``.
        norm_order: Minkowski norm order of every marginal distance and, through them, the joint metric (``inf``,
            Chebyshev, is the paper's choice).

    Returns:
        Scalar for unbatched input, ``[B]`` for batched, in ``x``'s floating dtype (float32 for half-precision
        input, the default dtype for integer input). Can be negative: it is a finite-sample estimate. ``NaN`` for a
        batch element with a point that has fewer than ``num_neighbors`` joint neighbors left after Theiler
        exclusion, or with a ``NaN``/``inf`` in ``x``, ``y`` or ``z``.

    Raises:
        ValueError: If ``num_neighbors`` is not an integer ``>= 1``, ``theiler_window`` is negative or not an
            integer, or the inputs disagree on batching or batch size.
    """
    num_neighbors = _as_int(num_neighbors, "num_neighbors")
    if num_neighbors < 1:
        raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
    theiler_window = _require_self_exclusion(theiler_window, "conditional mutual information")

    x_batched, x_ndim = _ensure_batched(x)
    y_batched, y_ndim = _ensure_batched(y)
    z_batched, z_ndim = _ensure_batched(z)
    _validate_matching_batch(x_batched, y_batched, x_ndim, y_ndim, check_feature_dim=False)
    _validate_matching_batch(x_batched, z_batched, x_ndim, z_ndim, check_feature_dim=False)

    length = min(x_batched.shape[-2], y_batched.shape[-2], z_batched.shape[-2])
    x_batched, y_batched, z_batched = x_batched[..., :length, :], y_batched[..., :length, :], z_batched[..., :length, :]
    invalid_point = (
        ~torch.isfinite(x_batched).all(dim=-1)
        | ~torch.isfinite(y_batched).all(dim=-1)
        | ~torch.isfinite(z_batched).all(dim=-1)
    )

    dist_x = pairwise_distance(x_batched, norm_order=norm_order)
    dist_y = pairwise_distance(y_batched, norm_order=norm_order)
    dist_z = pairwise_distance(z_batched, norm_order=norm_order)
    joint_distance = apply_theiler_window(torch.maximum(torch.maximum(dist_x, dist_y), dist_z), theiler_window)
    if num_neighbors > length:
        joint_distance = torch.nn.functional.pad(joint_distance, (0, num_neighbors - length), value=float("inf"))
    radius = joint_distance.topk(num_neighbors, dim=-1, largest=False).values[..., -1]

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
    *,
    embedding_dim: int = 1,
    delay: int = 1,
    prediction_lag: int = 1,
    num_neighbors: int = 3,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Transfer entropy from ``source`` to ``target``, in nats.

    The conditional mutual information between the target ``prediction_lag`` steps ahead and the source's past, given
    the target's past.

    Reference: Schreiber (2000), Phys. Rev. Lett. 85(2), 461-464.

    Gradient class: none (see :func:`~torchnlts.causality.conditional_mutual_information`).

    Args:
        source: Shape ``[T]``, ``[T, D]`` or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
        target: Same batching as ``source``, own feature dimension. Truncated together with ``source`` to the shorter
            length.
        embedding_dim: Embedding dimension of the source's and of the target's past (one shared value).
        delay: Embedding lag of both pasts.
        prediction_lag: Steps from the most recent sample of the pasts to the predicted target value (``eta`` in
            Schreiber and CausalityTools.jl). Must be ``>= 1``.
        num_neighbors: Forwarded to :func:`~torchnlts.causality.conditional_mutual_information`.
        theiler_window: Forwarded to :func:`~torchnlts.causality.conditional_mutual_information`.
        norm_order: Forwarded to :func:`~torchnlts.causality.conditional_mutual_information`.

    Returns:
        Scalar for unbatched input, ``[B]`` for batched. ``NaN`` as for
        :func:`~torchnlts.causality.conditional_mutual_information`.

    Raises:
        ValueError: If ``embedding_dim``, ``delay``, ``prediction_lag`` or ``num_neighbors`` is not an integer
            ``>= 1``, ``theiler_window`` is negative or not an integer, the series are shorter than
            ``(embedding_dim - 1) * delay + prediction_lag + 1``, or the inputs disagree on batching or batch size.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    prediction_lag = _as_int(prediction_lag, "prediction_lag")
    if prediction_lag < 1:
        raise ValueError(f"prediction_lag must be >= 1, got {prediction_lag}")

    source_batched, source_ndim = _ensure_batched(source)
    target_batched, target_ndim = _ensure_batched(target)
    _validate_matching_batch(source_batched, target_batched, source_ndim, target_ndim, check_feature_dim=False)

    length = min(source_batched.shape[-2], target_batched.shape[-2])
    source_batched, target_batched = source_batched[:, :length, :], target_batched[:, :length, :]

    source_past = delay_embedding(source_batched, embedding_dim, delay=delay)  # [B, N, embedding_dim * D_source]
    target_past = delay_embedding(target_batched, embedding_dim, delay=delay)  # [B, N, embedding_dim * D_target]
    num_embedded = source_past.shape[-2]
    # Embedded row n ends at raw index n + offset; the target is predicted prediction_lag steps after that.
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

    # conditional_mutual_information sees already-batched inputs; restore the original batch rank here.
    cmi = conditional_mutual_information(
        target_future,
        source_past,
        target_past,
        num_neighbors=num_neighbors,
        theiler_window=theiler_window,
        norm_order=norm_order,
    )
    return _restore_batch(cmi, max(source_ndim, target_ndim))
