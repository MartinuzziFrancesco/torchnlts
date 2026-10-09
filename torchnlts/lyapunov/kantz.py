"""Kantz's largest Lyapunov exponent estimator, from observed data.

Reference: Kantz, H. (1994), "A robust method to estimate the maximal
Lyapunov exponent of a time series", Phys. Lett. A 185(1), 77-87, eq. 1-2
(also Kantz & Schreiber, "Nonlinear Time Series Analysis", 2nd ed.,
Cambridge University Press, 2004, Ch. 5): delay-embed the trajectory,
find each point's full epsilon-neighborhood (excluding temporally
correlated points via a Theiler window, unlike
:func:`torchnlts.lyapunov.rosenstein_divergence`'s single
nearest neighbor), then track
S(k) = <ln( (1/|U_i|) * sum_{j in U_i} ||X_{i+k} - X_{j+k}|| )>_i as the
whole neighborhood is evolved forward k steps -- the *log of the mean*
neighborhood distance, averaged (in linear space) before the log
(Rosenstein's single-neighbor divergence has no such within-neighborhood
averaging). The slope of S(k) vs
k over a well-defined early-k region estimates the largest Lyapunov
exponent, via the same :func:`torchnlts.lyapunov.rosenstein_lyapunov`
OLS fit used for Rosenstein's curve (the fit itself doesn't care which
method produced the curve).

Confirmed, deliberate deviation from ChaosTools.jl's more general
``lyapunov_from_data`` at ``ntype=WithinRange(radius)``: reading its
source shows it takes the log of *each* neighbor's distance individually
before averaging (``E_m[j] = log(delay_distance(...))`` then
``E_n .+= E_m``, divided by the neighborhood size) -- the *mean of logs*,
not log of the mean. That's a real algorithmic choice by Parlitz's
generalization (which the function's own docstring says it implements,
not Kantz's literal 1994 formula), confirmed by hand-computing both
quantities for the same input and finding they numerically disagree
(Jensen's inequality: log-of-mean >= mean-of-logs in general, and they
differ here by more than floating-point noise) -- not a documentation
quirk to reconcile away. See the locked disagreement test in
``tests/test_kantz.py``.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _is_complex, _mask_nonfinite_rows
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import pairwise_distance
from torchnlts.neighbors.theiler import _require_self_exclusion, theiler_window_mask


def kantz_divergence(
    trajectory,
    k_values,
    radius: float,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = 2.0,
) -> torch.Tensor:
    """Kantz's average log-neighborhood-divergence curve S(k).

    Gradient class: piecewise (built on a hard epsilon-ball threshold for
    neighborhood membership: the evolved-point distances differentiate
    normally w.r.t. the trajectory once a neighborhood is fixed, but
    which points belong to it is a discrete, non-differentiable decision
    that can flip at an infinitesimal perturbation -- same caveat as
    :func:`torchnlts.lyapunov.rosenstein_divergence`).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        k_values: Evolution steps to evaluate S(k) at. Scalar or ``[K]``
            tensor to sweep.
        radius: Neighborhood radius (epsilon), inclusive (``<=``). Unlike
            ``k_values``, not swept in this function -- call it again
            with a different ``radius`` to check the estimate's
            sensitivity to that choice.
        embedding_dim: Delay-embedding dimension.
        delay: Embedding lag.
        theiler_window: Theiler window over embedded-point start indices
            (see :func:`torchnlts.neighbors.theiler_window_mask`).
            Must be ``>= 0`` (self-matches must be excluded
            unconditionally, or a point is always its own "neighbor" at
            distance 0).
        norm_order: Minkowski norm order for both the neighborhood search
            and the evolved-point divergence.

    Returns:
        S(k), shape ``[]``/``[K]`` for an unbatched trajectory or
        ``[B]``/``[K, B]`` for a batched one. ``NaN`` if no reference
        point has both a nonempty epsilon-neighborhood (after Theiler
        exclusion) and enough remaining trajectory length -- for itself
        and at least one of its neighbors -- to evolve
        ``max(k_values)`` steps forward. ``-inf`` at a ``k`` where *any*
        contributing reference point's mean neighborhood distance
        happens to be exactly ``0`` (e.g. its whole neighborhood is
        exact duplicate points) -- one such point suffices to make the
        whole average ``-inf``, matching ``log(0)`` rather than silently
        flooring it to a large finite value. ``NaN`` for any row containing
        NaN/inf.

    Raises:
        ValueError: If ``k_values`` is empty, has more than one
            dimension, or contains a negative, non-finite, or
            non-integer-valued entry, if ``k_values`` is complex,
            ``radius`` is not a real number ``>= 0``, ``embedding_dim``/``delay``
            are not integers, or ``theiler_window < 0``.
    """
    # See rosenstein_divergence for why float64 validation matters (a
    # plain Python float/list argument would otherwise silently lose
    # precision to torch.as_tensor's default dtype before validation).
    if _is_complex(k_values):
        raise ValueError(f"k_values must be real, got {k_values!r}")
    k_values_tensor = torch.as_tensor(k_values, dtype=torch.float64)
    if k_values_tensor.numel() == 0:
        raise ValueError("k_values must not be empty")
    if k_values_tensor.dim() > 1:
        raise ValueError(f"k_values must be scalar or 1D, got shape {tuple(k_values_tensor.shape)}")
    if torch.any(~torch.isfinite(k_values_tensor)):
        raise ValueError("k_values must be finite")
    if torch.any(k_values_tensor < 0):
        raise ValueError("k_values must be non-negative (evolution steps forward in time)")
    if torch.any(k_values_tensor != k_values_tensor.round()):
        raise ValueError("k_values must be integer-valued (they index discrete time steps)")
    theiler_window = _as_int(theiler_window, "theiler_window")
    _require_self_exclusion(theiler_window, "Kantz's Lyapunov estimator")
    try:
        radius_value = None if _is_complex(radius) or isinstance(radius, str) else float(radius)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        radius_value = None
    if radius_value is None or not radius_value >= 0:
        raise ValueError(f"radius must be a real number >= 0, got {radius!r}")

    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    delay = _as_int(delay, "delay")
    # Limitation: see rosenstein.py's identical workaround/comment -- both
    # delay_embedding and theiler_window_mask can raise OverflowError
    # converting an astronomically large Python int to a tensor scalar.
    embedding_delay = 1 if embedding_dim == 1 and delay >= 1 else delay
    # Rows containing NaN/inf are zeroed before embedding, so their evolved
    # distances cannot put NaN into backward, and are NaN in the result.
    raw, ndim = _ensure_batched(trajectory)
    raw, nonfinite = _mask_nonfinite_rows(raw)
    batched = delay_embedding(raw, embedding_dim, embedding_delay)
    num_points = batched.shape[-2]
    theiler_window = min(theiler_window, num_points)

    k_list = k_values_tensor.reshape(-1).tolist()
    max_k = int(max(k_list))

    # The evolved-point distance ||embedded[i+k] - embedded[j+k]|| is
    # just entry [i+k, j+k] of the SAME full pairwise distance matrix
    # computed below (which already holds every pair's distance, not
    # only the ones within radius) -- so a shifted slice of it, not a
    # fresh distance computation, gives the distance at any k. No
    # separate gather per k needed, unlike rosenstein_divergence's
    # single-neighbor-per-row indexing (there, a different neighbor is
    # selected per row, so a plain shifted slice can't express it).
    distance = pairwise_distance(batched, norm_order=norm_order)
    neighbor_mask = (distance <= radius_value) & theiler_window_mask(num_points, theiler_window, device=batched.device)

    if max_k >= num_points:
        num_usable = 0
    else:
        num_usable = num_points - max_k
    active_neighbor_mask = neighbor_mask[..., :num_usable, :num_usable]
    neighbor_count = active_neighbor_mask.sum(dim=-1)
    row_valid = neighbor_count > 0
    safe_neighbor_count = neighbor_count.clamp(min=1).to(distance.dtype)
    row_valid_count = row_valid.sum(dim=-1)

    curve = []
    for k in k_list:
        k = min(int(k), num_points)
        distance_at_k = distance[..., k : k + num_usable, k : k + num_usable]
        masked_distance = torch.where(active_neighbor_mask, distance_at_k, torch.zeros_like(distance_at_k))
        mean_distance = masked_distance.sum(dim=-1) / safe_neighbor_count
        # Same masked-branch gradient-poisoning fix as
        # rosenstein_divergence: substitute a safe dummy (1, whose
        # log-gradient 1/1 is finite) at rows with no neighbors, so
        # torch.where's *unselected* branch below never needs a
        # gradient through log(0)'s 1/0, even though its forward value
        # is discarded either way.
        # A valid row with zero mean distance keeps its -inf forward value
        # (see Returns) but also gets a safe operand, so log(0)'s infinite
        # gradient cannot reach a parameter shared with other rows.
        usable = row_valid & (mean_distance > 0)
        safe_mean_distance = torch.where(usable, mean_distance, torch.ones_like(mean_distance))
        log_mean_distance = torch.where(row_valid & ~usable, float("-inf"), torch.log(safe_mean_distance))
        masked_log = torch.where(row_valid, log_mean_distance, torch.zeros_like(log_mean_distance))
        curve.append(masked_log.sum(dim=-1) / row_valid_count)

    result = torch.stack(curve, dim=0) if k_values_tensor.dim() > 0 else curve[0]
    result = torch.where(nonfinite, float("nan"), result)
    # See rosenstein_divergence: a swept result is `[K, B]`, batch
    # trailing, so the axis to drop for an unbatched trajectory is the
    # last one, not dim 0 (`_restore_batch` assumes the latter).
    return result.squeeze(-1) if ndim != 3 else result
