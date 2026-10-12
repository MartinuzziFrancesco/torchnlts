"""Lyapunov spectrum from a scalar time series (Eckmann et al. 1986).

Reference: Eckmann, J.-P., Kamphorst, S.O., Ruelle, D. & Ciliberto, S. (1986), "Liapunov exponents from time series",
Phys. Rev. A 34(6), 4971-4979. The series is delay-embedded in dimension ``d_E``; at every orbit point the tangent map
is fitted by least squares over the nearest neighbors, in the compressed ``d_M``-dimensional "matrix dimension" form
(companion matrix with last row ``a``, coordinates ``m`` steps apart, ``d_E - 1 = (d_M - 1) * m``). The tangent maps are
multiplied along the orbit, re-orthonormalized by QR, and ``log |diag(R)|`` is averaged. Each fitted map advances the
orbit by ``m`` steps, so the product follows the points ``i, i + m, i + 2m, ...`` (Sec. 2c); ``m`` interleaved chains,
one per starting phase, use every orbit point.

Compared with ``nolds.lyap_e`` (``emb_dim``, ``matrix_dim``, ``min_nb``, ``min_tsep``), the neighbor selection (max-norm
of the full embedded vector, ``|i - j| <= theiler_window`` excluded), the intercept-free least-squares fit of
``x[j + d_M * m] - x[i + d_M * m]`` on the neighbor differences and the per-exponent average agree: for ``m = 1`` the
two match to float64 precision on well-conditioned fits. Exponents are per sample step (nolds: ``tau = 1``); divide by
the sampling interval for per-unit-time values. Deviations:

- ``m > 1``: nolds composes the maps at every consecutive orbit point (one step per map instead of ``m``), which does
  not follow the paper. This module follows the paper, so the two differ for ``d_E > d_M``.
- ``delay`` spaces the embedding coordinates (``delay = 1`` is nolds); a map advances ``m * delay`` samples and the
  exponents are divided by it.
- Ties: exactly ``num_neighbors`` nearest points are used, whereas nolds also keeps every point tied with the last
  (``distance <= r``). Under the max-norm such ties occur (windows overlapping at lag ``m * delay`` share coordinates),
  so the two can differ at those points.
- Degenerate rows are ``NaN`` where nolds raises or returns ``+inf``: a series too short, a point with fewer than
  ``num_neighbors`` admissible neighbors, a neighbor difference matrix of rank below ``matrix_dim`` (numerical rank,
  ``torch.linalg.matrix_rank`` default tolerance), or a zero or subnormal growth factor in the QR chain (a singular
  tangent map, or a product singular to rounding). Constant and exactly periodic series are therefore ``NaN``, not
  ``-inf``. nolds takes the minimum-norm solution of a rank-deficient fit instead.
- The ``d_M`` exponents come in the order QR produces them (generally descending). Only the first few are meaningful;
  with ``d_M`` above the system's dimension the others are spurious (paper, Sec. 4).

Limitation: the paper's adaptive neighborhood enlargement and minimum-rank requirement are not implemented (a fixed
neighbor count and rejection of rank-deficient neighborhoods instead); a radius-growing search is the upgrade path.

Limitation: the neighbor search builds the full ``[N, N]`` distance matrix; a tree or chunked search is the upgrade
path for long series.

Limitation: half and bfloat16 input is computed in float32 but its gradient is cast back to the input dtype; for
moderately ill-conditioned fits (condition ~1e3) the input gradient can exceed half's range and become ``inf``. Keep the
trainable computation upstream in float32 when gradients are needed.

Gradient class: piecewise. Neighbor selection is discrete; the least-squares fit and the QR chain are differentiable
while the neighbor differences have full rank and the tangent maps are nonsingular. Degenerate points get placeholder
operands before the linear algebra, so a ``NaN`` row has zero gradient and puts nothing into other rows or shared
parameters. Gradients through the QR chain inherit the exploding-variance risk of products of Jacobians (Metz et al.
2021) for long series.
"""

from __future__ import annotations

import torch

from torchnlts._shape import (
    _as_int,
    _ensure_batched,
    _mask_nonfinite_rows,
    _output_dtype,
    _require_univariate,
    _restore_batch,
    _work_dtype,
)
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import _validate_norm_order
from torchnlts.neighbors.knn import knn_search
from torchnlts.neighbors.theiler import _require_self_exclusion


def _least_squares(design: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Least-squares solution of the full-rank system ``design @ x = response``, ``[..., k, d]`` and ``[..., k]`` ->
    ``[..., d]``, through a reduced QR factorization: its backward stays accurate for ill-conditioned designs, where
    ``torch.linalg.lstsq``'s is off by orders of magnitude."""
    orthonormal, upper = torch.linalg.qr(design)
    projected = orthonormal.transpose(-2, -1) @ response.unsqueeze(-1)
    return torch.linalg.solve_triangular(upper, projected, upper=True).squeeze(-1)


def _qr_chain(tangent: torch.Tensor, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed ``log |diag R|`` ``[B, d_M]`` of the QR-reorthonormalized products of ``tangent`` ``[B, N, d_M, d_M]``,
    one chain per starting phase advancing ``shift`` points per map, and the ``[B]`` rows where a growth factor is zero
    or subnormal (QR's backward divides by it)."""
    batch_size, num_points, matrix_dim, _ = tangent.shape
    identity = torch.eye(matrix_dim, dtype=tangent.dtype, device=tangent.device)
    num_chains = min(shift, num_points)
    chain_start = torch.arange(num_chains, device=tangent.device)
    orthonormal = identity.expand(batch_size, num_chains, -1, -1)
    total = torch.zeros(batch_size, matrix_dim, dtype=tangent.dtype, device=tangent.device)
    degenerate = torch.zeros(batch_size, dtype=torch.bool, device=tangent.device)
    smallest_normal = torch.finfo(tangent.dtype).tiny
    for step in range(-(-num_points // shift)):
        point = chain_start + step * shift  # [C]
        in_range = point < num_points
        step_tangent = tangent[:, point.clamp(max=num_points - 1)]
        step_tangent = torch.where(in_range.view(1, -1, 1, 1), step_tangent, identity)
        orthonormal, upper = torch.linalg.qr(step_tangent @ orthonormal)
        growth = torch.diagonal(upper, dim1=-2, dim2=-1).abs()  # [B, C, d_M]
        usable = in_range.view(1, -1, 1) & (growth >= smallest_normal) & torch.isfinite(growth)
        degenerate = degenerate | (in_range.view(1, -1, 1) & ~usable).flatten(1).any(dim=-1)
        log_growth = torch.log(torch.where(usable, growth, torch.ones_like(growth)))
        total = total + log_growth.sum(dim=1)
    return total, degenerate


def eckmann_spectrum(
    trajectory,
    embedding_dim: int,
    matrix_dim: int,
    *,
    num_neighbors: int | None = None,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Lyapunov exponents of a scalar series, per sample step.

    Reference: Eckmann, Kamphorst, Ruelle & Ciliberto (1986), Phys. Rev. A 34(6), 4971-4979.

    Gradient class: piecewise (see the module docstring).

    Args:
        trajectory: Scalar series, shape ``[T]``, ``[T, 1]`` or ``[B, T, 1]``. A tensor, numpy array, or pandas
            Series/DataFrame.
        embedding_dim: Embedding dimension ``d_E``; ``embedding_dim - 1`` must be divisible by ``matrix_dim - 1``.
        matrix_dim: Matrix dimension ``d_M``, in ``[2, d_E]``; the number of exponents returned.
        num_neighbors: Neighbors per orbit point for the tangent-map fit, ``>= matrix_dim``. Default:
            ``min(2 * matrix_dim, matrix_dim + 4)``.
        delay: Sample spacing of the embedding coordinates, ``>= 1``.
        theiler_window: Neighbors with ``|i - j| <= theiler_window`` are excluded (see
            :func:`torchnlts.neighbors.theiler_window_mask`). Must be ``>= 0``.
        norm_order: Minkowski norm order of the neighbor search (default: the max-norm, as in nolds).

    Returns:
        Exponents of shape ``[matrix_dim]`` or ``[B, matrix_dim]``, in the input's floating dtype (the default dtype
        for integer input; half precision is computed in float32). ``NaN`` for a row containing NaN/inf and for a
        degenerate row (series too short, too few or rank-deficient neighborhoods, singular fitted map; constant and
        exactly periodic series).

    Raises:
        ValueError: If the trajectory is not scalar or is complex, ``embedding_dim``, ``matrix_dim``,
            ``num_neighbors``, ``delay`` or ``theiler_window`` is not an integer in its range, the dimensions are
            inconsistent, or ``norm_order`` is not a real number ``>= 0``.
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    matrix_dim = _as_int(matrix_dim, "matrix_dim")
    delay = _as_int(delay, "delay")
    if matrix_dim < 2:
        raise ValueError(f"matrix_dim must be >= 2, got {matrix_dim}")
    if embedding_dim < matrix_dim:
        raise ValueError(f"embedding_dim must be >= matrix_dim, got {embedding_dim} < {matrix_dim}")
    if (embedding_dim - 1) % (matrix_dim - 1) != 0:
        raise ValueError(f"embedding_dim - 1 must be divisible by matrix_dim - 1, got {embedding_dim} and {matrix_dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if num_neighbors is None:
        num_neighbors = min(2 * matrix_dim, matrix_dim + 4)
    num_neighbors = _as_int(num_neighbors, "num_neighbors")
    if num_neighbors < matrix_dim:
        raise ValueError(
            f"num_neighbors must be >= matrix_dim (a smaller neighborhood cannot determine the fit), got {num_neighbors} < {matrix_dim}"
        )
    theiler_window = _require_self_exclusion(theiler_window, "the Eckmann et al. Lyapunov spectrum")
    norm_order = _validate_norm_order(norm_order)

    batched, ndim = _ensure_batched(trajectory)
    _require_univariate(batched, "eckmann_spectrum")
    dtype = _output_dtype(batched)
    series, nonfinite = _mask_nonfinite_rows(batched.to(_work_dtype(dtype)))
    series = series[..., 0]
    batch_size, length = series.shape
    device = series.device

    matrix_step = (embedding_dim - 1) // (matrix_dim - 1)
    shift = matrix_step * delay
    num_points = length - (embedding_dim - 1) * delay - shift
    nan_result = torch.full((batch_size, matrix_dim), float("nan"), dtype=series.dtype, device=device)
    if num_points < 1:
        # Zero-weighted sum keeps the result connected to the input.
        result = nan_result + series.sum(dim=-1, keepdim=True) * 0
        return _restore_batch(result, ndim).to(dtype)

    embedded = delay_embedding(series[:, : length - shift].unsqueeze(-1), embedding_dim, delay=delay)
    _, neighbor_index = knn_search(embedded, num_neighbors, theiler_window=theiler_window, norm_order=norm_order)
    has_neighbors = (neighbor_index >= 0).all(dim=-1)  # [B, N]
    safe_index = torch.where(neighbor_index >= 0, neighbor_index, torch.zeros_like(neighbor_index))

    coordinates = embedded[..., ::matrix_step]  # [B, N, d_M]
    target = series[:, matrix_dim * shift : matrix_dim * shift + num_points]
    flat_index = safe_index.reshape(batch_size, -1)
    neighbor_coordinates = torch.take_along_dim(coordinates, flat_index.unsqueeze(-1), dim=1).reshape(
        batch_size, num_points, num_neighbors, matrix_dim
    )
    neighbor_target = torch.take_along_dim(target, flat_index, dim=1).reshape(batch_size, num_points, num_neighbors)
    design = neighbor_coordinates - coordinates.unsqueeze(2)
    response = neighbor_target - target.unsqueeze(2)

    # Degenerate points get placeholder operands so no singular system reaches QR.
    with torch.no_grad():
        full_rank = (
            torch.linalg.matrix_rank(
                torch.where(has_neighbors.unsqueeze(-1).unsqueeze(-1), design, torch.zeros_like(design))
            )
            == matrix_dim
        )
    fit_ok = has_neighbors & full_rank
    placeholder = torch.eye(num_neighbors, matrix_dim, dtype=series.dtype, device=device)
    design = torch.where(fit_ok.unsqueeze(-1).unsqueeze(-1), design, placeholder)
    response = torch.where(fit_ok.unsqueeze(-1), response, torch.zeros_like(response))
    coefficients = _least_squares(design, response)  # [B, N, d_M]

    with torch.no_grad():
        tangent_ok = fit_ok & torch.isfinite(coefficients).all(dim=-1)
    identity = torch.eye(matrix_dim, dtype=series.dtype, device=device)
    tangent = torch.cat(
        [identity[1:].expand(batch_size, num_points, -1, -1), coefficients.unsqueeze(-2)], dim=-2
    )  # [B, N, d_M, d_M]

    # An exactly singular chain product makes QR's backward NaN; a first pass without autograd finds such rows,
    # and they and rows with a rejected tangent map run on identity maps and are NaN.
    with torch.no_grad():
        _, chain_degenerate = _qr_chain(tangent.detach(), shift)
    bad = nonfinite | ~tangent_ok.all(dim=-1) | chain_degenerate
    tangent = torch.where(bad.view(-1, 1, 1, 1), identity, tangent)
    total, degenerate = _qr_chain(tangent, shift)
    exponents = total / (num_points * shift)
    bad = bad | degenerate
    exponents = torch.where(bad.unsqueeze(-1), nan_result, exponents)
    return _restore_batch(exponents, ndim).to(dtype)
