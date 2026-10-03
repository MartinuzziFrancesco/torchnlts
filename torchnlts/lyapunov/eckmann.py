"""Lyapunov spectrum from a scalar time series (Eckmann et al. 1986).

Reference: Eckmann, J.-P., Kamphorst, S.O., Ruelle, D. & Ciliberto, S.
(1986), "Liapunov exponents from time series", Phys. Rev. A 34(6),
4971-4979: delay-embed the series in dimension ``d_E``; at every orbit
point fit the tangent map by least squares over its nearest neighbors, in
the compressed ``d_M``-dimensional "matrix dimension" form (companion
matrix with last row ``a``, coordinates spaced ``m`` steps apart,
``d_E - 1 = (d_M - 1) * m``); multiply the successive tangent maps along
the orbit, re-orthonormalizing by QR (Benettin-style), and average
``log |diag(R)|``. Each fitted map advances the orbit by ``m`` steps, so
the product follows the orbit points ``i, i + m, i + 2m, ...`` (Sec. 2c);
``m`` interleaved chains, one per starting phase, use every orbit point.

Compared with ``nolds.lyap_e`` (``emb_dim``, ``matrix_dim``, ``min_nb``,
``min_tsep``): neighbor selection (max-norm of the full embedded vector,
``|i - j| <= theiler_window`` excluded), the intercept-free least-squares
fit of ``x[j + d_M * m] - x[i + d_M * m]`` on the neighbor differences,
and the per-exponent average agree, so for ``m = 1`` the two match to
float64 precision on well-conditioned fits. The exponents are per sample
step (nolds: ``tau=1``); divide by the sampling interval for
per-unit-time values. Deviations:

- ``m > 1``: nolds composes the maps at every consecutive orbit point
  (advancing one step per map instead of ``m``), which does not follow
  the paper; this module follows the paper, so the two differ for
  ``d_E > d_M`` (``m = 1`` is unaffected).
- ``delay`` generalizes the sample spacing of the embedding (``delay=1``
  is nolds); a map advances ``m * delay`` samples and exponents are
  divided by it.
- Ties: exactly ``num_neighbors`` nearest points are used. nolds also
  keeps every point tied with the last one (``distance <= r``). Under
  the max-norm such ties do occur (windows overlapping at lag ``m * delay``
  share coordinates), so the two can differ at those points.
- Degenerate rows are ``NaN`` where nolds raises or returns ``+inf``: a
  series too short, an orbit point with fewer than ``num_neighbors``
  admissible neighbors, a neighbor difference matrix of rank below
  ``matrix_dim`` (numerical rank, ``torch.linalg.matrix_rank`` default
  tolerance), a fitted tangent map with last-row first entry exactly
  zero (singular companion matrix), or a zero growth factor. Constant and
  exactly periodic series are therefore ``NaN``, not ``-inf``. nolds
  instead takes the minimum-norm solution of a rank-deficient fit.
  Because every accepted system has full rank, the least-squares driver
  (CPU ``gelsy``, CUDA ``gels``) does not matter.
- ``d_M`` exponents are returned in the order QR produces them (generally
  descending). Only the first few are meaningful; with ``d_M`` above the
  system's dimension the others are spurious (see the paper, Sec. 4).

Limitation: the paper's adaptive neighborhood enlargement and minimum-rank
requirement on the neighbor set are not implemented (a fixed neighbor
count and a rejection of rank-deficient neighborhoods instead); a
radius-growing search is the upgrade path.

Limitation: the neighbor search builds the full ``[N, N]`` distance
matrix; a tree or chunked search is the upgrade path for long series.

Gradient class: piecewise. Neighbor selection is discrete; the
least-squares fit and QR are differentiable while the neighbor
differences have full rank and the tangent maps are nonsingular (QR's
backward is undefined for singular input). Degeneracy is decided
without autograd before the linear algebra, and degenerate points get
placeholder operands, so a degenerate row returns ``NaN`` with zero
gradient and nothing reaches other rows or shared parameters. Gradients
through the chain of QR steps inherit the exploding-variance risk of
products of Jacobians (Metz et al. 2021) for long series.
"""

from __future__ import annotations

import operator

import torch

from torchnlts._shape import _ensure_batched, _mask_nonfinite_rows, _output_dtype, _work_dtype
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.distance import _validate_norm_order
from torchnlts.neighbors.knn import knn_search
from torchnlts.neighbors.theiler import _require_self_exclusion


def _as_int(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def eckmann_spectrum(
    trajectory,
    embedding_dim: int,
    matrix_dim: int,
    num_neighbors: int | None = None,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Lyapunov exponents of a scalar series (Eckmann, Kamphorst, Ruelle & Ciliberto).

    Gradient class: piecewise (see the module docstring).

    Args:
        trajectory: Scalar series, shape ``[T]``, ``[T, 1]``, or
            ``[B, T, 1]``. A tensor, numpy array, or pandas
            Series/DataFrame.
        embedding_dim: Embedding dimension ``d_E``. ``embedding_dim - 1``
            must be divisible by ``matrix_dim - 1``.
        matrix_dim: Matrix dimension ``d_M``, ``2 <= d_M <= d_E``; the
            number of exponents returned.
        num_neighbors: Neighbors per orbit point for the tangent-map fit,
            at least ``matrix_dim``. Defaults to
            ``min(2 * matrix_dim, matrix_dim + 4)``.
        delay: Sample spacing of the embedding coordinates.
        theiler_window: Neighbors with ``|i - j| <= theiler_window`` are
            excluded (see :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
        norm_order: Minkowski norm order of the neighbor search
            (default max-norm, as in nolds).

    Returns:
        Exponents per sample step, shape ``[matrix_dim]`` or
        ``[B, matrix_dim]``. ``NaN`` for a row containing NaN/inf and for
        any degenerate row (see the module docstring: too short, too few
        or rank-deficient neighborhoods, singular fitted map, constant or
        exactly periodic series).

    Raises:
        ValueError: If ``trajectory`` is not scalar or is complex, the
            dimensions are inconsistent, or any count is not an integer
            in range (``delay < 1``, ``num_neighbors < matrix_dim``,
            ``theiler_window < 0``, ``norm_order`` not a real number
            ``>= 0``).
    """
    embedding_dim = _as_int(embedding_dim, "embedding_dim")
    matrix_dim = _as_int(matrix_dim, "matrix_dim")
    delay = _as_int(delay, "delay")
    theiler_window = _as_int(theiler_window, "theiler_window")
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
        raise ValueError(f"num_neighbors must be >= matrix_dim (a smaller neighborhood cannot determine the fit), got {num_neighbors} < {matrix_dim}")
    _require_self_exclusion(theiler_window, "the Eckmann et al. Lyapunov spectrum")
    norm_order = _validate_norm_order(norm_order)

    batched, ndim = _ensure_batched(trajectory)
    if batched.shape[-1] != 1:
        raise ValueError(f"expected a scalar series [T], [T, 1], or [B, T, 1], got {tuple(batched.shape)}")
    dtype = _output_dtype(batched)
    series, nonfinite = _mask_nonfinite_rows(batched[..., 0].to(_work_dtype(dtype)).unsqueeze(-1))
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
        return (result.squeeze(0) if ndim != 3 else result).to(dtype)

    embedded = delay_embedding(series[:, : length - shift].unsqueeze(-1), embedding_dim, delay)
    _, neighbor_index = knn_search(embedded, num_neighbors, theiler_window=min(theiler_window, num_points), norm_order=norm_order)
    has_neighbors = (neighbor_index >= 0).all(dim=-1)  # [B, N]
    safe_index = torch.where(neighbor_index >= 0, neighbor_index, torch.zeros_like(neighbor_index))

    coordinates = embedded[..., ::matrix_step]  # [B, N, d_M]
    target = series[:, matrix_dim * shift : matrix_dim * shift + num_points]  # x[i + d_M * shift]
    flat_index = safe_index.reshape(batch_size, -1)
    neighbor_coordinates = torch.take_along_dim(coordinates, flat_index.unsqueeze(-1), dim=1).reshape(batch_size, num_points, num_neighbors, matrix_dim)
    neighbor_target = torch.take_along_dim(target, flat_index, dim=1).reshape(batch_size, num_points, num_neighbors)
    design = neighbor_coordinates - coordinates.unsqueeze(2)
    response = neighbor_target - target.unsqueeze(2)

    # Degeneracy is decided here, without autograd, and degenerate points
    # get placeholder operands, so no singular system reaches lstsq or QR.
    with torch.no_grad():
        full_rank = torch.linalg.matrix_rank(torch.where(has_neighbors.unsqueeze(-1).unsqueeze(-1), design, torch.zeros_like(design))) == matrix_dim
    fit_ok = has_neighbors & full_rank
    placeholder = torch.eye(num_neighbors, matrix_dim, dtype=series.dtype, device=device)
    design = torch.where(fit_ok.unsqueeze(-1).unsqueeze(-1), design, placeholder)
    response = torch.where(fit_ok.unsqueeze(-1), response, torch.zeros_like(response))
    coefficients = torch.linalg.lstsq(design, response.unsqueeze(-1)).solution.squeeze(-1)  # [B, N, d_M]

    # The companion matrix has determinant +-coefficients[..., 0].
    with torch.no_grad():
        tangent_ok = fit_ok & (coefficients[..., 0] != 0) & torch.isfinite(coefficients).all(dim=-1)
    identity = torch.eye(matrix_dim, dtype=series.dtype, device=device)
    coefficients = torch.where(tangent_ok.unsqueeze(-1), coefficients, identity[-1])
    tangent = torch.cat([identity[1:].expand(batch_size, num_points, -1, -1), coefficients.unsqueeze(-2)], dim=-2)  # [B, N, d_M, d_M]

    # One chain per starting phase, advancing `shift` orbit points per map.
    num_chains = min(shift, num_points)
    num_steps = -(-num_points // shift)
    chain_start = torch.arange(num_chains, device=device)
    orthonormal = identity.expand(batch_size, num_chains, -1, -1)
    total = torch.zeros(batch_size, matrix_dim, dtype=series.dtype, device=device)
    degenerate = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(num_steps):
        point = chain_start + step * shift  # [C]
        in_range = point < num_points
        step_tangent = tangent[:, point.clamp(max=num_points - 1)]
        step_tangent = torch.where(in_range.view(1, -1, 1, 1), step_tangent, identity)
        orthonormal, upper = torch.linalg.qr(step_tangent @ orthonormal)
        growth = torch.diagonal(upper, dim1=-2, dim2=-1).abs()  # [B, C, d_M]
        counted = in_range.view(1, -1, 1)
        usable = counted & (growth > 0) & torch.isfinite(growth)
        degenerate = degenerate | (counted & ~usable).flatten(1).any(dim=-1)
        log_growth = torch.log(torch.where(usable, growth, torch.ones_like(growth)))
        total = total + log_growth.sum(dim=1)
    exponents = total / (num_points * shift)
    bad = nonfinite | ~tangent_ok.all(dim=-1) | degenerate
    exponents = torch.where(bad.unsqueeze(-1), nan_result, exponents)
    return (exponents.squeeze(0) if ndim != 3 else exponents).to(dtype)
