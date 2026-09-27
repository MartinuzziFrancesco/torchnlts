"""Generalized (Rényi) dimension spectrum: order-q scaling of a
box-counting histogram's Rényi entropy with box size.

Reference: Russell, Hanson & Ott (1980), Phys. Rev. Lett. 45(14),
1175-1178, for the box-counting/histogram estimation method used here;
Hentschel & Procaccia (1983), Physica D 8(3), 435-444, for the D_q
spectrum's own general definition and its explicit identification of
order-2 D_q with the correlation dimension
(:func:`torchnlts.dimension.correlation.correlation_dimension`) in the
limit -- the same underlying quantity, estimated here by a genuinely
different method (see below), not merely "related". Partition the
state space into boxes of width ``box_size``,
take the occupation-probability distribution over occupied boxes, and
estimate the generalized dimension D_q as the scaling exponent of that
distribution's order-q Rényi entropy against ``-log(box_size)``:
``H_q(box_size) ~ D_q * -log(box_size)``. Matches FractalDimensions.jl's
``generalized_dim`` in method (box-counting + Rényi entropy + a
log-linear slope fit) -- confirmed directly against
``entropy(Renyi(;q,base=ℯ), ValueHistogram(ε), X)`` for several ``q``
and a multivariate point set, including the box-occupation
probabilities themselves, not just the final entropy -- but, like
:func:`torchnlts.dimension.correlation.correlation_dimension`, doesn't
replicate that package's own ``estimate_boxsizes`` heuristic (the
caller supplies ``box_size``) or its automatic
largest-linear-region detection (``slopefit`` with
``LargestLinearRegion``); this module fits every given point, the same
"caller picks the sizes, no automatic scaling-region detection"
convention ``correlation_dimension`` already uses.

Distinct from :func:`torchnlts.dimension.correlation.correlation_sum`'s
own generalization path: FractalDimensions.jl separately exposes a
*pairwise-distance*-based q-order correlation sum
(``correlationsum(X, εs; q)``, Grassberger 2007's definition, as
corrected in Kantz & Schreiber's "Nonlinear Time Series Analysis" Ch.
11.3) which is a genuinely different *estimator* of the same D_q
family (Hentschel & Procaccia's order-2 identification, above, is
between that estimator's own ``q=2`` and this module's ``order=2``,
not between two different theoretical quantities), restricted to
``q > 1`` (its own package explicitly
warns the correlation sum is ill-defined for ``q <= 1``, so it cannot
recover D0/D1 the way this box-counting route does) -- not implemented
here, since it doesn't naturally unify D0/D1/D2 the way this module's
approach does, which was the point of this addition.

Special cases (via :func:`torchnlts.entropy.generalized.renyi_entropy`,
reused directly rather than reimplemented): order 0 is "capacity"/
box-counting dimension (log of the occupied-box count), order 1 is
information dimension, order 2 is (a box-counting, not
pairwise-distance, estimate of) correlation dimension.

Box occupancy is computed by floor-dividing each point's own
per-dimension offset from the trajectory's own per-dimension minimum
by ``box_size`` (left-closed-right-open bins anchored at the data's own
minima, extended outward as needed to cover every point) -- matches
ComplexityMeasures.jl's ``RectangularBinning``/``RectangularBinEncoding``
convention exactly (confirmed by reading its source), not a fixed
external grid.

Gradient class: none. Box-index assignment (``floor``) and occupied-box
counting (``unique``) are both discrete, non-differentiable operations
with no natural continuous relaxation the way a distance threshold can
be kernel-softened -- same reasoning as
:func:`torchnlts.entropy.permutation.ordinal_pattern_probabilities`/
:func:`torchnlts.entropy.dispersion.dispersion_entropy`.

Not ``torch.vmap``-compatible, and not vectorized across a box-size
sweep or a batch either: the number of occupied boxes is inherently
data- and scale-dependent (finer boxes and larger point clouds occupy
more distinct boxes), so unlike this codebase's distance-threshold-based
measures, there's no shared fixed-size shape to vectorize against --
same reasoning, and the same "loop in Python" resolution, as
:func:`torchnlts.entropy.lempel_ziv.lempel_ziv_complexity`'s own
per-row combinatorial loop.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _ensure_batched, _ols_slope
from torchnlts.entropy.generalized import renyi_entropy


def box_occupation_probabilities(trajectory, box_size: float) -> torch.Tensor:
    """Relative frequency of occupied boxes in a fixed-width space
    partition of a trajectory's points.

    Gradient class: none (see module docstring).

    Args:
        trajectory: Shape ``[T]`` or ``[T, D]`` (unbatched only -- see
            module docstring for why a batched ``[B, T, D]`` input
            can't share this function's fixed-size output contract; use
            :func:`box_counting_entropy` for a batched trajectory).
        box_size: Box width, the same in every dimension. Must be
            positive (not validated, matching this codebase's general
            trust-the-caller convention for well-formed parameters).

    Returns:
        Probability distribution over the occupied boxes, shape
        ``[num_occupied]`` (data-dependent, not a fixed size), summing
        to 1. A single ``NaN`` entry (``[nan]``) if ``trajectory``
        contains any non-finite value (``NaN`` or ``inf``) -- box
        occupancy is undefined there. A ``NaN`` alone would already
        propagate through ``points.min()`` for the affected dimension;
        an ``inf`` doesn't (``min`` with an ``-inf`` entry returns
        ``-inf``, not ``NaN``), but combining it with a *different*
        point's own offset from that ``-inf`` minimum produces a mix
        of ``NaN`` (``-inf - (-inf)``) and ``inf`` (``finite - (-inf)``)
        that the overflow check below can't reliably catch either
        (``NaN`` comparisons are always ``False``) -- confirmed
        directly: without this explicit non-finite check up front, a
        trajectory containing ``-inf`` or ``inf`` alone both silently
        converted to a deterministic-looking but meaningless box
        coordinate via ``.to(torch.int64)`` instead of actually
        failing. The same ``[nan]`` result if the trajectory's own
        range divided by ``box_size`` would overflow ``int64`` (box
        indices are integer-valued internally) -- same failure class
        as this codebase's own ``rosenstein.py`` int64 overflow lesson
        (an absurdly small ``box_size`` relative to the data's range
        silently wrapping to a plausible-looking but meaningless box
        coordinate instead), guarded against proactively here. The comparison uses the exact power of
        two ``2**63`` (exactly representable in every float dtype used
        here), not ``torch.iinfo(torch.int64).max`` (``2**63 - 1``,
        which isn't exactly representable in ``float64`` and silently
        rounds *up* to ``2**63`` when compared against a float tensor
        -- confirmed directly: comparing against it let a scaled value
        of exactly ``2**63`` slip through the original ``>`` check
        into the ``int64`` cast).

    Raises:
        ValueError: If ``trajectory`` is batched (``[B, T, D]`` with
            ``B != 1``).
    """
    tensor, _ = _ensure_batched(trajectory)
    if tensor.shape[0] != 1:
        raise ValueError(
            f"box_occupation_probabilities only supports a single (unbatched) trajectory, got batch size {tensor.shape[0]}"
        )
    points = tensor[0]
    if not torch.isfinite(points).all():
        return torch.full((1,), float("nan"), dtype=points.dtype, device=points.device)
    minimum = points.min(dim=0, keepdim=True).values
    scaled = (points - minimum) / box_size
    if scaled.abs().max() >= 2**63:
        return torch.full((1,), float("nan"), dtype=points.dtype, device=points.device)
    box_index = torch.floor(scaled).to(torch.int64)
    _, counts = torch.unique(box_index, dim=0, return_counts=True)
    return counts.to(points.dtype) / counts.sum()


def box_counting_entropy(trajectory, box_size, order: float = 1.0) -> torch.Tensor:
    """Order-``order`` Rényi entropy of the box-occupation distribution,
    at one or more box sizes.

    Gradient class: none (see module docstring).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        box_size: Box width. Scalar or ``[S]`` tensor to sweep.
        order: Rényi order, forwarded to
            :func:`torchnlts.entropy.generalized.renyi_entropy`. Must
            be ``>= 0``.

    Returns:
        Entropy, shape ``[]``/``[S]`` for an unbatched trajectory or
        ``[B]``/``[S, B]`` for a batched one (matching whether
        ``box_size`` is scalar or ``[S]``), matching
        :func:`torchnlts.dimension.correlation.correlation_sum`'s own
        sweep/batch shape convention.

    Raises:
        ValueError: If ``order < 0``.
    """
    tensor, ndim = _ensure_batched(trajectory)
    dtype = tensor.dtype
    sizes = torch.as_tensor(box_size, dtype=dtype, device=tensor.device)
    swept = sizes.dim() > 0
    size_list = sizes.reshape(-1).tolist()
    batch_size = tensor.shape[0]

    values = torch.empty(len(size_list), batch_size, dtype=dtype, device=tensor.device)
    for batch_index in range(batch_size):
        for size_index, size in enumerate(size_list):
            probability = box_occupation_probabilities(tensor[batch_index], size)
            values[size_index, batch_index] = renyi_entropy(probability, order)

    if not swept:
        values = values.squeeze(0)
    if ndim != 3:
        values = values.squeeze(-1)
    return values


def generalized_dimension(box_size, entropy: torch.Tensor) -> torch.Tensor:
    """Generalized dimension D_q: OLS slope of Rényi entropy vs
    ``-log(box_size)``.

    Gradient class: exact (closed-form OLS slope; ordinary autodiff
    through whatever produced ``entropy`` -- inherits
    :func:`box_counting_entropy`'s ``none``, since that's this
    function's only realistic caller).

    This is a plain regression over every ``(box_size, entropy)`` pair
    given, with no automatic scaling-region detection -- see module
    docstring, same convention as
    :func:`torchnlts.dimension.correlation.correlation_dimension`.

    Reference: Russell, Hanson & Ott (1980) (D_q as the log-linear
    slope of the order-q Rényi entropy against box size).

    Args:
        box_size: Shape ``[S]``, the sweep ``entropy`` was evaluated at.
        entropy: Shape ``[S]`` (unbatched trajectory) or ``[S, B]``
            (batched trajectory), as returned by
            :func:`box_counting_entropy` for the same sweep.

    Returns:
        D_q estimate(s), shape ``[]`` for an unbatched ``entropy`` or
        ``[B]`` for a batched one. ``NaN`` if ``box_size`` has fewer than
        two distinct values.

    Raises:
        ValueError: If ``box_size``/``entropy`` do not have the documented
            shapes with matching sweep lengths.
    """
    neg_log_size = -torch.log(torch.as_tensor(box_size, dtype=entropy.dtype, device=entropy.device))
    return _ols_slope(neg_log_size, entropy)
