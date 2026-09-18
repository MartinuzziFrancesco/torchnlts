"""Recurrence quantification analysis (RQA) measures.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _validate_square_matrix
from torchnlts.neighbors.theiler import theiler_window_mask


def recurrence_rate(recurrence: torch.Tensor, theiler_window: int = 0) -> torch.Tensor:
    """Recurrence rate (RR): fraction of recurrent points.

    Gradient class: exact (a mean, differentiable whenever ``recurrence``
    is; ``recurrence`` itself may come from the non-differentiable
    :func:`torchnlts.recurrence.matrix.hard_recurrence_matrix` or the
    smooth :func:`torchnlts.recurrence.matrix.soft_recurrence_matrix`).

    Reference: Marwan et al. (2007), eq. 11. Matches
    RecurrenceAnalysis.jl's ``recurrencerate``, whose ``theiler`` keyword
    counts excluded diagonals differently (``theiler = theiler_window + 1``
    reproduces the same denominator; see its docstring).

    Args:
        recurrence: Bool or float recurrence matrix, shape ``[T, T]`` or
            ``[B, T, T]``, as returned by ``hard_recurrence_matrix`` or
            ``soft_recurrence_matrix``.
        theiler_window: Theiler window (see
            :func:`torchnlts.neighbors.theiler.theiler_window_mask`).
            Pairs with ``|i - j| <= theiler_window`` are excluded from
            both the numerator and denominator; ``0`` (default) excludes
            only the always-recurrent diagonal. Negative excludes nothing.

    Returns:
        Scalar (or ``[B]``) recurrence rate in ``[0, 1]``, or ``NaN`` if
        ``theiler_window`` excludes every pair (e.g. ``T <= theiler_window``),
        the honest answer for an empty denominator rather than a crash.
    """
    _validate_square_matrix(recurrence)
    weights = recurrence.to(torch.get_default_dtype()) if recurrence.dtype == torch.bool else recurrence
    mask = theiler_window_mask(recurrence.shape[-1], theiler_window, device=recurrence.device)
    masked = torch.where(mask, weights, torch.zeros_like(weights))
    return masked.sum(dim=(-2, -1)) / mask.sum()


def _diagonal_line_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Count of diagonal lines of each length, outside the Theiler window.

    Shared by :func:`determinism` and (once added) the other diagonal-line
    RQA measures (average/max/entropy of diagonal line length), matching
    RecurrenceAnalysis.jl's ``diagonalhistogram`` split into a histogram
    plus per-measure reductions.

    Each diagonal is run-length encoded via a group-id trick (cumulative
    count of ``False`` entries is constant within a run of ``True``s), so
    the whole histogram is built without a Python loop over points --
    only over the ``2 * num_points - 1`` diagonal offsets, which is
    unavoidable since diagonals have different lengths.

    Returns:
        Long tensor of shape ``[..., num_points + 1]``; entry ``l`` is the
        number of diagonal lines of length exactly ``l`` (entry ``0`` is
        always ``0``, kept only so indices line up with lengths).
    """
    if recurrence.dtype != torch.bool:
        raise ValueError(f"expected a boolean recurrence matrix, got dtype {recurrence.dtype}")
    _validate_square_matrix(recurrence)
    if theiler_window < 0:
        # Unlike recurrence_rate's plain point count, a negative window
        # (recurrence_rate's documented "exclude nothing") here would
        # count the always-recurrent main diagonal itself as one giant
        # diagonal line of length num_points -- a different and murkier
        # convention that RecurrenceAnalysis.jl special-cases rather than
        # applying uniformly, and one this port doesn't replicate.
        raise ValueError(f"theiler_window must be >= 0 for diagonal-line measures, got {theiler_window}")
    num_points = recurrence.shape[-1]
    histogram = recurrence.new_zeros(*recurrence.shape[:-2], num_points + 1, dtype=torch.long)
    for offset in range(-(num_points - 1), num_points):
        if abs(offset) <= theiler_window:
            continue
        diag = torch.diagonal(recurrence, offset=offset, dim1=-2, dim2=-1).long()  # [..., length]
        group = torch.cumsum(1 - diag, dim=-1)
        bin_counts = diag.new_zeros(*diag.shape[:-1], diag.shape[-1] + 1).scatter_add(-1, group, diag)
        # Each run of True contributes exactly one positive entry to
        # bin_counts (indexed by its group id), so scattering "1 where
        # positive" by bin_counts itself both selects each run once and
        # bins it by length -- no separate start-of-run detection needed.
        histogram.scatter_add_(-1, bin_counts, (bin_counts > 0).long())
    return histogram


def _qualifying_line_lengths(recurrence: torch.Tensor, lmin: int, theiler_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared setup for every diagonal-line RQA measure: validate ``lmin``,
    build the diagonal line histogram, and slice both it and a matching
    ``lengths`` tensor down to ``l >= lmin`` (long dtype; callers needing
    float cast it themselves, since dl_max wants exact integers).

    Returns:
        ``(histogram, lengths)``, both shape ``[..., num_points + 1 - lmin]``
        (``lengths`` broadcastable against ``histogram``'s last dim), or
        both empty along the last dim if ``lmin`` exceeds every possible
        diagonal line length (e.g. ``lmin > num_points``).
    """
    if lmin < 1:
        raise ValueError(f"lmin must be >= 1, got {lmin}")
    histogram = _diagonal_line_histogram(recurrence, theiler_window)[..., lmin:]
    lengths = torch.arange(lmin, histogram.shape[-1] + lmin, device=recurrence.device, dtype=torch.long)
    return histogram, lengths


def determinism(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Determinism (DET): fraction of recurrent points forming diagonal lines.

    Gradient class: none (discrete run-length structure of a boolean matrix).

    Reference: Marwan et al. (2007), eq. 13. Matches RecurrenceAnalysis.jl's
    ``determinism`` (``theiler = theiler_window + 1``, see
    :func:`recurrence_rate`), including its convention of ``0.0`` (not
    ``NaN``) when no line reaches ``lmin`` -- a deliberate empty-case
    convention (Julia's implementation special-cases it directly); it's a
    safe default here too, since the numerator (points in qualifying
    lines) can only be positive if the denominator (all recurrent points)
    is too, so a zero denominator always coincides with a zero numerator.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count as "deterministic".
        theiler_window: Theiler window, forwarded to both the line search
            and the point-count denominator (see :func:`recurrence_rate`).
            Unlike :func:`recurrence_rate`, must be ``>= 0`` here (see
            :func:`_diagonal_line_histogram`).

    Returns:
        Scalar (or ``[B]``) determinism in ``[0, 1]``, ``0.0`` if no line
        reaches ``lmin``.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    numerator = (lengths * histogram).sum(dim=-1).to(torch.get_default_dtype())
    mask = theiler_window_mask(recurrence.shape[-1], theiler_window, device=recurrence.device)
    denominator = torch.where(mask, recurrence, torch.zeros_like(recurrence)).sum(dim=(-2, -1)).to(numerator.dtype)
    return numerator / denominator.clamp(min=1)


def dl_average(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """L: average length of diagonal lines (Marwan et al. (2007), eq. 12).

    Gradient class: none (see :func:`determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_average`` (``theiler =
    theiler_window + 1``, see :func:`recurrence_rate`), including its
    convention of ``0.0`` (not ``NaN``) when no line reaches ``lmin``.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count.
        theiler_window: Theiler window, forwarded to the line search. Must
            be ``>= 0`` (see :func:`_diagonal_line_histogram`).

    Returns:
        Scalar (or ``[B]``) average line length, ``0.0`` if none qualify.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    num_lines = histogram.sum(dim=-1)
    total_length = (lengths * histogram).sum(dim=-1)
    return total_length.to(torch.get_default_dtype()) / num_lines.clamp(min=1).to(torch.get_default_dtype())


def dl_max(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Lmax: length of the longest diagonal line.

    Gradient class: none (see :func:`determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_max`` (``theiler = theiler_window
    + 1``, see :func:`recurrence_rate`), including its convention of
    ``0`` when no line reaches ``lmin``.

    Args: same as :func:`dl_average`.

    Returns:
        Scalar (or ``[B]``) longest diagonal line length, ``0`` if none qualify.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    if histogram.shape[-1] == 0:
        # lmin exceeds every possible diagonal line length (e.g. a
        # 1-point trajectory with the default lmin=2): no line can ever
        # qualify, and amax has no identity for an empty reduction.
        return torch.zeros(recurrence.shape[:-2], dtype=torch.long, device=recurrence.device)
    present = histogram > 0
    return torch.where(present, lengths, torch.zeros_like(lengths)).amax(dim=-1)


def dl_entropy(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """ENTR: Shannon entropy of the diagonal line length distribution.

    Gradient class: none (see :func:`determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_entropy`` (``theiler =
    theiler_window + 1``, see :func:`recurrence_rate`).

    Args: same as :func:`dl_average`.

    Returns:
        Scalar (or ``[B]``) entropy (nats), or ``NaN`` if no line reaches
        ``lmin`` (an empty distribution has no defined entropy, unlike
        :func:`dl_average`/:func:`dl_max`'s ``0`` convention).
    """
    histogram, _ = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    counts = histogram.to(torch.get_default_dtype())
    num_lines = counts.sum(dim=-1, keepdim=True)
    probability = counts / num_lines.clamp(min=1)
    # probability is 0 wherever histogram is 0, so 0 * log(clamped-tiny) is
    # always 0 -- no separate has_lines mask needed to avoid 0 * -inf.
    log_probability = torch.log(probability.clamp(min=torch.finfo(probability.dtype).tiny))
    entropy = -(probability * log_probability).sum(dim=-1)
    return torch.where(num_lines.squeeze(-1) > 0, entropy, torch.full_like(entropy, float("nan")))
