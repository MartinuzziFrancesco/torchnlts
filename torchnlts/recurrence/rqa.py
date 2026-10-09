"""Recurrence quantification analysis (RQA) measures.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6),
237-329.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional

from torchnlts._shape import _as_int, _validate_square_matrix
from torchnlts.neighbors.theiler import theiler_window_mask

_WINDOW_ELEMENT_BUDGET = 2**24  # max matrix elements gathered per measure call in windowed_rqa


def recurrence_rate(recurrence: torch.Tensor, theiler_window: int = 0) -> torch.Tensor:
    """Recurrence rate (RR): fraction of recurrent points.

    Gradient class: exact (a mean, differentiable whenever ``recurrence``
    is; ``recurrence`` itself may come from the non-differentiable
    :func:`torchnlts.recurrence.hard_recurrence_matrix` or the
    smooth :func:`torchnlts.recurrence.soft_recurrence_matrix`).

    Reference: Marwan et al. (2007), eq. 11. Matches
    RecurrenceAnalysis.jl's ``recurrencerate``, whose ``theiler`` keyword
    counts excluded diagonals differently (``theiler = theiler_window + 1``
    reproduces the same denominator; see its docstring).

    Args:
        recurrence: Bool or float recurrence matrix, shape ``[T, T]`` or
            ``[B, T, T]``, as returned by ``hard_recurrence_matrix`` or
            ``soft_recurrence_matrix``.
        theiler_window: Theiler window (see
            :func:`torchnlts.neighbors.theiler_window_mask`).
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


def _check_line_input(recurrence: torch.Tensor, theiler_window: int) -> int:
    """Validate a boolean recurrence matrix and a line-measure Theiler window."""
    if recurrence.dtype != torch.bool:
        raise ValueError(f"expected a boolean recurrence matrix, got dtype {recurrence.dtype}")
    _validate_square_matrix(recurrence)
    theiler_window = _as_int(theiler_window, "theiler_window")
    if theiler_window < 0:
        # Unlike recurrence_rate's plain point count, a negative window
        # (recurrence_rate's documented "exclude nothing") here would
        # count the always-recurrent main diagonal itself as one giant
        # line of length num_points -- a different and murkier
        # convention that RecurrenceAnalysis.jl special-cases rather than
        # applying uniformly, and one this port doesn't replicate.
        raise ValueError(f"theiler_window must be >= 0 for line measures, got {theiler_window}")
    return theiler_window


def _run_length_counts(lines: torch.Tensor) -> torch.Tensor:
    """Count of runs of ``True`` of each length along the last dim.

    Each line is run-length encoded via a group-id trick (cumulative count
    of ``False`` entries is constant within a run of ``True``s), so no
    Python loop over points is needed.

    Returns:
        Long tensor ``[..., L + 1]`` for ``lines`` ``[..., L]``; entry ``l``
        is the number of runs of length exactly ``l`` (entry ``0`` is ``0``).
    """
    lines = lines.long()
    group = torch.cumsum(1 - lines, dim=-1)
    bin_counts = lines.new_zeros(*lines.shape[:-1], lines.shape[-1] + 1).scatter_add(-1, group, lines)
    # Each run of True contributes exactly one positive entry to
    # bin_counts (indexed by its group id), so scattering "1 where
    # positive" by bin_counts itself both selects each run once and
    # bins it by length -- no separate start-of-run detection needed.
    return torch.zeros_like(bin_counts).scatter_add(-1, bin_counts, (bin_counts > 0).long())


def _diagonal_line_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Count of diagonal lines of each length, outside the Theiler window.

    Shared by the diagonal-line RQA measures, matching RecurrenceAnalysis.jl's
    ``diagonalhistogram``. Loops only over the ``2 * num_points - 1``
    diagonal offsets, which is unavoidable since diagonals differ in length.

    Returns:
        Long tensor of shape ``[..., num_points + 1]``; entry ``l`` is the
        number of diagonal lines of length exactly ``l`` (entry ``0`` is
        always ``0``, kept only so indices line up with lengths).
    """
    theiler_window = _check_line_input(recurrence, theiler_window)
    num_points = recurrence.shape[-1]
    histogram = recurrence.new_zeros(*recurrence.shape[:-2], num_points + 1, dtype=torch.long)
    for offset in range(-(num_points - 1), num_points):
        if abs(offset) <= theiler_window:
            continue
        counts = _run_length_counts(torch.diagonal(recurrence, offset=offset, dim1=-2, dim2=-1))
        histogram[..., : counts.shape[-1]] += counts
    return histogram


def _columns_outside_window(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """``[..., column, row]`` view of ``recurrence`` with the Theiler window set to ``False``."""
    mask = theiler_window_mask(recurrence.shape[-1], theiler_window, device=recurrence.device)
    return (recurrence & mask).transpose(-1, -2)


def _vertical_line_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Count of vertical lines of each length, outside the Theiler window.

    Matches RecurrenceAnalysis.jl's ``vl_histogram``: a column's line is cut
    by the Theiler window. Same layout as :func:`_diagonal_line_histogram`.
    """
    theiler_window = _check_line_input(recurrence, theiler_window)
    return _run_length_counts(_columns_outside_window(recurrence, theiler_window)).sum(dim=-2)


def _recurrence_time_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Count of recurrence times of each value, outside the Theiler window.

    Within each column, the time between consecutive vertical lines is the
    white gap between them plus half (floored) of each line's length, as in
    RecurrenceAnalysis.jl's ``rt_histogram``. All line lengths take part
    (no ``lmin``), and the Theiler window counts as white.

    Returns:
        Long tensor ``[..., num_points + 1]``; entry ``t`` is the number of
        recurrence times equal to ``t``.
    """
    theiler_window = _check_line_input(recurrence, theiler_window)
    lines = _columns_outside_window(recurrence, theiler_window)  # [..., column, row]
    num_points = lines.shape[-1]
    index = torch.arange(num_points, device=lines.device)
    is_start = lines & ~functional.pad(lines, (1, 0))[..., :-1]
    is_end = lines & ~functional.pad(lines, (0, 1))[..., 1:]
    # Sorting pushes the sentinel ``num_points`` to the end, so entry k of
    # starts and ends describes the k-th line of each column.
    starts = torch.where(is_start, index, num_points).sort(dim=-1).values
    ends = torch.where(is_end, index, num_points).sort(dim=-1).values
    half = torch.where(starts < num_points, (ends - starts + 1) // 2, 0)
    times = starts[..., 1:] - ends[..., :-1] - 1 + half[..., :-1] + half[..., 1:]
    valid = starts[..., 1:] < num_points
    # Out-of-place and derived from a batched tensor so torch.vmap works.
    histogram = times.new_zeros(*times.shape[:-1], num_points + 1).scatter_add(
        -1, torch.where(valid, times, 0), valid.long()
    )
    return histogram.sum(dim=-2)


def _qualifying_line_lengths(
    recurrence: torch.Tensor, lmin: int, theiler_window: int, vertical: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared setup for every line RQA measure: validate ``lmin``, build the
    diagonal (or vertical) line histogram, and slice both it and a matching
    ``lengths`` tensor down to ``l >= lmin`` (long dtype; callers needing
    float cast it themselves, since the max measures want exact integers).

    Returns:
        ``(histogram, lengths)``, both shape ``[..., num_points + 1 - lmin]``
        (``lengths`` broadcastable against ``histogram``'s last dim), or
        both empty along the last dim if ``lmin`` exceeds every possible
        line length (e.g. ``lmin > num_points``).
    """
    lmin = _as_int(lmin, "lmin")
    if lmin < 1:
        raise ValueError(f"lmin must be >= 1, got {lmin}")
    build = _vertical_line_histogram if vertical else _diagonal_line_histogram
    histogram = build(recurrence, theiler_window)[..., lmin:]
    lengths = torch.arange(lmin, histogram.shape[-1] + lmin, device=recurrence.device, dtype=torch.long)
    return histogram, lengths


def _line_point_fraction(recurrence: torch.Tensor, lmin: int, theiler_window: int, vertical: bool) -> torch.Tensor:
    """Fraction of recurrent points outside the Theiler window lying in lines of length >= ``lmin``."""
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical)
    numerator = (lengths * histogram).sum(dim=-1).to(torch.get_default_dtype())
    mask = theiler_window_mask(recurrence.shape[-1], theiler_window, device=recurrence.device)
    denominator = torch.where(mask, recurrence, torch.zeros_like(recurrence)).sum(dim=(-2, -1)).to(numerator.dtype)
    return numerator / denominator.clamp(min=1)


def _average_line_length(histogram: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    total_length = (lengths * histogram).sum(dim=-1).to(torch.get_default_dtype())
    return total_length / histogram.sum(dim=-1).clamp(min=1).to(torch.get_default_dtype())


def _max_line_length(histogram: torch.Tensor, lengths: torch.Tensor, recurrence: torch.Tensor) -> torch.Tensor:
    if histogram.shape[-1] == 0:
        # lmin exceeds every possible line length (e.g. a 1-point
        # trajectory with the default lmin=2): no line can ever
        # qualify, and amax has no identity for an empty reduction.
        return torch.zeros(recurrence.shape[:-2], dtype=torch.long, device=recurrence.device)
    return torch.where(histogram > 0, lengths, torch.zeros_like(lengths)).amax(dim=-1)


def _line_entropy(histogram: torch.Tensor) -> torch.Tensor:
    counts = histogram.to(torch.get_default_dtype())
    num_lines = counts.sum(dim=-1, keepdim=True)
    probability = counts / num_lines.clamp(min=1)
    # probability is 0 wherever histogram is 0, so 0 * log(clamped-tiny) is
    # always 0 -- no separate has_lines mask needed to avoid 0 * -inf.
    log_probability = torch.log(probability.clamp(min=torch.finfo(probability.dtype).tiny))
    entropy = -(probability * log_probability).sum(dim=-1)
    return torch.where(num_lines.squeeze(-1) > 0, entropy, torch.full_like(entropy, float("nan")))


def determinism(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Determinism (DET): fraction of recurrent points forming diagonal lines.

    Gradient class: none (discrete run-length structure of a boolean matrix).

    Reference: Marwan et al. (2007), eq. 13. Matches RecurrenceAnalysis.jl's
    ``determinism`` (``theiler = theiler_window + 1``, see
    :func:`~torchnlts.recurrence.recurrence_rate`), including its convention of ``0.0`` (not
    ``NaN``) when no line reaches ``lmin`` -- a deliberate empty-case
    convention (Julia's implementation special-cases it directly); it's a
    safe default here too, since the numerator (points in qualifying
    lines) can only be positive if the denominator (all recurrent points)
    is too, so a zero denominator always coincides with a zero numerator.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count as "deterministic".
        theiler_window: Theiler window, forwarded to both the line search
            and the point-count denominator (see :func:`~torchnlts.recurrence.recurrence_rate`).
            Unlike :func:`~torchnlts.recurrence.recurrence_rate`, must be ``>= 0`` here (see
            :func:`_diagonal_line_histogram`).

    Returns:
        Scalar (or ``[B]``) determinism in ``[0, 1]``, ``0.0`` if no line
        reaches ``lmin``.
    """
    return _line_point_fraction(recurrence, lmin, theiler_window, vertical=False)


def dl_average(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """L: average length of diagonal lines (Marwan et al. (2007), eq. 12).

    Gradient class: none (see :func:`~torchnlts.recurrence.determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_average`` (``theiler =
    theiler_window + 1``, see :func:`~torchnlts.recurrence.recurrence_rate`), including its
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
    return _average_line_length(histogram, lengths)


def dl_max(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Lmax: length of the longest diagonal line.

    Gradient class: none (see :func:`~torchnlts.recurrence.determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_max`` (``theiler = theiler_window
    + 1``, see :func:`~torchnlts.recurrence.recurrence_rate`), including its convention of
    ``0`` when no line reaches ``lmin``.

    Args: same as :func:`~torchnlts.recurrence.dl_average`.

    Returns:
        Scalar (or ``[B]``) longest diagonal line length, ``0`` if none qualify.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    return _max_line_length(histogram, lengths, recurrence)


def dl_entropy(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """ENTR: Shannon entropy of the diagonal line length distribution.

    Gradient class: none (see :func:`~torchnlts.recurrence.determinism`).

    Matches RecurrenceAnalysis.jl's ``dl_entropy`` (``theiler =
    theiler_window + 1``, see :func:`~torchnlts.recurrence.recurrence_rate`).

    Args: same as :func:`~torchnlts.recurrence.dl_average`.

    Returns:
        Scalar (or ``[B]``) entropy (nats), or ``NaN`` if no line reaches
        ``lmin`` (an empty distribution has no defined entropy, unlike
        :func:`~torchnlts.recurrence.dl_average`/:func:`~torchnlts.recurrence.dl_max`'s ``0`` convention).
    """
    histogram, _ = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    return _line_entropy(histogram)


def laminarity(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Laminarity (LAM): fraction of recurrent points forming vertical lines.

    Gradient class: none (discrete run-length structure of a boolean matrix).

    Reference: Marwan et al. (2007), eq. 62. Matches RecurrenceAnalysis.jl's
    ``rqa(...)[:LAM]`` (``theiler = theiler_window + 1``), with ``0.0`` when no
    line reaches ``lmin``. Vertical lines are cut by the Theiler window.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum vertical line length to count.
        theiler_window: Theiler window, ``>= 0`` (see :func:`~torchnlts.recurrence.determinism`).

    Returns:
        Scalar (or ``[B]``) laminarity in ``[0, 1]``.
    """
    return _line_point_fraction(recurrence, lmin, theiler_window, vertical=True)


def trapping_time(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """TT: average length of vertical lines.

    Gradient class: none (see :func:`~torchnlts.recurrence.laminarity`).

    Reference: Marwan et al. (2007), eq. 63. Matches RecurrenceAnalysis.jl's
    ``trappingtime`` (``theiler = theiler_window + 1``), ``0.0`` if no line
    reaches ``lmin``.

    Args: same as :func:`~torchnlts.recurrence.laminarity`.

    Returns:
        Scalar (or ``[B]``) average vertical line length.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _average_line_length(histogram, lengths)


def vl_max(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Vmax: length of the longest vertical line.

    Gradient class: none (see :func:`~torchnlts.recurrence.laminarity`).

    Reference: Marwan et al. (2007), eq. 64. Matches RecurrenceAnalysis.jl's
    ``vl_max`` (``theiler = theiler_window + 1``), ``0`` if no line reaches ``lmin``.

    Args: same as :func:`~torchnlts.recurrence.laminarity`.

    Returns:
        Scalar (or ``[B]``) longest vertical line length (long).
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _max_line_length(histogram, lengths, recurrence)


def vl_entropy(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """VENTR: Shannon entropy of the vertical line length distribution.

    Gradient class: none (see :func:`~torchnlts.recurrence.laminarity`).

    Matches RecurrenceAnalysis.jl's ``vl_entropy`` (``theiler = theiler_window + 1``).

    Args: same as :func:`~torchnlts.recurrence.laminarity`.

    Returns:
        Scalar (or ``[B]``) entropy (nats), ``NaN`` if no line reaches ``lmin``.
    """
    histogram, _ = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _line_entropy(histogram)


def rt_entropy(recurrence: torch.Tensor, theiler_window: int = 0) -> torch.Tensor:
    """RTE: Shannon entropy of the recurrence time distribution.

    Gradient class: none (see :func:`~torchnlts.recurrence.laminarity`).

    Reference: Little et al. (2007), BioMed. Eng. OnLine 6, 23 (recurrence
    period density entropy); Baptista et al. (2010). Recurrence times are the
    gaps between vertical lines of a column (white vertical lines) plus half
    of each neighbouring line's length, as in RecurrenceAnalysis.jl's
    ``rt_entropy`` (``theiler = theiler_window + 1``). Unnormalized, in nats,
    and independent of ``lmin``.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        theiler_window: Theiler window, ``>= 0`` (see :func:`~torchnlts.recurrence.determinism`).

    Returns:
        Scalar (or ``[B]``) entropy (nats), ``NaN`` if no column has two lines.
    """
    return _line_entropy(_recurrence_time_histogram(recurrence, theiler_window))


def ratio(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """RATIO: determinism over recurrence rate, ``DET / RR``.

    Gradient class: none (see :func:`~torchnlts.recurrence.determinism`).

    Reference: Marwan et al. (2007), Sec. 3.5.2. RecurrenceAnalysis.jl has no
    ``ratio``; this is ``determinism / recurrencerate`` with
    ``theiler = theiler_window + 1``.

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count.
        theiler_window: Theiler window, ``>= 0`` (see :func:`~torchnlts.recurrence.determinism`).

    Returns:
        Scalar (or ``[B]``) ratio, ``NaN`` if there are no recurrent points
        outside the Theiler window (``0 / 0``).
    """
    return determinism(recurrence, lmin, theiler_window) / recurrence_rate(recurrence, theiler_window)


def divergence(recurrence: torch.Tensor, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """DIV: inverse of the longest diagonal line, ``1 / Lmax``.

    Gradient class: none (see :func:`~torchnlts.recurrence.determinism`).

    Reference: Marwan et al. (2007), Sec. 3.5.2. Matches RecurrenceAnalysis.jl's
    ``divergence`` and ``rqa(...)[:DIV]`` (``theiler = theiler_window + 1``),
    including ``inf`` when no line reaches ``lmin`` (``Lmax = 0``).

    Args: same as :func:`~torchnlts.recurrence.dl_max`.

    Returns:
        Scalar (or ``[B]``) divergence, ``inf`` if no line qualifies.
    """
    return dl_max(recurrence, lmin, theiler_window).to(torch.get_default_dtype()).reciprocal()


def windowed_rqa(
    recurrence: torch.Tensor, measure: Callable[..., torch.Tensor], window_size: int, step: int = 1, **measure_kwargs
) -> torch.Tensor:
    """RQA measure over square windows slid along the main diagonal.

    Gradient class: inherits ``measure``'s.

    Reference: RecurrenceAnalysis.jl's ``windowed``. Window ``k`` is the
    sub-block ``recurrence[k * step : k * step + window_size]`` on both
    axes; a trailing partial window is dropped. Windows are evaluated in
    batched calls of ``measure``, in chunks of at most ``2**24`` gathered
    matrix elements (one call if everything fits).

    Limitation: the budget bounds the blocks gathered per call over the
    visible batch dimensions only (dimensions mapped by ``torch.vmap``
    multiply it, as do the measure's own temporaries, and a differentiable
    measure keeps every chunk alive until backward); each chunk copies its
    overlapping blocks, at least one window. A strided view without copies
    would need a measure that accepts one.

    Args:
        recurrence: Recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        measure: Function mapping ``[..., w, w]`` matrices to one value each,
            e.g. :func:`~torchnlts.recurrence.determinism`.
        window_size: Window width ``w``, ``2 <= w <= T``.
        step: Offset between consecutive windows, ``>= 1``.
        **measure_kwargs: Forwarded to ``measure`` (e.g. ``lmin``).

    Returns:
        ``[W]`` (or ``[B, W]``) values, ``W = (T - window_size) // step + 1``.

    Raises:
        ValueError: On a non-callable ``measure``, non-integer or out-of-range
            ``window_size`` / ``step``, or a non-square ``recurrence``.
    """
    _validate_square_matrix(recurrence)
    if not callable(measure):
        raise ValueError(f"measure must be callable, got {measure!r}")
    window_size = _as_int(window_size, "window_size")
    step = _as_int(step, "step")
    num_points = recurrence.shape[-1]
    if not 2 <= window_size <= num_points:
        raise ValueError(f"window_size must be in [2, {num_points}], got {window_size}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    num_windows = (num_points - window_size) // step + 1
    offsets = torch.arange(window_size, device=recurrence.device)
    batch_size = max(1, recurrence.shape[:-2].numel())
    chunk = max(1, _WINDOW_ELEMENT_BUDGET // (batch_size * window_size * window_size))
    results = []
    for first in range(0, num_windows, chunk):
        starts = torch.arange(first, min(first + chunk, num_windows), device=recurrence.device) * step
        index = starts[:, None] + offsets
        blocks = recurrence[..., index[:, :, None], index[:, None, :]]  # [..., W_chunk, w, w]
        values = measure(blocks.reshape(-1, window_size, window_size), **measure_kwargs)
        results.append(values.reshape(*recurrence.shape[:-2], index.shape[0]))
    return torch.cat(results, dim=-1)
