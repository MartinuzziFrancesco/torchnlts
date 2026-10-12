"""Recurrence quantification analysis (RQA) measures.

Reference: Marwan, Romano, Thiel & Kurths (2007), Physics Reports 438(5-6), 237-329.

``theiler_window`` excludes the pairs with ``|i - j| <= theiler_window``; RecurrenceAnalysis.jl's ``theiler``
keyword is ``theiler_window + 1``. The average and maximum line lengths return ``0`` when no line reaches ``lmin``,
as in Julia; the entropies return ``NaN``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional

from torchnlts._shape import _as_int, _shannon_entropy, _validate_square_matrix
from torchnlts.neighbors.theiler import theiler_window_mask

_WINDOW_ELEMENT_BUDGET = 2**24


def recurrence_rate(recurrence: torch.Tensor, *, theiler_window: int = 0) -> torch.Tensor:
    """Recurrence rate (RR): fraction of recurrent points.

    Reference: Marwan et al. (2007), eq. 11. Matches RecurrenceAnalysis.jl's ``recurrencerate``
    (``theiler = theiler_window + 1``).

    Gradient class: exact (a mean, differentiable whenever ``recurrence`` is, e.g. a soft recurrence matrix).

    Args:
        recurrence: Bool or float recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``, as returned by
            :func:`~torchnlts.recurrence.hard_recurrence_matrix` or :func:`~torchnlts.recurrence.soft_recurrence_matrix`.
        theiler_window: Pairs with ``|i - j| <= theiler_window`` are excluded from the numerator and the
            denominator. ``0`` (default) excludes only the always-recurrent diagonal; a negative window excludes
            nothing.

    Returns:
        Recurrence rate, shape ``[]`` or ``[B]``, in ``[0, 1]`` for a 0/1 or kernel-weight matrix; ``NaN`` if the
        window excludes every pair (``T <= theiler_window``).

    Raises:
        ValueError: If ``recurrence`` is not a square matrix or ``theiler_window`` is not an integer.
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
        # A negative window would count the main diagonal as one line of length num_points.
        raise ValueError(f"theiler_window must be >= 0 for line measures, got {theiler_window}")
    return theiler_window


def _run_length_counts(lines: torch.Tensor) -> torch.Tensor:
    """Number of runs of ``True`` of each length along the last dim: long ``[..., L + 1]`` for ``lines`` ``[..., L]``.

    Entry ``l`` counts the runs of length exactly ``l`` (entry ``0`` is ``0``). The cumulative count of ``False``
    entries is constant within a run of ``True``s, which identifies the runs without a Python loop.
    """
    lines = lines.long()
    group = torch.cumsum(1 - lines, dim=-1)
    bin_counts = lines.new_zeros(*lines.shape[:-1], lines.shape[-1] + 1).scatter_add(-1, group, lines)
    # Each run adds one positive bin_count at its group id, so scattering by bin_counts bins each run once.
    return torch.zeros_like(bin_counts).scatter_add(-1, bin_counts, (bin_counts > 0).long())


def _diagonal_line_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Number of diagonal lines of each length outside the Theiler window: long ``[..., num_points + 1]``.

    Entry ``l`` counts the lines of length exactly ``l`` (entry ``0`` is always ``0``). Matches RecurrenceAnalysis.jl's
    ``diagonalhistogram``. Loops over the ``2 * num_points - 1`` diagonal offsets, since diagonals differ in length.
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
    """Number of vertical lines of each length outside the Theiler window, laid out like the diagonal histogram.

    A column's line is cut by the Theiler window, as in RecurrenceAnalysis.jl's ``vl_histogram``.
    """
    theiler_window = _check_line_input(recurrence, theiler_window)
    return _run_length_counts(_columns_outside_window(recurrence, theiler_window)).sum(dim=-2)


def _recurrence_time_histogram(recurrence: torch.Tensor, theiler_window: int) -> torch.Tensor:
    """Number of recurrence times of each value outside the Theiler window: long ``[..., num_points + 1]``.

    Within a column, the time between consecutive vertical lines is the white gap between them plus half (floored)
    of each line's length, as in RecurrenceAnalysis.jl's ``rt_histogram``. All line lengths take part (no ``lmin``)
    and the Theiler window counts as white. Entry ``t`` counts the recurrence times equal to ``t``.
    """
    theiler_window = _check_line_input(recurrence, theiler_window)
    lines = _columns_outside_window(recurrence, theiler_window)  # [..., column, row]
    num_points = lines.shape[-1]
    index = torch.arange(num_points, device=lines.device)
    is_start = lines & ~functional.pad(lines, (1, 0))[..., :-1]
    is_end = lines & ~functional.pad(lines, (0, 1))[..., 1:]
    # Sorting puts the sentinel num_points last, so entry k of starts and ends is the k-th line of each column.
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
    """Validate ``lmin`` and build the diagonal (or vertical) line histogram and the line lengths ``l >= lmin``.

    Returns:
        ``(histogram, lengths)``, long, both ``[..., num_points + 1 - lmin]`` (``lengths`` broadcastable against
        the histogram's last dim), or empty along the last dim if ``lmin`` exceeds every possible line length.
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
        # No line can reach lmin, and amax has no identity for an empty reduction.
        return torch.zeros(recurrence.shape[:-2], dtype=torch.long, device=recurrence.device)
    return torch.where(histogram > 0, lengths, torch.zeros_like(lengths)).amax(dim=-1)


def _line_entropy(histogram: torch.Tensor) -> torch.Tensor:
    counts = histogram.to(torch.get_default_dtype())
    num_lines = counts.sum(dim=-1, keepdim=True)
    entropy = _shannon_entropy(counts / num_lines.clamp(min=1))
    return torch.where(num_lines.squeeze(-1) > 0, entropy, torch.full_like(entropy, float("nan")))


def determinism(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Determinism (DET): fraction of recurrent points forming diagonal lines.

    Reference: Marwan et al. (2007), eq. 13. Matches RecurrenceAnalysis.jl's ``determinism``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0`` (a negative window would count the main diagonal as a
            line). It applies to the line search and to the count of recurrent points.

    Returns:
        Determinism, shape ``[]`` or ``[B]``, in ``[0, 1]``; ``0.0`` (not ``NaN``) if no line reaches ``lmin``, as
        in Julia.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    return _line_point_fraction(recurrence, lmin, theiler_window, vertical=False)


def dl_average(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """L: average length of the diagonal lines of length ``>= lmin``.

    Reference: Marwan et al. (2007), eq. 12. Matches RecurrenceAnalysis.jl's ``dl_average``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Average line length, shape ``[]`` or ``[B]``; ``0.0`` (not ``NaN``) if no line reaches ``lmin``, as in
        Julia.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    return _average_line_length(histogram, lengths)


def dl_max(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Lmax: length of the longest diagonal line.

    Reference: Marwan et al. (2007), Sec. 2.1. Matches RecurrenceAnalysis.jl's ``dl_max``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Longest line length, long, shape ``[]`` or ``[B]``; ``0`` if no line reaches ``lmin``, as in Julia.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    return _max_line_length(histogram, lengths, recurrence)


def dl_entropy(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """ENTR: Shannon entropy of the diagonal line length distribution.

    Reference: Marwan et al. (2007), eq. 14. Matches RecurrenceAnalysis.jl's ``dl_entropy``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Entropy in nats, shape ``[]`` or ``[B]``; ``NaN`` if no line reaches ``lmin`` (an empty distribution has no
        entropy, unlike the ``0`` of :func:`~torchnlts.recurrence.dl_average` and :func:`~torchnlts.recurrence.dl_max`).

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, _ = _qualifying_line_lengths(recurrence, lmin, theiler_window)
    return _line_entropy(histogram)


def laminarity(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Laminarity (LAM): fraction of recurrent points forming vertical lines.

    Reference: Marwan et al. (2007), eq. 62. Matches RecurrenceAnalysis.jl's ``rqa(...)[:LAM]``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum vertical line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``. Vertical lines are cut by the window.

    Returns:
        Laminarity, shape ``[]`` or ``[B]``, in ``[0, 1]``; ``0.0`` if no line reaches ``lmin``.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    return _line_point_fraction(recurrence, lmin, theiler_window, vertical=True)


def trapping_time(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """TT: average length of the vertical lines of length ``>= lmin``.

    Reference: Marwan et al. (2007), eq. 63. Matches RecurrenceAnalysis.jl's ``trappingtime``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum vertical line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``. Vertical lines are cut by the window.

    Returns:
        Average vertical line length, shape ``[]`` or ``[B]``; ``0.0`` if no line reaches ``lmin``.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _average_line_length(histogram, lengths)


def vl_max(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """Vmax: length of the longest vertical line.

    Reference: Marwan et al. (2007), eq. 64. Matches RecurrenceAnalysis.jl's ``vl_max``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum vertical line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``. Vertical lines are cut by the window.

    Returns:
        Longest vertical line length, long, shape ``[]`` or ``[B]``; ``0`` if no line reaches ``lmin``.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, lengths = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _max_line_length(histogram, lengths, recurrence)


def vl_entropy(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """VENTR: Shannon entropy of the vertical line length distribution.

    Reference: Marwan et al. (2007), Sec. 3.5.3. Matches RecurrenceAnalysis.jl's ``vl_entropy``
    (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum vertical line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``. Vertical lines are cut by the window.

    Returns:
        Entropy in nats, shape ``[]`` or ``[B]``; ``NaN`` if no line reaches ``lmin``.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    histogram, _ = _qualifying_line_lengths(recurrence, lmin, theiler_window, vertical=True)
    return _line_entropy(histogram)


def rt_entropy(recurrence: torch.Tensor, *, theiler_window: int = 0) -> torch.Tensor:
    """RTE: Shannon entropy of the recurrence time distribution.

    Recurrence times are the gaps between the vertical lines of a column plus half (floored) of each neighbouring
    line's length, as in RecurrenceAnalysis.jl's ``rt_entropy`` (``theiler = theiler_window + 1``). The entropy is
    unnormalized and independent of ``lmin``.

    Reference: Little et al. (2007), BioMed. Eng. OnLine 6, 23 (recurrence period density entropy); Baptista et al.
    (2010).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Entropy in nats, shape ``[]`` or ``[B]``; ``NaN`` if no column has two lines.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, or ``theiler_window`` is not an integer ``>= 0``.
    """
    return _line_entropy(_recurrence_time_histogram(recurrence, theiler_window))


def ratio(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """RATIO: determinism over recurrence rate, ``DET / RR``.

    Reference: Marwan et al. (2007), Sec. 3.5.2. RecurrenceAnalysis.jl has no ``ratio``; this is
    ``determinism / recurrencerate`` (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Ratio, shape ``[]`` or ``[B]``; ``NaN`` if there are no recurrent points outside the Theiler window
        (``0 / 0``).

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    rate = recurrence_rate(recurrence, theiler_window=theiler_window)
    return determinism(recurrence, lmin=lmin, theiler_window=theiler_window) / rate


def divergence(recurrence: torch.Tensor, *, lmin: int = 2, theiler_window: int = 0) -> torch.Tensor:
    """DIV: inverse of the longest diagonal line, ``1 / Lmax``.

    Reference: Marwan et al. (2007), Sec. 3.5.2. Matches RecurrenceAnalysis.jl's ``divergence`` and
    ``rqa(...)[:DIV]`` (``theiler = theiler_window + 1``).

    Gradient class: none (run lengths of a boolean matrix).

    Args:
        recurrence: Bool recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        lmin: Minimum diagonal line length to count, an integer ``>= 1``.
        theiler_window: Theiler window, an integer ``>= 0``.

    Returns:
        Divergence, shape ``[]`` or ``[B]``; ``inf`` if no line reaches ``lmin`` (``Lmax = 0``), as in Julia.

    Raises:
        ValueError: If ``recurrence`` is not a square bool matrix, ``lmin`` is not an integer ``>= 1``, or
            ``theiler_window`` is not an integer ``>= 0``.
    """
    return dl_max(recurrence, lmin=lmin, theiler_window=theiler_window).to(torch.get_default_dtype()).reciprocal()


def windowed_rqa(
    recurrence: torch.Tensor, measure: Callable[..., torch.Tensor], window_size: int, *, step: int = 1, **measure_kwargs
) -> torch.Tensor:
    """RQA measure over square windows slid along the main diagonal.

    Window ``k`` is the sub-block ``recurrence[k * step : k * step + window_size]`` on both axes; a trailing partial
    window is dropped. Windows are evaluated in batched calls of ``measure``, in chunks of at most ``2**24`` gathered
    matrix elements.

    Reference: RecurrenceAnalysis.jl's ``windowed``; Marwan et al. (2007), Sec. 3.5.

    Gradient class: inherits ``measure``'s.

    Limitation: the chunk budget ignores dimensions mapped by ``torch.vmap`` and the measure's own temporaries, and a
    differentiable measure keeps every chunk alive until backward; each chunk copies its overlapping blocks. A
    strided view without copies would need a measure that accepts one.

    Args:
        recurrence: Recurrence matrix, shape ``[T, T]`` or ``[B, T, T]``.
        measure: Function mapping ``[..., w, w]`` matrices to one value each, e.g.
            :func:`~torchnlts.recurrence.determinism`.
        window_size: Window width ``w``, an integer with ``2 <= w <= T``.
        step: Offset between consecutive windows, an integer ``>= 1``.
        **measure_kwargs: Forwarded to ``measure`` (e.g. ``lmin``).

    Returns:
        Values of shape ``[W]`` or ``[B, W]``, ``W = (T - window_size) // step + 1``.

    Raises:
        ValueError: If ``measure`` is not callable, ``window_size`` or ``step`` is not an integer in range, or
            ``recurrence`` is not a square matrix.
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
