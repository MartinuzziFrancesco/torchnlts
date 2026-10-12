"""Multiscale entropy: sample entropy of coarse-grained copies of a trajectory over a range of scales.

Reference: Costa, Goldberger & Peng (2002), Phys. Rev. Lett. 89, 068102, the (unnumbered) coarse-graining formula on
p. 2 (their eq. 1 is the Shannon entropy definition): coarse-grain at scale ``s`` (non-overlapping length-``s`` window
means), take the sample entropy of each coarse-grained series, and read the result as a curve over scales. The
composite variant (Wu, Wu, Lin, Wang & Lee (2013), Entropy 15(3), 1069-1084) averages over all window offsets at each
scale.

:func:`~torchnlts.entropy.coarse_grain` is the only new computation; the curves loop over scales and apply
:func:`~torchnlts.entropy.sample_entropy` or :func:`~torchnlts.entropy.soft_sample_entropy`, whose gradient class they
inherit.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _output_dtype, _restore_shape_preserving, _work_dtype
from torchnlts.convert import as_tensor
from torchnlts.entropy.sample import _mask_nonfinite_input, sample_entropy, soft_sample_entropy


def _validate_scales(scales) -> list[int]:
    """``scales`` as a non-empty list of positive Python ints."""
    try:
        scales = [_as_int(scale, "scale") for scale in scales]
    except TypeError:
        raise ValueError(f"scales must be an iterable of positive integers, got {scales!r}") from None
    if not scales or any(scale < 1 for scale in scales):
        raise ValueError(f"scales must be a non-empty iterable of positive integers, got {scales!r}")
    return scales


def coarse_grain(trajectory, scale: int, *, offset: int = 0) -> torch.Tensor:
    """Costa et al.'s regular coarse-graining: non-overlapping length-``scale`` window means along the time axis.

    Gradient class: exact (a reshape and a mean).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A tensor, numpy array, or pandas Series/DataFrame.
            Integer input is converted to the default dtype; half precision is averaged in float32 and cast back.
        scale: Window length, a positive integer; ``scale=1`` returns ``trajectory`` unchanged.
        offset: Leading samples skipped before windowing (Wu et al.'s ``k - 1``), ``0 <= offset < scale``.

    Returns:
        Shape ``[T']``, ``[T', D]`` or ``[B, T', D]`` with ``T' = (T - offset) // scale``; the trailing
        ``(T - offset) % scale`` samples are dropped. The feature axis keeps its meaning (unlike
        :func:`torchnlts.embedding.delay_embedding`), so a ``[T]`` input stays ``[T']``. Non-finite samples are not
        checked: a window containing one has a non-finite mean, and dropped samples are ignored.

    Raises:
        ValueError: If ``scale`` is not a positive integer or ``offset`` is not an integer in ``[0, scale)``.
    """
    scale = _as_int(scale, "scale")
    offset = _as_int(offset, "offset")
    if scale < 1:
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    if not 0 <= offset < scale:
        raise ValueError(f"offset must satisfy 0 <= offset < scale, got offset={offset!r}, scale={scale!r}")
    batched, ndim = _ensure_batched(trajectory)
    out_dtype = _output_dtype(batched)
    batched = batched[..., offset:, :].to(_work_dtype(batched.dtype))
    num_windows = batched.shape[-2] // scale
    truncated = batched[..., : num_windows * scale, :]
    windows = truncated.reshape(*truncated.shape[:-2], num_windows, scale, truncated.shape[-1])
    return _restore_shape_preserving(windows.mean(dim=-2).to(out_dtype), ndim)


def multiscale_entropy(
    trajectory,
    scales,
    radius,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Sample entropy at each of a range of coarse-graining scales.

    Gradient class: none (inherits :func:`~torchnlts.entropy.sample_entropy`'s).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Non-empty iterable of positive integers; ``scales=[1]`` recovers plain ``sample_entropy``.
        radius: Tolerance r, forwarded unchanged at every scale (Costa et al. fix r relative to the original series,
            commonly ``0.15 * trajectory.std()``, which the caller computes).
        embedding_dim: Template length m, forwarded to ``sample_entropy``.
        delay: Embedding lag, forwarded to ``sample_entropy``.
        theiler_window: Forwarded to ``sample_entropy``.
        norm_order: Forwarded to ``sample_entropy``.

    Returns:
        Entropy curve of shape ``[num_scales]`` prepended to the shape of one ``sample_entropy`` call (``[]``/``[R]``
        unbatched, ``[B]``/``[R, B]`` batched), e.g. ``[num_scales, B]`` for a batched trajectory and a scalar
        ``radius``. Entries follow ``sample_entropy``'s ``NaN``/``inf`` cases. A row containing ``NaN`` or ``inf``
        anywhere, including samples that ``coarse_grain`` drops, is ``NaN`` at every scale.

    Raises:
        ValueError: If ``scales`` is empty or holds a non-integer or a scale ``< 1``, ``embedding_dim`` or ``delay`` is
            not a positive integer, ``theiler_window`` is not an integer ``>= 0``, or a scale leaves fewer than
            ``embedding_dim * delay + 1`` coarse-grained points to embed (this aborts the whole curve rather than giving
            ``NaN`` for that scale).
    """
    scales = _validate_scales(scales)
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    curve = torch.stack(
        [
            sample_entropy(
                coarse_grain(trajectory, scale),
                radius,
                embedding_dim=embedding_dim,
                delay=delay,
                theiler_window=theiler_window,
                norm_order=norm_order,
            )
            for scale in scales
        ]
    )
    return torch.where(nonfinite, float("nan"), curve).to(out_dtype)


def soft_multiscale_entropy(
    trajectory,
    scales,
    temperature,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted multiscale entropy, a smooth stand-in for :func:`~torchnlts.entropy.multiscale_entropy`.

    Gradient class: smooth (inherits :func:`~torchnlts.entropy.soft_sample_entropy`'s, including its Chebyshev-tie
    caveat).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Non-empty iterable of positive integers.
        temperature: Kernel bandwidth, forwarded unchanged at every scale (see
            :func:`~torchnlts.entropy.multiscale_entropy`'s ``radius``).
        embedding_dim: Template length m, forwarded to ``soft_sample_entropy``.
        delay: Embedding lag, forwarded to ``soft_sample_entropy``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_sample_entropy``.
        theiler_window: Forwarded to ``soft_sample_entropy``.
        norm_order: Forwarded to ``soft_sample_entropy``.

    Returns:
        Entropy curve, with the shape and ``NaN`` conventions of :func:`~torchnlts.entropy.multiscale_entropy`.

    Raises:
        ValueError: As :func:`~torchnlts.entropy.multiscale_entropy`.
    """
    scales = _validate_scales(scales)
    trajectory, nonfinite = _mask_nonfinite_input(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    curve = torch.stack(
        [
            soft_sample_entropy(
                coarse_grain(trajectory, scale),
                temperature,
                embedding_dim=embedding_dim,
                delay=delay,
                kind=kind,
                theiler_window=theiler_window,
                norm_order=norm_order,
            )
            for scale in scales
        ]
    )
    return torch.where(nonfinite, float("nan"), curve).to(out_dtype)


def _composite_curve(trajectory, scales, entropy) -> torch.Tensor:
    """Average ``entropy`` over all window offsets at each scale (half input is computed in float32)."""
    scales = _validate_scales(scales)
    trajectory = as_tensor(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    curve = torch.stack(
        [
            torch.stack([entropy(coarse_grain(trajectory, scale, offset=shift)) for shift in range(scale)]).mean(dim=0)
            for scale in scales
        ]
    )
    return curve.to(out_dtype)


def composite_multiscale_entropy(
    trajectory,
    scales,
    radius,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Composite multiscale entropy: sample entropy averaged over all window offsets at each scale.

    At scale ``s`` the ``s`` coarse-grained series ``coarse_grain(trajectory, s, offset=offset)``, ``offset = 0..s-1``,
    each get a sample entropy, and the ``s`` values are averaged.

    Reference: Wu, Wu, Lin, Wang & Lee (2013), "Time series analysis using composite multiscale entropy", Entropy
    15(3), 1069-1084, doi:10.3390/e15031069.

    Gradient class: none (inherits :func:`~torchnlts.entropy.sample_entropy`'s).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Non-empty iterable of positive integers; ``scales=[1]`` recovers plain ``sample_entropy``.
        radius: Tolerance r, forwarded unchanged to every offset and scale.
        embedding_dim: Template length m, forwarded to ``sample_entropy``.
        delay: Embedding lag, forwarded to ``sample_entropy``.
        theiler_window: Forwarded to ``sample_entropy``.
        norm_order: Forwarded to ``sample_entropy``.

    Returns:
        Entropy curve with the shape convention of :func:`~torchnlts.entropy.multiscale_entropy`. The offsets are
        combined by a plain mean, so an entry is ``NaN``/``inf`` if any offset's sample entropy is (as in Wu et al.
        and ComplexityMeasures.jl); a row containing ``NaN`` or ``inf`` is ``NaN`` at every scale. Each offset keeps its
        natural length ``(T - offset) // scale`` (as in the paper), so offsets can differ by one window.

    Raises:
        ValueError: If ``scales`` is empty or holds a non-integer or a scale ``< 1``, ``embedding_dim`` or ``delay`` is
            not a positive integer, ``theiler_window`` is not an integer ``>= 0``, or any offset's coarse-grained
            series is too short to embed (fewer than ``embedding_dim * delay + 1`` samples).
    """
    return _composite_curve(
        trajectory,
        scales,
        lambda series: sample_entropy(
            series,
            radius,
            embedding_dim=embedding_dim,
            delay=delay,
            theiler_window=theiler_window,
            norm_order=norm_order,
        ),
    )


def soft_composite_multiscale_entropy(
    trajectory,
    scales,
    temperature,
    *,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted composite multiscale entropy, a smooth stand-in for
    :func:`~torchnlts.entropy.composite_multiscale_entropy`.

    Gradient class: smooth (inherits :func:`~torchnlts.entropy.soft_sample_entropy`'s, including its Chebyshev-tie
    caveat).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Non-empty iterable of positive integers.
        temperature: Kernel bandwidth, forwarded to ``soft_sample_entropy``.
        embedding_dim: Template length m, forwarded to ``soft_sample_entropy``.
        delay: Embedding lag, forwarded to ``soft_sample_entropy``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_sample_entropy``.
        theiler_window: Forwarded to ``soft_sample_entropy``.
        norm_order: Forwarded to ``soft_sample_entropy``.

    Returns:
        Entropy curve, with the shape convention and offset averaging of
        :func:`~torchnlts.entropy.composite_multiscale_entropy`.

    Raises:
        ValueError: As :func:`~torchnlts.entropy.composite_multiscale_entropy`.
    """
    return _composite_curve(
        trajectory,
        scales,
        lambda series: soft_sample_entropy(
            series,
            temperature,
            embedding_dim=embedding_dim,
            delay=delay,
            kind=kind,
            theiler_window=theiler_window,
            norm_order=norm_order,
        ),
    )
