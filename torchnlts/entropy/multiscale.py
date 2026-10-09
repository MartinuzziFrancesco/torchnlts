"""Multiscale entropy: applying an entropy estimator across a range of
coarse-graining scales, not a new entropy statistic of its own.

Reference: Costa, Goldberger & Peng (2002), Phys. Rev. Lett. 89,
068102, "Multiscale entropy analysis of complex physiologic time
series", the (unnumbered) coarse-graining formula on p. 2 -- their
own eq. 1 is the Shannon entropy definition, not this. A single-scale
entropy (sample entropy at "scale 1") can rank a random, uncorrelated
signal as more "complex" than a correlated one with genuine multiscale
structure, simply because randomness maximizes short-range
irregularity. Their fix: coarse-grain the signal at scale ``s``
(non-overlapping length-``s`` window means) and compute sample entropy
on *each* coarse-grained series, producing a complexity *curve* over
scales instead of one number.

Built entirely on already-validated primitives -- :func:`~torchnlts.entropy.coarse_grain`
is the only new logic here (a length-changing reshape-and-mean, no
counting/thresholding of its own), and :func:`~torchnlts.entropy.multiscale_entropy`/
:func:`~torchnlts.entropy.soft_multiscale_entropy` are a Python loop applying
:func:`torchnlts.entropy.sample_entropy`/``soft_sample_entropy``
per scale and stacking the results -- so this module's tests validate
the coarse-graining formula and the loop's composition, not sample
entropy itself over again.
"""

from __future__ import annotations

import torch

from torchnlts._shape import _as_int, _ensure_batched, _output_dtype, _restore_shape_preserving, _work_dtype
from torchnlts.convert import as_tensor
from torchnlts.entropy.sample import sample_entropy, soft_sample_entropy


def coarse_grain(trajectory, scale: int, offset: int = 0) -> torch.Tensor:
    """Costa et al.'s regular coarse-graining: non-overlapping length-``scale``
    window means along the time axis.

    Gradient class: exact (a reshape + mean).

    Non-float input is converted to the default dtype; half precision is
    averaged in float32 and cast back.

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``. A
            tensor, numpy array, or pandas Series/DataFrame.
        scale: Window length. ``scale=1`` returns ``trajectory``
            unchanged (mean of length-1 windows is the identity). Must
            be a positive integer.
        offset: Number of leading samples skipped before windowing
            (Wu et al. (2013)'s ``k - 1``), ``0 <= offset < scale``.

    Returns:
        Coarse-grained trajectory, same shape convention as
        ``trajectory`` (``[T']``, ``[T', D]``, or ``[B, T', D]``) with
        ``T' = (T - offset) // scale``, trailing ``(T - offset) % scale`` samples dropped --
        unlike :func:`torchnlts.embedding.delay_embedding`, the
        feature axis keeps its original meaning here (still ``D``
        channels, not a new ``embedding_dim``-sized axis), so a scalar
        ``[T]`` input stays ``[T']``, not ``[T', 1]``.

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
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Sample entropy at each of a range of coarse-graining scales.

    Gradient class: none (inherits :func:`torchnlts.entropy.sample_entropy`'s).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Iterable of positive integers, the coarse-graining
            scales to sweep (``scales=[1]`` recovers plain
            ``sample_entropy`` on the original series).
        radius: Tolerance r, forwarded to ``sample_entropy`` unchanged
            at every scale -- per Costa et al., r is fixed relative to
            the *original* series (commonly ``0.15 * trajectory.std()``,
            the caller's responsibility to compute, matching
            ``sample_entropy``'s own un-normalized ``radius``
            convention), not recomputed per coarse-grained scale.
        embedding_dim: Template length m, forwarded to ``sample_entropy``.
        delay: Embedding lag, forwarded to ``sample_entropy``.
        theiler_window: Forwarded to ``sample_entropy``.
        norm_order: Forwarded to ``sample_entropy``.

    Returns:
        Entropy curve, shape ``[num_scales]`` prepended onto whatever
        shape a single ``sample_entropy`` call would return (``[]``/
        ``[R]`` unbatched, ``[B]``/``[R, B]`` batched) -- e.g.
        ``[num_scales, B]`` for a batched trajectory with a scalar
        ``radius``. A scale that still leaves enough coarse-grained
        points for the requested embedding, but too few surviving
        matches, follows ``sample_entropy``'s own ``NaN``/``inf``
        convention for that entry (see its own Returns docstring). A
        scale coarse enough that the coarse-grained series itself is
        *too short* to embed at all -- fewer than ``embedding_dim + 1``
        points after :func:`~torchnlts.entropy.coarse_grain` -- raises instead (see
        Raises), aborting the whole curve rather than degrading that
        one entry to ``NaN``; multiscale sweeps are far more likely to
        hit this than a single ``sample_entropy`` call, since large
        scales are exactly what shrinks the series that much.

    Raises:
        ValueError: If any ``scale`` in ``scales`` leaves the
            coarse-grained trajectory shorter than ``embedding_dim + 1``
            samples (propagated from
            :func:`torchnlts.embedding.delay_embedding` via
            ``sample_entropy``), or if any ``scale < 1`` (from
            :func:`~torchnlts.entropy.coarse_grain`).
    """
    return torch.stack(
        [
            sample_entropy(coarse_grain(trajectory, scale), radius, embedding_dim, delay, theiler_window, norm_order)
            for scale in scales
        ]
    )


def soft_multiscale_entropy(
    trajectory,
    scales,
    temperature,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted multiscale entropy, a smooth stand-in for :func:`~torchnlts.entropy.multiscale_entropy`.

    Gradient class: smooth (inherits :func:`torchnlts.entropy.soft_sample_entropy`'s,
    including its Chebyshev-tie caveat).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Iterable of positive integers, forwarded to :func:`~torchnlts.entropy.coarse_grain`.
        temperature: Kernel bandwidth, forwarded to ``soft_sample_entropy``
            unchanged at every scale (see :func:`~torchnlts.entropy.multiscale_entropy`'s
            ``radius`` for why).
        embedding_dim: Template length m, forwarded to ``soft_sample_entropy``.
        delay: Embedding lag, forwarded to ``soft_sample_entropy``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_sample_entropy``.
        theiler_window: Forwarded to ``soft_sample_entropy``.
        norm_order: Forwarded to ``soft_sample_entropy``.

    Returns:
        Entropy curve, same shape convention as :func:`~torchnlts.entropy.multiscale_entropy`.

    Raises:
        ValueError: Same conditions as :func:`~torchnlts.entropy.multiscale_entropy`'s own
            Raises (a too-coarse scale, or ``scale < 1``).
    """
    return torch.stack(
        [
            soft_sample_entropy(
                coarse_grain(trajectory, scale), temperature, embedding_dim, delay, kind, theiler_window, norm_order
            )
            for scale in scales
        ]
    )


def _composite_curve(trajectory, scales, entropy) -> torch.Tensor:
    """Average ``entropy`` over all window offsets at each scale (half input is computed in float32)."""
    try:
        scales = [_as_int(scale, "scale") for scale in scales]
    except TypeError:
        raise ValueError(f"scales must be an iterable of positive integers, got {scales!r}") from None
    if not scales or any(scale < 1 for scale in scales):
        raise ValueError(f"scales must be a non-empty iterable of positive integers, got {scales!r}")
    trajectory = as_tensor(trajectory)
    out_dtype = _output_dtype(trajectory)
    trajectory = trajectory.to(_work_dtype(trajectory.dtype))
    curve = torch.stack(
        [
            torch.stack([entropy(coarse_grain(trajectory, scale, offset)) for offset in range(scale)]).mean(dim=0)
            for scale in scales
        ]
    )
    return curve.to(out_dtype)


def composite_multiscale_entropy(
    trajectory,
    scales,
    radius,
    embedding_dim: int = 2,
    delay: int = 1,
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Composite multiscale entropy: sample entropy averaged over all window offsets at each scale.

    Reference: Wu, Wu, Lin, Wang & Lee (2013), "Time series analysis using
    composite multiscale entropy", Entropy 15(3), 1069-1084,
    doi:10.3390/e15031069. At scale ``s`` the ``s`` coarse-grained series
    ``coarse_grain(trajectory, s, offset)``, ``offset = 0..s-1``, each get a
    sample entropy, and the ``s`` values are averaged.

    Gradient class: none (inherits :func:`torchnlts.entropy.sample_entropy`'s).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Iterable of positive integers (``scales=[1]`` recovers
            plain ``sample_entropy``).
        radius: Tolerance r, forwarded unchanged to every offset and scale.
        embedding_dim: Template length m, forwarded to ``sample_entropy``.
        delay: Embedding lag, forwarded to ``sample_entropy``.
        theiler_window: Forwarded to ``sample_entropy``.
        norm_order: Forwarded to ``sample_entropy``.

    Returns:
        Entropy curve with the same shape convention as
        :func:`~torchnlts.entropy.multiscale_entropy`. The per-offset values are combined by a
        plain mean, so an entry is ``NaN``/``inf`` if any offset's sample
        entropy is (Wu et al. average without exclusion; the same holds in
        ComplexityMeasures.jl). Offsets keep their natural lengths
        ``(T - offset) // scale`` (as in the paper), so they can differ by
        one window.

    Raises:
        ValueError: If ``scales`` is empty or holds a non-positive or
            non-integer scale, or if any offset's coarse-grained series is
            too short to embed (fewer than ``embedding_dim * delay + 1`` samples).
    """
    return _composite_curve(
        trajectory,
        scales,
        lambda series: sample_entropy(series, radius, embedding_dim, delay, theiler_window, norm_order),
    )


def soft_composite_multiscale_entropy(
    trajectory,
    scales,
    temperature,
    embedding_dim: int = 2,
    delay: int = 1,
    kind: str = "gaussian",
    theiler_window: int = 0,
    norm_order: float = float("inf"),
) -> torch.Tensor:
    """Kernel-weighted composite multiscale entropy, a smooth stand-in for :func:`~torchnlts.entropy.composite_multiscale_entropy`.

    Gradient class: smooth (inherits :func:`torchnlts.entropy.soft_sample_entropy`'s,
    including its Chebyshev-tie caveat).

    Args:
        trajectory: Shape ``[T]``, ``[T, D]``, or ``[B, T, D]``.
        scales: Iterable of positive integers.
        temperature: Kernel bandwidth, forwarded to ``soft_sample_entropy``.
        embedding_dim: Template length m, forwarded to ``soft_sample_entropy``.
        delay: Embedding lag, forwarded to ``soft_sample_entropy``.
        kind: ``"gaussian"`` or ``"exponential"``, forwarded to ``soft_sample_entropy``.
        theiler_window: Forwarded to ``soft_sample_entropy``.
        norm_order: Forwarded to ``soft_sample_entropy``.

    Returns:
        Entropy curve, same shape convention and offset averaging as
        :func:`~torchnlts.entropy.composite_multiscale_entropy`.

    Raises:
        ValueError: Same conditions as :func:`~torchnlts.entropy.composite_multiscale_entropy`.
    """
    return _composite_curve(
        trajectory,
        scales,
        lambda series: soft_sample_entropy(series, temperature, embedding_dim, delay, kind, theiler_window, norm_order),
    )
