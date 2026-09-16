"""TrajectoryKernel: temperature-controlled kernel weights over distances.

Gradient class: **smooth**. Both kernels below are smooth (infinitely
differentiable) functions of ``distance`` and ``temperature`` everywhere,
including at ``distance == 0``, so there is no boundary to document --
unlike the hard primitives in :mod:`torchnlts.neighbors.knn` and
:mod:`torchnlts.neighbors.radius`, which this kernel exists to relax.

No Julia equivalent exists for this primitive: neither ChaosTools.jl nor
DynamicalSystems.jl implement a standalone kernel-weighting object (grepped
ChaosTools.jl's source for "kernel"/"ccm"/"crossmap": no matches). Instead,
this is validated against the two papers
that motivate its two kernel kinds directly, plus hand-derived analytic
values and a gradcheck.

- ``kind="gaussian"``: ``exp(-distance^2 / (2 * temperature^2))``, the
  smooth correlation-sum kernel of Judd, K. (1992), "An improved
  estimator of dimension and some comments on providing confidence
  intervals", Physica D 56(2-3), 216-228 -- a differentiable stand-in
  for the Grassberger-Procaccia (1983) Heaviside step, with
  ``temperature`` playing the role of the smoothing bandwidth.
- ``kind="exponential"``: ``exp(-distance / temperature)``, the
  neighbor-weighting kernel of Sugihara et al. (2012), "Detecting
  Causality in Complex Ecosystems", Science 338(6106), 496-500 (their
  convergent-cross-mapping weights ``u_i = exp(-d_i / d_1)``, recovered
  here by passing the nearest-neighbor distance ``d_1`` as a per-row
  ``temperature``).

Limitation: known ceiling -- ``distance`` and ``temperature`` both within
about one order of magnitude of float32's representable floor (~1.2e-38,
e.g. ``temperature=2e-38``) can still yield an infinite ``temperature``
gradient in float32: the division's backward needs ``temperature**2``,
which underflows to exactly 0 at that scale independent of how the
forward pass is arranged (confirmed: no reformulation avoids it here,
since the true gradient itself is only barely representable in float32's
~38-order dynamic range). float64 has ~308 orders of range and is
unaffected (verified: same inputs give the exact finite analytic
gradient). No real trajectory/bandwidth data lives within 1e-30 of zero,
so this is left as a documented float32 ceiling rather than chased with
log-space rescaling; upgrade path is float64 for that regime.
"""

from __future__ import annotations

import torch


def trajectory_kernel(distance: torch.Tensor, temperature: torch.Tensor | float, kind: str = "gaussian") -> torch.Tensor:
    """Kernel weights over a distance matrix.

    Args:
        distance: Non-negative distances, any shape (typically the
            output of :func:`torchnlts.neighbors.distance.pairwise_distance`).
        temperature: Bandwidth, either a scalar or a tensor broadcastable
            against ``distance`` (e.g. one bandwidth per row, as CCM's
            per-point nearest-neighbor distance). Must be positive.
        kind: ``"gaussian"`` (Judd 1992) or ``"exponential"`` (Sugihara
            et al. 2012).

    Returns:
        Kernel weights, same shape as ``distance`` (broadcast with
        ``temperature``), in ``(0, 1]``.
    """
    if kind not in ("gaussian", "exponential"):
        raise ValueError(f"kind must be 'gaussian' or 'exponential', got {kind!r}")

    # Distances coming straight from apply_theiler_window / knn_search
    # padding are often inf. exp(-inf) is a correct 0 forward, but its
    # backward w.r.t. temperature is 0 * inf = NaN. Substitute a finite
    # placeholder before any arithmetic, then select the correct 0
    # weight afterward -- the substituted branch never contributes to
    # the gradient of the selected branch (see torch.where semantics).
    finite = torch.isfinite(distance)
    safe_distance = torch.where(finite, distance, torch.zeros_like(distance))

    # Plain division's own backward formula (-distance / temperature**2)
    # can overflow to +-inf whenever distance and temperature differ in
    # scale enough -- whether from a tiny temperature or a huge distance
    # -- and multiplying that by even an exactly-zero upstream gradient
    # (the kernel is already saturated at 0 out here) gives NaN
    # regardless of downstream masking; masking the *output* can't fix an
    # input-side operation's own backward. Instead floor temperature
    # *relative to distance* so the ratio the division produces can never
    # exceed RATIO_MAX: past that point the kernel has already underflowed
    # to exactly 0 in both float32 and float64, so clamping there changes
    # no forward value that wasn't already indistinguishable from 0, but
    # keeps every intermediate (and hence every gradient) finite. This
    # leaves the ratio, and thus the true kernel value, exactly unchanged
    # at any scale where it wasn't already saturated -- see
    # test_trajectory_kernel_gaussian_large_scale_no_overflow.
    RATIO_MAX = 1000.0
    floor = safe_distance.abs() / RATIO_MAX
    safe_temperature = torch.clamp(floor, min=temperature)

    if kind == "gaussian":
        # Divide before squaring so a shared large/small scale between
        # distance and temperature cancels instead of overflowing (or
        # underflowing) distance**2 and temperature**2 independently.
        ratio = safe_distance / safe_temperature
        weight = torch.exp(-(ratio**2) / 2)
    else:
        weight = torch.exp(-safe_distance / safe_temperature)
    return torch.where(finite, weight, torch.zeros_like(weight))
