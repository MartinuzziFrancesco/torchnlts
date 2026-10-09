"""Recurrence plots and recurrence quantification analysis."""

from torchnlts.recurrence.cross_recurrence import (
    hard_cross_recurrence_matrix,
    soft_cross_recurrence_matrix,
)
from torchnlts.recurrence.joint_recurrence import (
    hard_joint_recurrence_matrix,
    soft_joint_recurrence_matrix,
)
from torchnlts.recurrence.matrix import hard_recurrence_matrix, soft_recurrence_matrix
from torchnlts.recurrence.rqa import (
    recurrence_rate,
    determinism,
    dl_average,
    dl_max,
    dl_entropy,
    laminarity,
    trapping_time,
    vl_max,
    vl_entropy,
    rt_entropy,
    ratio,
    divergence,
    windowed_rqa,
)

__all__ = [
    "hard_cross_recurrence_matrix",
    "soft_cross_recurrence_matrix",
    "hard_joint_recurrence_matrix",
    "soft_joint_recurrence_matrix",
    "hard_recurrence_matrix",
    "soft_recurrence_matrix",
    "recurrence_rate",
    "determinism",
    "dl_average",
    "dl_max",
    "dl_entropy",
    "laminarity",
    "trapping_time",
    "vl_max",
    "vl_entropy",
    "rt_entropy",
    "ratio",
    "divergence",
    "windowed_rqa",
]
