"""Matrix profile, motifs, discords and FLUSS segmentation."""

from torchnlts.matrixprofile.fluss import corrected_arc_curve, regime_boundaries
from torchnlts.matrixprofile.motifs import discords, motifs
from torchnlts.matrixprofile.profile import matrix_profile

__all__ = [
    "matrix_profile",
    "motifs",
    "discords",
    "corrected_arc_curve",
    "regime_boundaries",
]
