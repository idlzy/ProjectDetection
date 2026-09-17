"""Dense prediction heads."""

from .fcos3d_head import FCOS3DHead
from .fcos3d_head_hat_legacy import LegacyHatFCOS3DHead

__all__ = ["FCOS3DHead", "LegacyHatFCOS3DHead"]
