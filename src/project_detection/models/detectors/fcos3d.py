from __future__ import annotations

from torch import nn

from ..backbones import (
    EfficientNetB0HatCompatible,
    LegacyHatEfficientNetB0,
    ResNet101FCOS3D,
)
from ..heads import FCOS3DHead, LegacyHatFCOS3DHead
from ..necks import BiFPN, FPN, LegacyHatBiFPN


class FCOS3D(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="efficientnet_b0_hat_compatible",
        neck="bifpn",
        neck_channels=64,
        bifpn_stacks=3,
        head_channels=256,
        stacked_convs=2,
        num_attrs=9,
        probabilistic_depth=True,
        geometric_depth=False,
        depth_bin_unit=10.0,
        depth_bin_max=80.0,
        backbone_dcn_stages=(False, False, False, False),
        backbone_frozen_stages=-1,
        backbone_norm_eval=False,
        head_norm="batch",
        dcn_on_last_conv=False,
        deform_groups=1,
    ):
        super().__init__()
        if backbone == "efficientnet_b0_hat_compatible":
            self.backbone = EfficientNetB0HatCompatible()
        elif backbone == "legacy_hat_efficientnet_b0":
            self.backbone = LegacyHatEfficientNetB0()
        elif backbone == "resnet101_fcos3d":
            self.backbone = ResNet101FCOS3D(
                dcn_stages=backbone_dcn_stages,
                deform_groups=deform_groups,
                frozen_stages=backbone_frozen_stages,
                norm_eval=backbone_norm_eval,
            )
        else:
            raise ValueError("Unknown backbone: %s" % backbone)

        if neck == "bifpn":
            self.neck = BiFPN(
                self.backbone.out_channels, neck_channels, bifpn_stacks
            )
        elif neck == "legacy_hat_bifpn":
            self.neck = LegacyHatBiFPN(
                stacks=bifpn_stacks, out_channels=neck_channels
            )
        elif neck == "fpn":
            self.neck = FPN(self.backbone.out_channels, neck_channels)
        else:
            raise ValueError("Unknown neck: %s" % neck)
        if backbone == "legacy_hat_efficientnet_b0":
            self.head = LegacyHatFCOS3DHead(
                num_classes=num_classes,
                in_channels=neck_channels,
                feat_channels=head_channels,
                stacked_convs=stacked_convs,
                num_attrs=num_attrs,
                probabilistic_depth=probabilistic_depth,
                depth_bin_unit=depth_bin_unit,
                depth_bin_max=depth_bin_max,
            )
        else:
            self.head = FCOS3DHead(
                num_classes=num_classes,
                in_channels=neck_channels,
                feat_channels=head_channels,
                stacked_convs=stacked_convs,
                num_attrs=num_attrs,
                probabilistic_depth=probabilistic_depth,
                geometric_depth=geometric_depth,
                depth_bin_unit=depth_bin_unit,
                depth_bin_max=depth_bin_max,
                normalization=head_norm,
                dcn_on_last_conv=dcn_on_last_conv,
                deform_groups=deform_groups,
            )

    def forward(self, images):
        features = self.backbone(images)
        pyramid = self.neck(features)
        return self.head(pyramid)
