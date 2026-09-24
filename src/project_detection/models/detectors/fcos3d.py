from __future__ import annotations

import torch
from torch import nn

from ..backbones import (
    EfficientNetB0,
    LegacyHatEfficientNetB0,
    ResNet101FCOS3D,
)
from ..heads import FCOS3DHead, LegacyHatFCOS3DHead
from ..necks import BiFPN, FPN, LegacyHatBiFPN


class FCOS3D(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="efficientnet_b0",
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
        attribute_prediction_mode="parallel",
        chain_reliability_threshold=0.2,
        camera_conditioning=False,
    ):
        super().__init__()
        if camera_conditioning and backbone == "legacy_hat_efficientnet_b0":
            raise ValueError("Camera conditioning is not supported by the legacy HAT head")
        self.camera_conditioning = camera_conditioning
        if attribute_prediction_mode != "parallel" and backbone == "legacy_hat_efficientnet_b0":
            raise ValueError("Legacy HAT supports only parallel attribute prediction")
        if backbone == "efficientnet_b0":
            self.backbone = EfficientNetB0()
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
                attribute_prediction_mode=attribute_prediction_mode,
                chain_reliability_threshold=chain_reliability_threshold,
                camera_conditioning=camera_conditioning,
            )

    def forward(self, images, camera_matrices=None):
        features = self.backbone(images)
        pyramid = self.neck(features)
        if not self.camera_conditioning:
            return self.head(pyramid)
        if camera_matrices is None:
            raise ValueError("camera_matrices are required when camera conditioning is enabled")
        if camera_matrices.shape != (images.shape[0], 3, 3):
            raise ValueError("camera_matrices must have shape [batch, 3, 3]")
        camera_matrices = camera_matrices.to(device=images.device, dtype=images.dtype)
        focal_x = camera_matrices[:, 0, 0] / images.shape[-1]
        focal_y = camera_matrices[:, 1, 1] / images.shape[-2]
        if not bool(torch.isfinite(camera_matrices).all()) or bool((focal_x <= 0).any()) or bool((focal_y <= 0).any()):
            raise ValueError("camera_matrices must be finite with positive focal lengths")
        camera_features = torch.stack(
            (
                focal_x.log(),
                focal_y.log(),
                camera_matrices[:, 0, 2] / images.shape[-1],
                camera_matrices[:, 1, 2] / images.shape[-2],
            ),
            dim=1,
        )
        return self.head(pyramid, camera_features)
