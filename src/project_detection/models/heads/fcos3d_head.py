from __future__ import annotations

import math

import torch
from torch import nn

from ..layers import ConvNormAct, ModulatedDeformConv2dPack


def _tower(
    in_channels,
    out_channels,
    depth,
    normalization="batch",
    dcn_on_last_conv=False,
    deform_groups=1,
):
    layers = []
    for index in range(depth):
        layers.append(
            ConvNormAct(
                in_channels,
                out_channels,
                3,
                normalization=normalization,
                deformable=dcn_on_last_conv and index == depth - 1,
                deform_groups=deform_groups,
            )
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


class FCOS3DHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=64,
        feat_channels=256,
        stacked_convs=2,
        num_attrs=9,
        probabilistic_depth=True,
        geometric_depth=False,
        depth_bin_unit=10.0,
        depth_bin_max=80.0,
        normalization="batch",
        dcn_on_last_conv=False,
        deform_groups=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.probabilistic_depth = probabilistic_depth
        self.geometric_depth = geometric_depth
        self.depth_centers = torch.arange(
            0, depth_bin_max + depth_bin_unit, depth_bin_unit
        )
        tower_options = {
            "normalization": normalization,
            "dcn_on_last_conv": dcn_on_last_conv,
            "deform_groups": deform_groups,
        }
        self.cls_tower = _tower(
            in_channels, feat_channels, stacked_convs, **tower_options
        )
        self.reg_tower = _tower(
            in_channels, feat_channels, stacked_convs, **tower_options
        )
        self.conv_cls = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(feat_channels, 9, 3, padding=1)
        self.conv_dir = nn.Conv2d(feat_channels, 2, 3, padding=1)
        self.conv_attr = nn.Conv2d(feat_channels, num_attrs, 3, padding=1)
        self.conv_centerness = nn.Conv2d(feat_channels, 1, 3, padding=1)
        self.conv_depth_prob = (
            nn.Conv2d(feat_channels, len(self.depth_centers), 3, padding=1)
            if probabilistic_depth
            else None
        )
        self.depth_fuse_logit = (
            nn.Parameter(torch.tensor(0.0)) if probabilistic_depth else None
        )
        self.conv_geo_weight = (
            nn.Conv2d(feat_channels, 1, 3, padding=1)
            if geometric_depth
            else None
        )
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(5)]
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, ModulatedDeformConv2dPack):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Offset 0 and mask 0.5 make DCNv2 start as a regular sampled conv.
        for module in self.modules():
            if isinstance(module, ModulatedDeformConv2dPack):
                nn.init.zeros_(module.conv_offset_mask.weight)
                nn.init.zeros_(module.conv_offset_mask.bias)
        nn.init.constant_(self.conv_cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        outputs = []
        for level, feature in enumerate(features):
            cls_feature = self.cls_tower(feature)
            reg_feature = self.reg_tower(feature)
            outputs.append(
                {
                    "cls": self.conv_cls(cls_feature),
                    "bbox": self.conv_reg(reg_feature) * self.scales[level],
                    "direction": self.conv_dir(reg_feature),
                    "attribute": self.conv_attr(cls_feature),
                    "centerness": self.conv_centerness(reg_feature),
                    "depth_logits": (
                        self.conv_depth_prob(reg_feature)
                        if self.conv_depth_prob is not None
                        else None
                    ),
                    "geo_weight": (
                        self.conv_geo_weight(reg_feature)
                        if self.conv_geo_weight is not None
                        else None
                    ),
                }
            )
        return outputs

    def decode_depth(self, direct_raw, depth_logits=None):
        direct = direct_raw.exp()
        if depth_logits is None or self.depth_fuse_logit is None:
            return direct, None
        centers = self.depth_centers.to(depth_logits).view(1, -1, 1, 1)
        probabilities = depth_logits.softmax(dim=1)
        probabilistic = (probabilities * centers).sum(dim=1, keepdim=True)
        weight = self.depth_fuse_logit.sigmoid()
        depth = weight * direct + (1 - weight) * probabilistic
        confidence = probabilities.topk(2, dim=1).values.mean(
            dim=1, keepdim=True
        )
        return depth, confidence

    def decode_depth_candidates(self, direct_raw, depth_logits=None):
        """Decode flattened candidate depths with shape N and N x C."""
        direct = direct_raw.exp()
        if depth_logits is None or self.depth_fuse_logit is None:
            return direct, torch.ones_like(direct)
        centers = self.depth_centers.to(depth_logits).view(1, -1)
        probabilities = depth_logits.softmax(dim=1)
        probabilistic = (probabilities * centers).sum(dim=1)
        weight = self.depth_fuse_logit.sigmoid()
        local_depth = weight * direct + (1 - weight) * probabilistic
        confidence = probabilities.topk(2, dim=1).values.mean(dim=1)
        return local_depth, confidence

    def fuse_geometric_depth(
        self, local_depth, geometric_depth, geo_weight, geometry_valid=None
    ):
        if geo_weight is None:
            return local_depth
        weight = geo_weight.sigmoid()
        fused = weight * local_depth + (1 - weight) * geometric_depth
        if geometry_valid is not None:
            fused = torch.where(geometry_valid, fused, local_depth)
        return fused
