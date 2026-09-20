from __future__ import annotations

import math

import torch
from torch import nn

from ..layers import ModulatedDeformConv2dPack


def _normalization(kind, channels):
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        if channels % 32 != 0:
            raise ValueError("GroupNorm(32) requires channels divisible by 32")
        return nn.GroupNorm(32, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError("Unknown normalization: %s" % kind)


class _MultiLevelFeatureBlock(nn.Module):
    """Shared convolution weights with level-specific normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        levels,
        normalization,
        deformable=False,
        deform_groups=1,
    ):
        super().__init__()
        if deformable:
            self.convolution = ModulatedDeformConv2dPack(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                deform_groups=deform_groups,
                bias=normalization == "none",
            )
        else:
            self.convolution = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    bias=normalization == "none",
                ),
            )
        self.norms = nn.ModuleList(
            [_normalization(normalization, out_channels) for _ in range(levels)]
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, feature, level):
        feature = self.convolution(feature)
        feature = self.norms[level](feature)
        return self.activation(feature)


class _MultiLevelTower(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        levels=5,
        normalization="batch",
        dcn_on_last_conv=False,
        deform_groups=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for index in range(depth):
            self.blocks.append(
                _MultiLevelFeatureBlock(
                    in_channels,
                    out_channels,
                    levels,
                    normalization,
                    deformable=dcn_on_last_conv and index == depth - 1,
                    deform_groups=deform_groups,
                )
            )
            in_channels = out_channels

    def forward(self, feature, level):
        for block in self.blocks:
            feature = block(feature, level)
        return feature


class FCOS3DHead(nn.Module):
    """Lightweight multi-branch FCOS3D/PGDA prediction head.

    Classification and regression towers share convolution weights across
    pyramid levels while keeping normalization statistics level-specific.
    Regression groups predict image offset, depth, dimensions and yaw using
    the current framework's target conventions. Attribute chaining is optional.
    """

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
        attribute_chain=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.probabilistic_depth = probabilistic_depth
        self.geometric_depth = geometric_depth
        self.attribute_chain = attribute_chain
        self.num_levels = 5
        self.group_reg_dims = (2, 1, 3, 1)
        self.register_buffer(
            "depth_centers",
            torch.arange(0, depth_bin_max + depth_bin_unit, depth_bin_unit),
            persistent=False,
        )

        tower_options = {
            "levels": self.num_levels,
            "normalization": normalization,
            "dcn_on_last_conv": dcn_on_last_conv,
            "deform_groups": deform_groups,
        }
        self.cls_tower = _MultiLevelTower(
            in_channels,
            feat_channels,
            stacked_convs,
            **tower_options,
        )
        self.reg_tower = _MultiLevelTower(
            in_channels,
            feat_channels,
            stacked_convs,
            **tower_options,
        )

        branch_options = {
            "levels": self.num_levels,
            "normalization": normalization,
        }
        self.cls_branch = _MultiLevelTower(
            feat_channels, feat_channels, 1, **branch_options
        )
        self.reg_branches = nn.ModuleList(
            [
                _MultiLevelTower(
                    feat_channels,
                    feat_channels,
                    1,
                    **branch_options,
                )
                for index in range(len(self.group_reg_dims))
            ]
        )
        self.size_to_orientation = (
            nn.Conv2d(feat_channels, feat_channels, 1, bias=False)
            if attribute_chain else None
        )
        self.orientation_to_depth = (
            nn.Conv2d(feat_channels, feat_channels, 1, bias=False)
            if attribute_chain else None
        )
        self.direction_branch = _MultiLevelTower(
            feat_channels, feat_channels, 1, **branch_options
        )
        self.attribute_branch = _MultiLevelTower(
            feat_channels, feat_channels, 1, **branch_options
        )
        self.centerness_branch = _MultiLevelTower(
            feat_channels, in_channels, 1, **branch_options
        )

        self.conv_cls = nn.Conv2d(feat_channels, num_classes, 1)
        self.conv_regs = nn.ModuleList(
            [nn.Conv2d(feat_channels, dimension, 1) for dimension in self.group_reg_dims]
        )
        self.conv_dir = nn.Conv2d(feat_channels, 2, 1)
        self.conv_attr = nn.Conv2d(feat_channels, num_attrs, 1)
        self.conv_centerness = nn.Conv2d(in_channels, 1, 1)
        self.conv_depth_prob = (
            nn.Conv2d(feat_channels, len(self.depth_centers), 1)
            if probabilistic_depth
            else None
        )
        self.depth_fuse_logit = (
            nn.Parameter(torch.tensor(0.0)) if probabilistic_depth else None
        )
        self.conv_geo_weight = (
            nn.Conv2d(feat_channels, 1, 1) if geometric_depth else None
        )
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(self.num_levels)]
        )
        self._init_weights()
        if self.attribute_chain:
            nn.init.zeros_(self.size_to_orientation.weight)
            nn.init.zeros_(self.orientation_to_depth.weight)

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
        for module in self.modules():
            if isinstance(module, ModulatedDeformConv2dPack):
                nn.init.zeros_(module.conv_offset_mask.weight)
                nn.init.zeros_(module.conv_offset_mask.bias)
        nn.init.constant_(self.conv_cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        if len(features) != self.num_levels:
            raise ValueError("FCOS3DHead expects five pyramid feature levels")
        outputs = []
        for level, feature in enumerate(features):
            cls_feature = self.cls_tower(feature, level)
            reg_feature = self.reg_tower(feature, level)
            cls_branch = self.cls_branch(cls_feature, level)

            regression_features = [
                branch(reg_feature, level) for branch in self.reg_branches
            ]
            direction_input = reg_feature
            if self.attribute_chain:
                size_delta = self.size_to_orientation(regression_features[2])
                regression_features[3] = regression_features[3] + size_delta
                regression_features[1] = regression_features[1] + (
                    self.orientation_to_depth(regression_features[3])
                )
                direction_input = reg_feature + size_delta
            bbox_groups = [
                predictor(branch_feature)
                for predictor, branch_feature in zip(
                    self.conv_regs, regression_features
                )
            ]
            bbox = torch.cat(bbox_groups, dim=1) * self.scales[level]
            depth_feature = regression_features[1]

            outputs.append(
                {
                    "cls": self.conv_cls(cls_branch),
                    "bbox": bbox,
                    "direction": self.conv_dir(
                        self.direction_branch(direction_input, level)
                    ),
                    "attribute": self.conv_attr(
                        self.attribute_branch(cls_feature, level)
                    ),
                    "centerness": self.conv_centerness(
                        self.centerness_branch(reg_feature, level)
                    ),
                    "depth_logits": (
                        self.conv_depth_prob(depth_feature)
                        if self.conv_depth_prob is not None
                        else None
                    ),
                    "geo_weight": (
                        self.conv_geo_weight(depth_feature)
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
