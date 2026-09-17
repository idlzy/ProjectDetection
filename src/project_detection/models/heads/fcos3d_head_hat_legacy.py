"""Legacy HAT FCOS3D/PGDA head implemented with standard PyTorch ops."""

from __future__ import annotations

import torch
from torch import nn


class LegacyHatFCOS3DHead(nn.Module):
    """Reproduce the module graph used by HAT 1.8.2 float checkpoints."""

    legacy_hat_postprocess = True

    def __init__(
        self,
        num_classes=12,
        in_channels=64,
        feat_channels=256,
        stacked_convs=2,
        num_attrs=9,
        probabilistic_depth=True,
        depth_bin_unit=10.0,
        depth_bin_max=80.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_attrs = num_attrs
        self.probabilistic_depth = probabilistic_depth
        self.strides = [8, 16, 32, 64, 128]
        self.group_reg_dims = (2, 1, 3, 1, 2)

        self.cls_convs = self._init_shared_tower(stacked_convs)
        self.reg_convs = self._init_shared_tower(stacked_convs)
        self.conv_cls_prev = self._init_branch((256,))
        self.conv_cls = nn.Conv2d(256, num_classes, 1)

        branch_channels = ((256,), (256,), (256,), (256,), ())
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for channels, output_dim in zip(branch_channels, self.group_reg_dims):
            if channels:
                self.conv_reg_prevs.append(self._init_branch(channels))
                predictor_in = channels[-1]
            else:
                self.conv_reg_prevs.append(None)
                predictor_in = feat_channels
            self.conv_regs.append(nn.Conv2d(predictor_in, output_dim, 1))

        self.conv_dir_cls_prev = self._init_branch((256,))
        self.conv_dir_cls = nn.Conv2d(256, 2, 1)
        self.conv_attr_prev = self._init_branch((256,))
        self.conv_attr = nn.Conv2d(256, num_attrs, 1)
        self.conv_centerness_prev = self._init_branch((64,))
        self.conv_centerness = nn.Conv2d(64, 1, 1)

        bins = int(depth_bin_max // depth_bin_unit) + 1
        self.conv_depth_prob = (
            nn.Conv2d(256, bins, 1) if probabilistic_depth else None
        )
        self.depth_fuse_logit = (
            nn.Parameter(torch.zeros(1)) if probabilistic_depth else None
        )
        self.register_buffer(
            "depth_centers",
            torch.arange(bins, dtype=torch.float32) * depth_bin_unit,
            persistent=False,
        )

    def _separable_block(self, in_channels, out_channels, shared=None):
        if shared is None:
            depthwise = nn.Conv2d(
                in_channels, in_channels, 3, padding=1,
                groups=in_channels,
            )
            pointwise = nn.Conv2d(in_channels, out_channels, 1)
        else:
            depthwise, pointwise = shared[0], shared[1]
        return nn.Sequential(
            depthwise,
            pointwise,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _init_shared_tower(self, depth):
        towers = nn.ModuleList()
        for level in range(5):
            level_layers = nn.ModuleList()
            for index in range(depth):
                in_channels = self.in_channels if index == 0 else self.feat_channels
                shared = None if level == 0 else towers[0][index]
                level_layers.append(
                    self._separable_block(
                        in_channels, self.feat_channels, shared=shared
                    )
                )
            towers.append(level_layers)
        return towers

    def _init_branch(self, channels):
        branches = nn.ModuleList()
        all_channels = [self.feat_channels, *channels]
        for level in range(5):
            level_layers = nn.ModuleList()
            for index in range(len(channels)):
                shared = None if level == 0 else branches[0][index]
                level_layers.append(
                    self._separable_block(
                        all_channels[index], all_channels[index + 1], shared=shared
                    )
                )
            branches.append(level_layers)
        return branches

    @staticmethod
    def _apply_layers(x, layers):
        for layer in layers:
            x = layer(x)
        return x

    def forward(self, features):
        outputs = []
        for level, feature in enumerate(features):
            cls_feature = self._apply_layers(feature, self.cls_convs[level])
            reg_feature = self._apply_layers(feature, self.reg_convs[level])

            cls_branch = self._apply_layers(
                cls_feature, self.conv_cls_prev[level]
            )
            bbox_groups = []
            depth_feature = None
            for index, predictor in enumerate(self.conv_regs):
                branch = reg_feature
                previous = self.conv_reg_prevs[index]
                if previous is not None:
                    branch = self._apply_layers(branch, previous[level])
                bbox_groups.append(predictor(branch))
                if index == 1:
                    depth_feature = branch

            direction_feature = self._apply_layers(
                reg_feature, self.conv_dir_cls_prev[level]
            )
            attribute_feature = self._apply_layers(
                cls_feature, self.conv_attr_prev[level]
            )
            centerness_feature = self._apply_layers(
                reg_feature, self.conv_centerness_prev[level]
            )
            outputs.append(
                {
                    "cls": self.conv_cls(cls_branch),
                    "bbox": torch.cat(bbox_groups, dim=1),
                    "direction": self.conv_dir_cls(direction_feature),
                    "attribute": self.conv_attr(attribute_feature),
                    "centerness": self.conv_centerness(centerness_feature),
                    "depth_logits": (
                        self.conv_depth_prob(depth_feature)
                        if self.conv_depth_prob is not None
                        else None
                    ),
                    "geo_weight": None,
                }
            )
        return outputs

    def decode_depth_candidates(self, direct_raw, depth_logits=None):
        direct = direct_raw.exp()
        if depth_logits is None or self.depth_fuse_logit is None:
            return direct, torch.ones_like(direct)
        probabilities = depth_logits.softmax(dim=1)
        centers = self.depth_centers.to(depth_logits).view(1, -1)
        probabilistic = (probabilities * centers).sum(dim=1)
        weight = self.depth_fuse_logit.sigmoid()
        depth = weight * direct + (1.0 - weight) * probabilistic
        confidence = probabilities.topk(2, dim=1).values.mean(dim=1)
        return depth.clamp_min(0.1), confidence

    @staticmethod
    def fuse_geometric_depth(
        local_depth, geometric_depth, geo_weight, geometry_valid=None
    ):
        return local_depth
