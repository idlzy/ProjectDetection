"""Standard-PyTorch equivalent of the legacy HAT sum-BiFPN."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class LegacyMaybeApply1x1(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        if in_channels != out_channels:
            layers = [nn.Conv2d(in_channels, out_channels, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            self.lateral_conv = nn.ModuleList(layers)

    def forward(self, x):
        if hasattr(self, "lateral_conv"):
            for layer in self.lateral_conv:
                x = layer(x)
        return x


class LegacyResize(nn.Module):
    def __init__(self, sampling, in_channels, out_channels):
        super().__init__()
        self.sampling = sampling
        self.resize_layer = nn.ModuleList()
        if sampling == "down":
            lateral = LegacyMaybeApply1x1(in_channels, out_channels)
            if hasattr(lateral, "lateral_conv"):
                self.resize_layer.append(lateral)
            self.resize_layer.append(nn.MaxPool2d(2, 2))
        else:
            lateral = LegacyMaybeApply1x1(in_channels, out_channels)
            if hasattr(lateral, "lateral_conv"):
                self.resize_layer.append(lateral)

    def forward(self, x):
        for layer in self.resize_layer:
            x = layer(x)
        if self.sampling == "up":
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
        return x


_NODES = (
    ((3, 4), ("keep", "up")),
    ((2, 5), ("keep", "up")),
    ((1, 6), ("keep", "up")),
    ((0, 7), ("keep", "up")),
    ((1, 7, 8), ("keep", "keep", "down")),
    ((2, 6, 9), ("keep", "keep", "down")),
    ((3, 5, 10), ("keep", "keep", "down")),
    ((4, 11), ("keep", "down")),
)


class LegacyHatBiFPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        offset_channels = {
            0: out_channels,
            1: out_channels,
            2: out_channels,
            3: out_channels,
            4: out_channels,
            5: out_channels,
            6: out_channels,
            7: out_channels,
            8: out_channels,
            9: out_channels,
            10: out_channels,
            11: out_channels,
        }
        self.all_nodes = nn.ModuleDict()
        for node_index, (offsets, sampling_modes) in enumerate(_NODES):
            node = nn.ModuleList()
            for offset, sampling in zip(offsets, sampling_modes):
                source_channels = (
                    in_channels[offset] if offset < 5 else offset_channels[offset]
                )
                node.append(
                    LegacyResize(sampling, source_channels, out_channels)
                )
            # HAT stores Fusion here. Sum fusion has no parameters.
            node.append(_LegacySumFusion())
            node.append(
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                    ),
                )
            )
            self.all_nodes[str(node_index)] = node

    def forward(self, inputs):
        features = list(inputs)
        for node_index, (offsets, _) in enumerate(_NODES):
            node = self.all_nodes[str(node_index)]
            resized = [node[i](features[offset]) for i, offset in enumerate(offsets)]
            new_feature = node[len(offsets)](resized)
            new_feature = node[len(offsets) + 1](new_feature)
            features.append(new_feature)
        return features[-5:]


class _LegacySumFusion(nn.Module):
    def forward(self, inputs):
        result = inputs[0]
        for value in inputs[1:]:
            result = result + value
        return result


class LegacyHatBiFPN(nn.Module):
    """HAT BiFPN configuration used by the migrated 0903 checkpoint."""

    def __init__(self, stacks=3, out_channels=64):
        super().__init__()
        self.extra_downsamples = nn.ModuleList(
            [
                LegacyResize("down", 320, out_channels),
                LegacyResize("down", out_channels, out_channels),
            ]
        )
        self.bifpn_layers = nn.ModuleList()
        first_channels = [40, 112, 320, out_channels, out_channels]
        for index in range(stacks):
            channels = first_channels if index == 0 else [out_channels] * 5
            self.bifpn_layers.append(
                LegacyHatBiFPNLayer(channels, out_channels=out_channels)
            )

    def forward(self, inputs):
        features = list(inputs[2:5])
        for downsample in self.extra_downsamples:
            features.append(downsample(features[-1]))
        for layer in self.bifpn_layers:
            features = layer(features)
        return features
