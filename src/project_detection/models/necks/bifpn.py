from __future__ import annotations

import torch.nn.functional as functional
from torch import nn


class ChannelProjection(nn.Sequential):
    """Project a feature without changing its activation distribution."""

    def __init__(self, in_channels, out_channels):
        if in_channels == out_channels:
            super().__init__(nn.Identity())
        else:
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )


class SeparableRefinement(nn.Sequential):
    """ReLU followed by depthwise/pointwise feature refinement."""

    def __init__(self, channels):
        super().__init__(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )


def _upsample_like(source, reference):
    return functional.interpolate(
        source,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _downsample_like(source, reference):
    if min(source.shape[-2:]) >= 2:
        result = functional.max_pool2d(source, kernel_size=2, stride=2)
    else:
        result = source
    if result.shape[-2:] != reference.shape[-2:]:
        result = functional.interpolate(
            result, size=reference.shape[-2:], mode="nearest"
        )
    return result


def _pyramid_downsample(source):
    if min(source.shape[-2:]) >= 2:
        return functional.max_pool2d(source, kernel_size=2, stride=2)
    return source


class BiFPNLayer(nn.Module):
    """One unweighted bidirectional feature-pyramid layer.

    Each incoming edge owns its channel projection while all fused nodes use
    inexpensive depthwise-separable refinement. Explicit target sizes retain
    the reference graph's behavior and also support odd input shapes.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if len(in_channels) != 5:
            raise ValueError("BiFPNLayer expects five input feature levels")

        self.top_projections = nn.ModuleList(
            [ChannelProjection(channels, out_channels) for channels in in_channels[:4]]
        )
        self.bottom_projections = nn.ModuleList(
            [ChannelProjection(channels, out_channels) for channels in in_channels[1:4]]
        )
        self.p7_projection = ChannelProjection(in_channels[4], out_channels)
        self.top_refinements = nn.ModuleList(
            [SeparableRefinement(out_channels) for _ in range(4)]
        )
        self.bottom_refinements = nn.ModuleList(
            [SeparableRefinement(out_channels) for _ in range(4)]
        )

    def forward(self, features):
        p3, p4, p5, p6, p7 = features
        p7 = self.p7_projection(p7)

        top6_input = self.top_projections[3](p6)
        top6 = self.top_refinements[0](
            top6_input + _upsample_like(p7, top6_input)
        )
        top5_input = self.top_projections[2](p5)
        top5 = self.top_refinements[1](
            top5_input + _upsample_like(top6, top5_input)
        )
        top4_input = self.top_projections[1](p4)
        top4 = self.top_refinements[2](
            top4_input + _upsample_like(top5, top4_input)
        )
        top3_input = self.top_projections[0](p3)
        top3 = self.top_refinements[3](
            top3_input + _upsample_like(top4, top3_input)
        )

        out4 = self.bottom_refinements[0](
            self.bottom_projections[0](p4)
            + top4
            + _downsample_like(top3, top4)
        )
        out5 = self.bottom_refinements[1](
            self.bottom_projections[1](p5)
            + top5
            + _downsample_like(out4, top5)
        )
        out6 = self.bottom_refinements[2](
            self.bottom_projections[2](p6)
            + top6
            + _downsample_like(out5, top6)
        )
        out7 = self.bottom_refinements[3](
            p7 + _downsample_like(out6, p7)
        )
        return [top3, out4, out5, out6, out7]


class BiFPN(nn.Module):
    """Five-level lightweight bidirectional feature pyramid."""

    def __init__(self, in_channels, out_channels=64, stacks=3):
        super().__init__()
        selected_channels = list(in_channels[2:5])
        if len(selected_channels) != 3:
            raise ValueError("BiFPN expects at least five backbone outputs")

        self.p6_projection = ChannelProjection(
            selected_channels[-1], out_channels
        )
        layer_channels = [*selected_channels, out_channels, out_channels]
        self.layers = nn.ModuleList()
        for _ in range(stacks):
            self.layers.append(BiFPNLayer(layer_channels, out_channels))
            layer_channels = [out_channels] * 5

    def forward(self, inputs):
        p3, p4, p5 = inputs[2:5]
        p6 = _pyramid_downsample(self.p6_projection(p5))
        p7 = _pyramid_downsample(p6)
        features = [p3, p4, p5, p6, p7]
        for layer in self.layers:
            features = layer(features)
        return features
