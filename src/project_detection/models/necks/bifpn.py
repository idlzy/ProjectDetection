from __future__ import annotations

import torch.nn.functional as functional
from torch import nn

from ..layers import ConvBNAct


class SeparableConv(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, channels, 1),
        )


class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.top_down = nn.ModuleList([SeparableConv(channels) for _ in range(4)])
        self.bottom_up = nn.ModuleList([SeparableConv(channels) for _ in range(4)])

    def forward(self, features):
        top = list(features)
        for index in range(3, -1, -1):
            up = functional.interpolate(
                top[index + 1], size=top[index].shape[-2:], mode="nearest"
            )
            top[index] = self.top_down[index](top[index] + up)
        out = list(top)
        for index in range(1, 5):
            down = functional.max_pool2d(
                out[index - 1], kernel_size=3, stride=2, padding=1
            )
            out[index] = self.bottom_up[index - 1](
                features[index] + top[index] + down
            )
        return out


class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels=64, stacks=3):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvBNAct(ch, out_channels, 1) for ch in in_channels[2:]]
        )
        self.p6 = ConvBNAct(in_channels[-1], out_channels, 3, 2)
        self.p7 = ConvBNAct(out_channels, out_channels, 3, 2)
        self.layers = nn.ModuleList(
            [BiFPNLayer(out_channels) for _ in range(stacks)]
        )

    def forward(self, inputs):
        p3, p4, p5 = [
            layer(value) for layer, value in zip(self.projections, inputs[2:])
        ]
        p6 = self.p6(inputs[-1])
        p7 = self.p7(p6)
        features = [p3, p4, p5, p6, p7]
        for layer in self.layers:
            features = layer(features)
        return features
