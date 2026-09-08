from __future__ import annotations

import torch.nn.functional as functional
from torch import nn


class FPN(nn.Module):
    """FCOS3D FPN producing P3--P7 from ResNet C3--C5."""

    def __init__(self, in_channels, out_channels=256, start_level=1):
        super().__init__()
        selected_channels = in_channels[start_level:]
        if len(selected_channels) != 3:
            raise ValueError("FCOS3D FPN expects exactly three input levels")
        self.start_level = start_level
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, out_channels, 1)
            for channels in selected_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in selected_channels
        )
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs):
        laterals = [
            conv(feature)
            for conv, feature in zip(
                self.lateral_convs, inputs[self.start_level :]
            )
        ]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + functional.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:], mode="nearest"
            )
        outputs = [
            conv(feature) for conv, feature in zip(self.output_convs, laterals)
        ]
        outputs.append(self.p6(outputs[-1]))
        outputs.append(self.p7(functional.relu(outputs[-1])))
        return outputs
