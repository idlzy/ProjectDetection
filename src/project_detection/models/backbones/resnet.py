from __future__ import annotations

from typing import Iterable

from torch import nn

from ..layers import ModulatedDeformConv2dPack


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        deformable=False,
        deform_groups=1,
        downsample=None,
    ):
        super().__init__()
        # FCOS3D uses the Caffe ResNet style: stage stride is placed on conv1.
        self.conv1 = nn.Conv2d(
            in_channels, channels, 1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        conv2 = {
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": dilation,
            "dilation": dilation,
            "bias": False,
        }
        self.conv2 = (
            ModulatedDeformConv2dPack(
                **conv2, deform_groups=deform_groups
            )
            if deformable
            else nn.Conv2d(**conv2)
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(
            channels, channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet101FCOS3D(nn.Module):
    """ResNet-101 backbone used by the public FCOS3D recipe.

    It uses the Caffe bottleneck stride placement and optionally replaces every
    3x3 convolution in selected stages with modulated deformable convolution.
    """

    stage_blocks = (3, 4, 23, 3)
    out_channels = [256, 512, 1024, 2048]

    def __init__(
        self,
        dcn_stages: Iterable[bool] = (False, False, False, False),
        deform_groups=1,
        frozen_stages=-1,
        norm_eval=False,
    ):
        super().__init__()
        dcn_stages = tuple(dcn_stages)
        if len(dcn_stages) != 4:
            raise ValueError("dcn_stages must contain four booleans")
        self.frozen_stages = int(frozen_stages)
        self.norm_eval = bool(norm_eval)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        channels = (64, 128, 256, 512)
        self.stages = nn.ModuleList(
            self._make_stage(
                channel,
                blocks,
                stride=1 if index == 0 else 2,
                deformable=dcn_stages[index],
                deform_groups=deform_groups,
            )
            for index, (channel, blocks) in enumerate(
                zip(channels, self.stage_blocks)
            )
        )
        self._init_weights()
        self._freeze_stages()

    def _make_stage(
        self, channels, blocks, stride, deformable, deform_groups
    ):
        out_channels = channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [
            Bottleneck(
                self.in_channels,
                channels,
                stride=stride,
                deformable=deformable,
                deform_groups=deform_groups,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels
        layers.extend(
            Bottleneck(
                self.in_channels,
                channels,
                deformable=deformable,
                deform_groups=deform_groups,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, ModulatedDeformConv2dPack):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.zeros_(module.conv_offset_mask.weight)
                nn.init.zeros_(module.conv_offset_mask.bias)
            elif isinstance(module, Bottleneck):
                nn.init.zeros_(module.bn3.weight)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for module in (self.conv1, self.bn1):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False
        for index in range(min(self.frozen_stages, len(self.stages))):
            stage = self.stages[index]
            stage.eval()
            for parameter in stage.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
