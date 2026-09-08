import math

import torch
from torch import nn

try:
    from torchvision.ops import deform_conv2d
except (ImportError, RuntimeError):
    deform_conv2d = None


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class ModulatedDeformConv2dPack(nn.Module):
    """DCNv2 layer with an internal zero-initialized offset/mask predictor."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        kernel_h, kernel_w = _pair(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_h, kernel_w)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        offset_channels = deform_groups * 3 * kernel_h * kernel_w
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            offset_channels,
            (kernel_h, kernel_w),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        if deform_conv2d is None:
            raise RuntimeError(
                "torchvision deform_conv2d is unavailable; install a torchvision build matching torch"
            )
        offset_mask = self.conv_offset_mask(x)
        offset_channels = offset_mask.shape[1] * 2 // 3
        offset = offset_mask[:, :offset_channels]
        mask = offset_mask[:, offset_channels:].sigmoid()
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, activation=True, normalization="batch",
                 deformable=False, deform_groups=1, bias=None):
        padding = kernel_size // 2
        if bias is None:
            bias = normalization == "none"
        convolution = (
            ModulatedDeformConv2dPack(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, deform_groups=deform_groups, bias=bias,
            )
            if deformable
            else nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=bias,
            )
        )
        layers = [convolution]
        if normalization == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif normalization == "group":
            if out_channels % 32 != 0:
                raise ValueError("GroupNorm(32) requires channels divisible by 32")
            layers.append(nn.GroupNorm(32, out_channels))
        elif normalization != "none":
            raise ValueError("Unknown normalization: %s" % normalization)
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBNAct(ConvNormAct):
    """Convolution followed by batch normalization and optional ReLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        activation=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            activation,
            normalization="batch",
            bias=False,
        )
