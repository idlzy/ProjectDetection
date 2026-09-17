from __future__ import annotations

from torch import nn

from ..layers import ConvBNAct


class MBConv(nn.Module):
    """EfficientNet block using the deployment-friendly ReLU/no-SE recipe."""

    def __init__(
        self, in_channels, out_channels, expand, stride, kernel_size
    ):
        super().__init__()
        hidden = in_channels * expand
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(in_channels, hidden, 1))
        layers += [
            ConvBNAct(
                hidden,
                hidden,
                kernel_size,
                stride,
                groups=hidden,
            ),
            ConvBNAct(hidden, out_channels, 1, activation=False),
        ]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 feature extractor using ReLU and no SE blocks.

    The stage kernels follow EfficientNet-B0's 3/5-pixel pattern.  Linear
    projection outputs and residual sums deliberately have no trailing ReLU.
    """

    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, 2)
        settings = [
            # kernel, expand, channels, repeats, first stride
            (3, 1, 16, 1, 1),
            (3, 6, 24, 2, 2),
            (5, 6, 40, 2, 2),
            (3, 6, 80, 3, 2),
            (5, 6, 112, 3, 1),
            (5, 6, 192, 4, 2),
            (3, 6, 320, 1, 1),
        ]
        stages, in_channels = [], 32
        for kernel, expand, out_channels, repeats, stride in settings:
            blocks = [
                MBConv(
                    in_channels,
                    out_channels,
                    expand,
                    stride,
                    kernel,
                )
            ]
            blocks += [
                MBConv(out_channels, out_channels, expand, 1, kernel)
                for _ in range(repeats - 1)
            ]
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = [16, 24, 40, 112, 320]

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (0, 1, 2, 4, 6):
                outputs.append(x)
        return outputs
