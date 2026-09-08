from __future__ import annotations

from torch import nn

from ..layers import ConvBNAct


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand, stride):
        super().__init__()
        hidden = in_channels * expand
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(in_channels, hidden, 1))
        layers += [
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden),
            ConvBNAct(hidden, out_channels, 1, activation=False),
        ]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return self.activation(out)


class EfficientNetB0HatCompatible(nn.Module):
    """EfficientNet-B0 topology with HAT recipe changes: ReLU and no SE."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, 2)
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]
        stages, in_channels = [], 32
        for expand, out_channels, repeats, stride in settings:
            blocks = [MBConv(in_channels, out_channels, expand, stride)]
            blocks += [
                MBConv(out_channels, out_channels, expand, 1)
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
