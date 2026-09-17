"""Float PyTorch reproduction of the legacy HAT EfficientNet-B0 backbone.

The module hierarchy intentionally mirrors HAT 1.8 checkpoints.  It contains
no Horizon runtime dependency; the duplicated ``mod2``/``_blocks_conv`` names
are retained because old FX checkpoints stored both aliases.
"""

from __future__ import annotations

from collections import namedtuple

from torch import nn


BlockArgs = namedtuple(
    "BlockArgs",
    "kernel_size num_repeat in_filters out_filters expand_ratio id_skip stride",
)


def _conv_bn_act(
    in_channels, out_channels, kernel_size, stride=1, padding=0,
    groups=1, bias=False, activation=True,
):
    layers = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class LegacyHatMBConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.in_filters * args.expand_ratio
        if args.expand_ratio != 1:
            self._expand_conv = _conv_bn_act(
                args.in_filters, hidden, 1, activation=True
            )
        self._depthwise_conv = _conv_bn_act(
            hidden,
            hidden,
            args.kernel_size,
            stride=args.stride,
            padding=args.kernel_size // 2,
            groups=hidden,
            activation=True,
        )
        self._project_conv = _conv_bn_act(
            hidden, args.out_filters, 1, activation=False
        )
        self.use_shortcut = (
            args.id_skip
            and args.stride == 1
            and args.in_filters == args.out_filters
        )

    def forward(self, inputs):
        x = inputs
        if hasattr(self, "_expand_conv"):
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        x = self._project_conv(x)
        if self.use_shortcut:
            x = x + inputs
        return x


class LegacyHatEfficientNetB0(nn.Module):
    """HAT EfficientNet-B0 (ReLU, no SE, no classifier)."""

    out_channels = [16, 24, 40, 112, 320]

    def __init__(self):
        super().__init__()
        self.mod1 = nn.Sequential(_conv_bn_act(3, 32, 3, 2, 1))
        settings = (
            (3, 1, 32, 16, 1, True, 1),
            (3, 2, 16, 24, 6, True, 2),
            (5, 2, 24, 40, 6, True, 2),
            (3, 3, 40, 80, 6, True, 2),
            (5, 3, 80, 112, 6, True, 1),
            (5, 4, 112, 192, 6, True, 2),
            (3, 1, 192, 320, 6, True, 1),
        )
        blocks = []
        self.stage_divider = []
        for raw in settings:
            args = BlockArgs(*raw)
            blocks.append(LegacyHatMBConv(args))
            if args.stride == 2:
                self.stage_divider.append(len(blocks) - 1)
            for _ in range(args.num_repeat - 1):
                blocks.append(
                    LegacyHatMBConv(args._replace(in_filters=args.out_filters, stride=1))
                )
        self.stage_divider.append(len(blocks))
        self._blocks_conv = nn.ModuleList(blocks)
        # HAT registers the same ModuleList twice. Keep that state-dict layout
        # so the original checkpoint can be loaded with strict=True.
        self.mod2 = self._blocks_conv

    def forward(self, inputs):
        x = self.mod1(inputs)
        outputs = []
        for index, block in enumerate(self._blocks_conv):
            x = block(x)
            if index + 1 in self.stage_divider:
                outputs.append(x)
        return outputs
