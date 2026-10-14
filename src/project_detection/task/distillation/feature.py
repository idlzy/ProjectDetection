"""Train-only student adapters and foreground feature distillation."""

import torch
import torch.nn.functional as functional
from torch import nn


class FeatureDistillation(nn.Module):
    def __init__(self, options, student_config, teacher_config):
        super().__init__()
        self.options = options
        self.adapters = nn.ModuleDict()
        for tap in options.get("taps", []) if options.get("enabled", False) else []:
            name = tap["name"]
            key = "neck_channels" if name == "neck_pyramid" else "head_channels"
            sc, tc = student_config["model"][key], teacher_config["model"][key]
            if tap.get("adapter", "identity") == "identity":
                if sc != tc:
                    raise ValueError("identity adapter requires equal channels: %s" % name)
                self.adapters[name] = nn.Identity()
            else:
                self.adapters[name] = nn.Conv2d(sc, tc, 1)

    def forward(self, student, teacher, regions):
        losses = {}
        levels = self.options.get("levels", ["p3", "p4", "p5", "p6", "p7"])
        for tap in self.options.get("taps", []):
            name = tap["name"]
            loss = student[name][0].float().sum() * 0
            for index, (sf, tf) in enumerate(zip(student[name], teacher[name])):
                if "p%d" % (index + 3) not in levels:
                    continue
                sf = self.adapters[name](sf).float()
                tf = tf.float()
                if sf.shape != tf.shape:
                    raise ValueError("Feature grid mismatch for %s" % name)
                if tap.get("normalization", "channel_l2") == "channel_l2":
                    sf, tf = functional.normalize(sf, dim=1), functional.normalize(tf, dim=1)
                error = (sf - tf).square().mean(1).flatten(1)
                masks = torch.stack([entry[0] for entry in regions[index]])
                loss = loss + error[masks].sum()
            losses[name] = loss
        return losses
