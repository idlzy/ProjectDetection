"""Composable output and feature distillation, with global foreground normalization."""

import torch
import torch.distributed as distributed
from torch import nn

from .feature import FeatureDistillation
from .output import output_losses
from .regions import distillation_regions


class DistillationCriterion(nn.Module):
    def __init__(self, student_config, teacher_config):
        super().__init__()
        self.config = student_config
        self.options = student_config["train"]["distillation"]
        self.feature = FeatureDistillation(self.options.get("feature", {}), student_config, teacher_config)

    def forward(self, student, teacher, regions_targets, student_head, teacher_head,
                student_features, teacher_features, progress):
        if len(student) != len(teacher) or any(
            a["cls"].shape != b["cls"].shape for a, b in zip(student, teacher)
        ):
            raise ValueError("Teacher/student output grids differ; use matching pyramid sizes")
        regions = distillation_regions(student, regions_targets, self.config)
        zero = torch.nan_to_num(student[0]["cls"].float(), nan=0., posinf=0., neginf=0.).sum() * 0
        count = sum((entry[0].sum() for level in regions for entry in level), zero.detach().clone())
        normalizer = count.detach().clone()
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(normalizer)
            normalizer /= distributed.get_world_size()
        normalizer = normalizer.clamp_min(1)
        losses, stats = {}, {"valid_points": count.detach()}
        stats["positive_points"] = sum((entry[3] for level in regions for entry in level), zero.detach().clone())
        for component in ("output", "feature"):
            options = self.options.get(component, {})
            warmup = options.get("warmup_epochs", 0)
            ramp = min(max(progress / warmup, 0.0), 1.0) if warmup else 1.0
            stats[component + "_ramp"] = zero.detach().new_tensor(ramp)
            if not options.get("enabled", False):
                continue
            if component == "output":
                raw, quality = output_losses(student, teacher, regions, student_head, teacher_head, options)
                stats.update(quality)
                weights = {key: options.get(key, {}).get("weight", 0.1) for key in raw}
            else:
                raw = self.feature(student_features, teacher_features, regions)
                weights = {tap["name"]: tap.get("weight", 0.05) for tap in options["taps"]}
            for key, value in raw.items():
                name = "loss_kd_%s_%s" % (component, key)
                denominator = normalizer
                if component == "output":
                    denominator = stats[name + "_count"].detach().clone()
                    if distributed.is_available() and distributed.is_initialized():
                        distributed.all_reduce(denominator)
                        denominator /= distributed.get_world_size()
                    # Branch averaging and global_count/world_size may be < 1.
                    # Preserve positive fractional counts; guard only empty regions.
                    denominator = torch.where(denominator > 0, denominator,
                                              torch.ones_like(denominator))
                losses[name] = value / denominator * weights[key] * ramp
                stats[name + "_raw_sum"] = value.detach()
                stats[name + "_weighted_sum"] = value.detach() * weights[key] * ramp
        return losses, stats
