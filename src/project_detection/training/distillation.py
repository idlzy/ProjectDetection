"""Teacher lifecycle, input conversion and resumable distillation state."""

import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from ..data.preprocessing import normalize_image
from ..models import build_model, forward_with_targets
from ..task.distillation import DistillationCriterion
from ..task.distillation.output import DEPTH_OUTPUTS, OUTPUT_NAMES, output_enabled


def validate_distillation_config(config):
    options = config["train"].get("distillation", {})
    if not options:
        return
    if not isinstance(options, dict):
        raise ValueError("train.distillation must be a mapping")
    def boolean(mapping, key, default):
        if not isinstance(mapping.get(key, default), bool):
            raise ValueError("distillation.%s must be boolean" % key)
    boolean(options, "enabled", False)
    if not options.get("enabled", False):
        return
    allowed = {"enabled", "teacher", "region", "output", "feature"}
    if set(options) - allowed:
        raise ValueError("Unknown distillation keys: %s" % sorted(set(options) - allowed))
    for key in ("teacher", "region", "output", "feature"):
        if not isinstance(options.get(key, {}), dict):
            raise ValueError("distillation.%s must be a mapping" % key)
    teacher = options.get("teacher", {})
    if not teacher.get("checkpoint") or teacher.get("config_source", "checkpoint") != "checkpoint":
        raise ValueError("Distillation teacher requires checkpoint and config_source=checkpoint")
    if teacher.get("strict_load", True) is not True:
        raise ValueError("Distillation teacher requires strict_load=true")
    region = options.get("region", {})
    if region.get("mode", "gt_positive") != "gt_positive":
        raise ValueError("Only gt_positive distillation is supported")
    boolean(region, "exclude_padding", True)
    if config["model"]["backbone"].startswith("legacy_hat"):
        raise ValueError("Distillation requires the native FCOS3D model")
    active = False
    def number(value, name, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError("distillation.%s must be finite and %s" % (name, "positive" if positive else "nonnegative"))
    for name in ("output", "feature"):
        component = options.get(name, {})
        boolean(component, "enabled", False)
        number(component.get("warmup_epochs", 0), name + ".warmup_epochs")
        if not component.get("enabled", False):
            continue
        active = True
        if name == "output":
            allowed_output = set(OUTPUT_NAMES) | {"enabled", "branch_mode", "warmup_epochs", "quality"}
            if set(component) - allowed_output:
                raise ValueError("Unknown output distillation keys: %s" % sorted(set(component) - allowed_output))
            if any(output_enabled(component, key) for key in DEPTH_OUTPUTS) and not config["model"].get("probabilistic_depth", False):
                raise ValueError("Depth distillation requires probabilistic_depth=true")
            if component.get("branch_mode", "corresponding") != "corresponding":
                raise ValueError("Only corresponding output branches are supported")
            selected = False
            for key in OUTPUT_NAMES:
                item = component.get(key, {})
                if not isinstance(item, dict):
                    raise ValueError("distillation.output.%s must be a mapping" % key)
                boolean(item, "enabled", key in DEPTH_OUTPUTS)
                selected |= output_enabled(component, key)
                allowed_item = {"enabled", "weight", "temperature"} if key in (
                    "classification", "attribute", "centerness", "direction", "geometric_weight", "depth_distribution"
                ) else {"enabled", "weight", "loss", "beta"}
                if set(item) - allowed_item:
                    raise ValueError("Unknown %s options: %s" % (key, sorted(set(item) - allowed_item)))
                number(item.get("weight", 0.1), key + ".weight")
                number(item.get("temperature", 1.0), key + ".temperature", True)
                number(item.get("beta", 1.0 / 9.0), key + ".beta", True)
                if item.get("loss", "smooth_l1") != "smooth_l1":
                    raise ValueError("Only smooth_l1 regression distillation is supported")
            for key, required in (("bbox2d", "local_geometric_constraints"),
                                  ("geometric_weight", "geometric_depth")):
                if output_enabled(component, key) and not config["model"].get(required, False):
                    raise ValueError("%s distillation requires %s=true" % (key, required))
            if not selected:
                raise ValueError("Enable at least one output loss")
            quality = component.get("quality", {})
            boolean(quality, "enabled", True)
            if quality.get("mode", "gt_depth_relative_error") != "gt_depth_relative_error":
                raise ValueError("Unknown teacher quality mode")
            number(quality.get("relative_error_scale", 0.1), "relative_error_scale", True)
        else:
            taps = component.get("taps", [])
            names = [tap.get("name") for tap in taps]
            if not taps or len(set(names)) != len(names) or set(names) - {"neck_pyramid", "head_cls_tower", "head_reg_tower"}:
                raise ValueError("Feature taps must be nonempty, unique and supported")
            levels = component.get("levels", ["p3", "p4", "p5", "p6", "p7"])
            if not levels or len(set(levels)) != len(levels) or set(levels) - {"p3", "p4", "p5", "p6", "p7"}:
                raise ValueError("Invalid feature levels")
            for tap in taps:
                if tap.get("adapter", "identity") not in ("identity", "conv1x1") or tap.get("normalization", "channel_l2") not in ("channel_l2", "none") or tap.get("loss", "mse") != "mse":
                    raise ValueError("Unsupported feature adapter, normalization or loss")
                number(tap.get("weight", 0.05), "feature.weight")
    if not active:
        raise ValueError("Enable output or feature distillation")


def teacher_input(images, targets, student_data, teacher_data):
    """Undo only normalization; preserve geometry and refill teacher padding."""
    shape = (1, 3, 1, 1)
    raw = images.float()
    raw = raw * raw.new_tensor(student_data["image_std"]).view(shape)
    raw = raw + raw.new_tensor(student_data["image_mean"]).view(shape)
    height, width = images.shape[-2:]
    sizes = torch.stack([target["valid_image_size"] for target in targets]).to(images.device)
    y = torch.arange(height, device=images.device).view(1, 1, height, 1)
    x = torch.arange(width, device=images.device).view(1, 1, 1, width)
    valid = (y < sizes[:, 0, None, None, None]) & (x < sizes[:, 1, None, None, None])
    pad = raw.new_tensor(teacher_data["pad_value"]).view(shape)
    raw = torch.where(valid, raw, pad)
    return normalize_image(raw, teacher_data["image_mean"], teacher_data["image_std"])


class DistillationTraining(nn.Module):
    def __init__(self, teacher, teacher_config, student_config, fingerprint):
        super().__init__()
        # Teacher is intentionally NOT registered: DDP must not broadcast its
        # buffers, and student checkpoints/optimizers must not contain it.
        object.__setattr__(self, "teacher", teacher.eval().requires_grad_(False))
        self.teacher_config = teacher_config
        self.student_config = student_config
        self.options = copy.deepcopy(student_config["train"]["distillation"])
        self.fingerprint = fingerprint
        self.criterion = DistillationCriterion(student_config, teacher_config)
        feature = self.options.get("feature", {})
        self.taps = [tap["name"] for tap in feature.get("taps", [])] if feature.get("enabled", False) else []
        self.progress = 0.0
        self.last_stats = {}

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images, targets, student, outputs, features):
        with torch.no_grad():
            inputs = teacher_input(images, targets, self.student_config["data"], self.teacher_config["data"])
            result = forward_with_targets(self.teacher, inputs, targets, self.taps)
            teacher_outputs, teacher_features = result if self.taps else (result, {})
        losses, stats = self.criterion(outputs, teacher_outputs, targets, student.head,
                                      self.teacher.head, features, teacher_features, self.progress)
        self.last_stats = stats
        return losses

    def checkpoint_state(self):
        return {"state_dict": self.state_dict(), "teacher_sha256": self.fingerprint,
                "teacher_config": self.teacher_config, "options": self.options}

    def restore(self, state):
        expected = copy.deepcopy(self.options)
        saved = copy.deepcopy(state.get("options", {})) if state else {}
        # Moving an unchanged teacher file should not invalidate resume.
        expected.get("teacher", {}).pop("checkpoint", None)
        saved.get("teacher", {}).pop("checkpoint", None)
        if state is None or state.get("teacher_sha256") != self.fingerprint or saved != expected:
            raise ValueError("Distillation resume requires identical teacher and distillation options")
        self.load_state_dict(state["state_dict"], strict=True)


def build_distillation(config, device, logger=None):
    options = config["train"].get("distillation", {})
    if not options.get("enabled", False):
        return None
    validate_distillation_config(config)
    path = Path(options["teacher"]["checkpoint"])
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    teacher_config = checkpoint.get("config")
    if not teacher_config:
        raise ValueError("Teacher checkpoint must contain its resolved config")
    for section, keys in {
        "data": ("classes", "image_size", "color_space"),
        "model": ("strides", "regress_ranges", "depth_bin_unit", "depth_bin_max", "attribute_prediction_mode", "probabilistic_depth"),
    }.items():
        for key in keys:
            if teacher_config[section].get(key) != config[section].get(key):
                raise ValueError("Teacher/student mismatch: %s.%s" % (section, key))
    output = options.get("output", {})
    if output.get("enabled", False):
        for name, key in (("attribute", "num_attrs"), ("bbox2d", "local_geometric_constraints"),
                          ("geometric_weight", "geometric_depth")):
            if output_enabled(output, name) and teacher_config["model"].get(key) != config["model"].get(key):
                raise ValueError("Teacher/student mismatch for %s: model.%s" % (name, key))
    if len(config["model"]["strides"]) != 5:
        raise ValueError("Distillation requires five feature levels")
    if teacher_config["model"]["backbone"].startswith("legacy_hat"):
        raise ValueError("Legacy HAT teacher is unsupported")
    teacher = build_model(teacher_config).to(device)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("Teacher checkpoint has no model weights")
    teacher.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = DistillationTraining(teacher, teacher_config, config, digest.hexdigest()).to(device)
    if logger:
        logger.info("Distillation enabled | output=%s | feature=%s | region=gt_positive | separate_normalization=true",
                    options.get("output", {}).get("enabled", False), bool(result.taps))
        logger.info("Teacher loaded | path=%s | sha256=%s | strict=true | missing=0 | unexpected=0 | eval=true | requires_grad=false",
                    path, result.fingerprint)
        logger.info("Teacher architecture | backbone=%s | neck=%s | mode=%s",
                    teacher_config["model"]["backbone"], teacher_config["model"]["neck"], teacher.head.attribute_prediction_mode)
        logger.info("Distillation configuration | %s", options)
        logger.info("Distillation adapters | taps=%s | trainable_parameters=%d", result.taps,
                    sum(p.numel() for p in result.parameters()))
    return result


def summarize_distillation(stats):
    """Ratios of reduced SUMS, not a mean of per-batch/rank ratios."""
    values = {key: float(value) for key, value in stats.items()}
    count = values.get("valid_points", 0)
    quality_count = values.get("quality_count", 0)
    summary = {"valid_points": count, "positive_points": values.get("positive_points", 0)}
    summary["output_valid_points"] = quality_count
    for key, value in values.items():
        if key.endswith("_raw_sum") or key.endswith("_weighted_sum"):
            suffix = "_raw_sum" if key.endswith("_raw_sum") else "_weighted_sum"
            denominator = values.get(key[:-len(suffix)] + "_count", count)
            summary[key[:-4]] = value / denominator if denominator else None
        elif key.startswith("loss_kd_output_") and key.endswith("_count"):
            summary[key] = value
    for name, key in {"teacher_quality_mean": "quality_sum", "teacher_depth_mae": "teacher_error_sum",
                      "student_depth_mae": "student_error_sum", "teacher_better_fraction": "teacher_better_sum"}.items():
        summary[name] = values.get(key, 0) / quality_count if quality_count else None
    for component in ("output", "feature"):
        summary[component + "_total"] = sum(value or 0 for key, value in summary.items()
                                             if key.startswith("loss_kd_" + component) and key.endswith("_weighted"))
    summary["kd_total"] = summary["output_total"] + summary["feature_total"]
    return summary
