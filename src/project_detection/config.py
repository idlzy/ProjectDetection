from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_name = current.pop("_base_", None)
    config: Dict[str, Any] = {}
    if base_name:
        config = load_config(str(config_path.parent / base_name))
    config = _deep_merge(config, current)
    for item in overrides:
        if "=" not in item:
            raise ValueError("Overrides must use dotted.path=value: %s" % item)
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = config
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required = ("experiment", "data", "model", "train", "evaluation", "runtime")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing config sections: %s" % ", ".join(missing))
    if len(config["data"]["classes"]) < 1:
        raise ValueError("data.classes must not be empty")
    for key in ("image_mean", "image_std", "pad_value"):
        if key in config["data"] and len(config["data"][key]) != 3:
            raise ValueError("data.%s must contain three channel values" % key)
    if any(value == 0 for value in config["data"].get("image_std", ())):
        raise ValueError("data.image_std values must be non-zero")
    if config["model"]["depth_mode"] not in ("exp", "linear"):
        raise ValueError("model.depth_mode must be exp or linear")
    if len(config["model"]["strides"]) != 5:
        raise ValueError("FCOS3D expects five feature strides")
    backbone = config["model"].get(
        "backbone", "efficientnet_b0_hat_compatible"
    )
    if backbone not in ("efficientnet_b0_hat_compatible", "resnet101_fcos3d"):
        raise ValueError("Unknown model.backbone: %s" % backbone)
    neck = config["model"].get("neck", "bifpn")
    if neck not in ("bifpn", "fpn"):
        raise ValueError("Unknown model.neck: %s" % neck)
    dcn_stages = config["model"].get("backbone_dcn_stages")
    if dcn_stages is not None and len(dcn_stages) != 4:
        raise ValueError("model.backbone_dcn_stages must contain four booleans")
    head_norm = config["model"].get("head_norm", "batch")
    if head_norm not in ("batch", "group", "none"):
        raise ValueError("model.head_norm must be batch, group or none")
    nms_backend = config["evaluation"].get("nms_backend", "auto")
    if nms_backend not in ("auto", "horizon", "reference"):
        raise ValueError("evaluation.nms_backend must be auto, horizon or reference")
    if config["train"].get("validate_every", 1) < 1:
        raise ValueError("train.validate_every must be at least 1")
    if config["runtime"].get("log_every", 1) < 1:
        raise ValueError("runtime.log_every must be at least 1")
    geometry = config["model"].get("geometry", {})
    if geometry.get("enabled", False):
        if not config["model"].get("geometric_depth", False):
            raise ValueError("model.geometric_depth must be true when geometry.enabled is true")
        if not config["model"].get("probabilistic_depth", False):
            raise ValueError("full PGD geometry requires model.probabilistic_depth=true")
        if geometry.get("topk_edges", 0) < 1:
            raise ValueError("model.geometry.topk_edges must be at least 1")
        if geometry.get("max_nodes", 0) < 2:
            raise ValueError("model.geometry.max_nodes must be at least 2")
        unknown_classes = set(geometry.get("classes", ())) - set(config["data"]["classes"])
        if unknown_classes:
            raise ValueError("Unknown geometry classes: %s" % ", ".join(sorted(unknown_classes)))
