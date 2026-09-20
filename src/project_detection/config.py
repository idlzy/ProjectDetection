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
    backbone = config["model"].get("backbone", "efficientnet_b0")
    if backbone not in (
        "efficientnet_b0",
        "legacy_hat_efficientnet_b0",
        "resnet101_fcos3d",
    ):
        raise ValueError("Unknown model.backbone: %s" % backbone)
    attribute_chain = config["model"].get("attribute_chain", False)
    if not isinstance(attribute_chain, bool):
        raise ValueError("model.attribute_chain must be true or false")
    if attribute_chain and backbone == "legacy_hat_efficientnet_b0":
        raise ValueError("model.attribute_chain is not supported by Legacy HAT")
    neck = config["model"].get("neck", "bifpn")
    if neck not in ("bifpn", "legacy_hat_bifpn", "fpn"):
        raise ValueError("Unknown model.neck: %s" % neck)
    dcn_stages = config["model"].get("backbone_dcn_stages")
    if dcn_stages is not None and len(dcn_stages) != 4:
        raise ValueError("model.backbone_dcn_stages must contain four booleans")
    head_norm = config["model"].get("head_norm", "batch")
    if head_norm not in ("batch", "group", "none"):
        raise ValueError("model.head_norm must be batch, group or none")
    nms_chunk_size = config["evaluation"].get("nms_pairwise_chunk_size", 32)
    if not isinstance(nms_chunk_size, int) or isinstance(nms_chunk_size, bool) or nms_chunk_size < 1:
        raise ValueError("evaluation.nms_pairwise_chunk_size must be a positive integer")
    sharing_strategy = config["data"].get("multiprocessing_sharing_strategy")
    if sharing_strategy not in (None, "file_descriptor", "file_system"):
        raise ValueError(
            "data.multiprocessing_sharing_strategy must be file_descriptor or file_system"
        )
    min_recall = config["evaluation"].get("min_recall", 0.1)
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("evaluation.min_recall must be in [0, 1]")
    min_precision = config["evaluation"].get("min_precision", 0.1)
    if not 0.0 <= min_precision < 1.0:
        raise ValueError("evaluation.min_precision must be in [0, 1)")
    tp_distance = config["evaluation"].get("tp_distance_threshold", 2.0)
    if tp_distance not in config["evaluation"]["distance_thresholds"]:
        raise ValueError(
            "evaluation.tp_distance_threshold must be one of distance_thresholds"
        )
    if config["evaluation"].get("mean_ap_weight", 5) <= 0:
        raise ValueError("evaluation.mean_ap_weight must be greater than 0")
    evaluated_classes = config["evaluation"].get("classes")
    if evaluated_classes is not None:
        if not isinstance(evaluated_classes, list) or not evaluated_classes:
            raise ValueError("evaluation.classes must be null or a non-empty list")
        if len(evaluated_classes) != len(set(evaluated_classes)):
            raise ValueError("evaluation.classes must not contain duplicates")
        unknown_classes = set(evaluated_classes) - set(config["data"]["classes"])
        if unknown_classes:
            raise ValueError(
                "Unknown evaluation.classes: %s" % ", ".join(sorted(unknown_classes))
            )
    depth_bins = config["evaluation"].get("depth_bins", ())
    if not depth_bins or any(
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or interval[0] >= interval[1]
        for interval in depth_bins
    ):
        raise ValueError("evaluation.depth_bins must contain increasing [min, max] pairs")
    if config["train"].get("validate_every", 1) < 1:
        raise ValueError("train.validate_every must be at least 1")
    if not isinstance(config["train"].get("nonfinite_guard", True), bool):
        raise ValueError("train.nonfinite_guard must be true or false")
    if config["train"].get("nonfinite_parameter_check_every", 100) < 1:
        raise ValueError("train.nonfinite_parameter_check_every must be at least 1")
    if config["train"].get("nonfinite_max_consecutive_amp_overflows", 8) < 1:
        raise ValueError(
            "train.nonfinite_max_consecutive_amp_overflows must be at least 1"
        )
    if config["train"].get("amp_initial_scale", 2048.0) <= 0:
        raise ValueError("train.amp_initial_scale must be greater than 0")
    if config["train"].get("recovery_checkpoint_every_steps", 100) < 0:
        raise ValueError("train.recovery_checkpoint_every_steps must be non-negative")
    if not isinstance(
        config["train"].get("allow_world_size_change_on_resume", False), bool
    ):
        raise ValueError(
            "train.allow_world_size_change_on_resume must be true or false"
        )
    if not isinstance(config["train"].get("find_unused_parameters", True), bool):
        raise ValueError("train.find_unused_parameters must be true or false")
    assignment_chunk_size = config["train"].get(
        "target_assignment_chunk_size", 8
    )
    if (
        not isinstance(assignment_chunk_size, int)
        or isinstance(assignment_chunk_size, bool)
        or assignment_chunk_size < 1
    ):
        raise ValueError(
            "train.target_assignment_chunk_size must be a positive integer"
        )
    if config["runtime"].get("log_every", 1) < 1:
        raise ValueError("runtime.log_every must be at least 1")
    distributed_backend = config["runtime"].get("distributed_backend", "nccl")
    if distributed_backend != "nccl":
        raise ValueError("runtime.distributed_backend must be nccl")
    distributed_timeout = config["runtime"].get(
        "distributed_timeout_seconds", 600
    )
    if (
        not isinstance(distributed_timeout, int)
        or isinstance(distributed_timeout, bool)
        or distributed_timeout < 1
    ):
        raise ValueError(
            "runtime.distributed_timeout_seconds must be a positive integer"
        )
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
