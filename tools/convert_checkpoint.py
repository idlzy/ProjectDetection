#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from project_detection.config import load_config
from project_detection.models import build_model


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_distributed_prefix(key):
    while key.startswith("module."):
        key = key[7:]
    return key


def detect_source_format(state):
    keys = {strip_distributed_prefix(key) for key in state}
    if any(key.startswith("bbox_head.") for key in keys) and any(
        key.startswith("backbone.") for key in keys
    ):
        return "openmmlab-pgd"
    if "conv1.weight" in keys and any(
        key.startswith("layer1.") for key in keys
    ):
        return "openmmlab-resnet101-caffe"
    if "backbone.conv1.weight" in keys and any(
        key.startswith("backbone.layer1.") for key in keys
    ):
        return "openmmlab-resnet101-caffe"
    return "direct"


def remap_key(key, source_format):
    key = strip_distributed_prefix(key)
    if source_format == "direct":
        return key
    if source_format not in (
        "openmmlab-resnet101-caffe",
        "openmmlab-pgd",
    ):
        raise ValueError("Unknown source format: %s" % source_format)
    if re.match(r"^(conv1|bn1|layer[1-4])\.", key):
        key = "backbone." + key
    match = re.match(r"^backbone\.layer([1-4])\.(.+)$", key)
    if match:
        key = "backbone.stages.%d.%s" % (
            int(match.group(1)) - 1,
            match.group(2),
        )
    if source_format == "openmmlab-resnet101-caffe":
        return key

    # MMCV and the local torchvision wrapper use the same 18-offset + 9-mask
    # channel layout, but expose the predictor under different parameter names.
    if key.startswith("backbone."):
        return key.replace(".conv_offset.", ".conv_offset_mask.")

    match = re.match(
        r"^neck\.lateral_convs\.(\d+)\.conv\.(weight|bias)$", key
    )
    if match:
        return "neck.lateral_convs.%s.%s" % match.groups()
    match = re.match(
        r"^neck\.fpn_convs\.(\d+)\.conv\.(weight|bias)$", key
    )
    if match:
        index = int(match.group(1))
        parameter = match.group(2)
        if index < 3:
            return "neck.output_convs.%d.%s" % (index, parameter)
        return "neck.p%d.%s" % (index + 3, parameter)

    match = re.match(
        r"^bbox_head\.(cls|reg)_convs\.(\d+)\.conv\.weight$", key
    )
    if match:
        return "head.%s_tower.%s.0.weight" % match.groups()
    match = re.match(
        r"^bbox_head\.(cls|reg)_convs\.(\d+)\.conv\.bias$", key
    )
    if match:
        return "head.%s_tower.%s.0.bias" % match.groups()
    match = re.match(
        r"^bbox_head\.(cls|reg)_convs\.(\d+)\.conv\.conv_offset\."
        r"(weight|bias)$",
        key,
    )
    if match:
        return "head.%s_tower.%s.0.conv_offset_mask.%s" % match.groups()
    match = re.match(
        r"^bbox_head\.(cls|reg)_convs\.(\d+)\.gn\.(weight|bias)$", key
    )
    if match:
        return "head.%s_tower.%s.1.%s" % match.groups()

    # The PGD output branches, per-regression-group scales, and fuse_lambda
    # are deliberately not remapped. Their semantics differ from the compact
    # ProjectDetection head even when an individual tensor shape happens to
    # match.
    return key


def convert_state_dict(state, own_state, source_format):
    compatible = {}
    mapped_keys = {}
    mismatched = {}
    unused = []
    for source_key, value in state.items():
        target_key = remap_key(source_key, source_format)
        if target_key not in own_state:
            unused.append(source_key)
            continue
        if own_state[target_key].shape != value.shape:
            mismatched[source_key] = {
                "target": target_key,
                "source_shape": list(value.shape),
                "target_shape": list(own_state[target_key].shape),
            }
            continue
        if target_key in compatible:
            raise ValueError("Multiple source keys map to %s" % target_key)
        compatible[target_key] = value
        if target_key != strip_distributed_prefix(source_key):
            mapped_keys[source_key] = target_key
    return compatible, mapped_keys, mismatched, sorted(unused)


def tensor_elements(state):
    return sum(value.numel() for value in state.values())


def main():
    parser = argparse.ArgumentParser(
        description="Convert and audit a checkpoint for ProjectDetection"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-format",
        choices=(
            "auto",
            "direct",
            "openmmlab-resnet101-caffe",
            "openmmlab-pgd",
        ),
        default="auto",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state = unwrap_state_dict(checkpoint)
    source_format = (
        detect_source_format(state)
        if args.source_format == "auto"
        else args.source_format
    )
    own_state = build_model(load_config(args.config)).state_dict()
    compatible, mapped_keys, mismatched, unused = convert_state_dict(
        state, own_state, source_format
    )
    missing = sorted(set(own_state) - set(compatible))
    backbone_state = {
        key: value for key, value in own_state.items() if key.startswith("backbone.")
    }
    loaded_backbone = {
        key: value for key, value in compatible.items() if key.startswith("backbone.")
    }
    report = {
        "source": str(Path(args.input).resolve()),
        "source_format": source_format,
        "source_tensors": len(state),
        "loaded_tensors": len(compatible),
        "mapped_tensors": len(mapped_keys),
        "loaded_model_element_ratio": tensor_elements(compatible)
        / tensor_elements(own_state),
        "loaded_backbone_element_ratio": tensor_elements(loaded_backbone)
        / tensor_elements(backbone_state),
        "missing": missing,
        "unused_source": unused,
        "shape_mismatches": mismatched,
        "mapped_keys": mapped_keys,
    }
    output = Path(args.output)
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: report[key]
        for key in (
            "source_format",
            "source_tensors",
            "loaded_tensors",
            "mapped_tensors",
            "loaded_model_element_ratio",
            "loaded_backbone_element_ratio",
        )
    }
    summary.update(
        {
            "missing_tensors": len(missing),
            "unused_source_tensors": len(unused),
            "shape_mismatches": len(mismatched),
            "report": str(report_path.resolve()),
        }
    )
    print(json.dumps(summary, indent=2))
    if (missing or mismatched or unused) and not args.allow_partial:
        raise SystemExit(
            "Checkpoint is partial; inspect the report and rerun with "
            "--allow-partial for a controlled warm-start"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": compatible,
            "source": report["source"],
            "conversion_report": report,
        },
        output,
    )


if __name__ == "__main__":
    main()
