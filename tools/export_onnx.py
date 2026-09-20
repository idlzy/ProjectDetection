#!/usr/bin/env python3
"""Export FCOS3D raw feature-pyramid heads to an ONNX model."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from project_detection.config import load_config
from project_detection.engine import load_checkpoint
from project_detection.models import build_model

if __package__:
    from tools.onnx_contract import BASE_OUTPUT_FIELDS, output_names
else:
    from onnx_contract import BASE_OUTPUT_FIELDS, output_names


class RawHeadExportWrapper(nn.Module):
    """Flatten per-level output dictionaries into a stable ONNX tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        flat = []
        for level in self.model(image):
            flat.extend(level[field] for field in BASE_OUTPUT_FIELDS)
            if level["depth_logits"] is not None:
                flat.append(level["depth_logits"])
            if level.get("geo_weight") is not None:
                flat.append(level["geo_weight"])
        return tuple(flat)


def legacy_export_options(export_function=torch.onnx.export):
    """Use the legacy exporter when newer PyTorch defaults to Dynamo."""
    if "dynamo" in inspect.signature(export_function).parameters:
        return {"dynamo": False}
    return {}


def export_model(config, checkpoint, output, opset=13, overwrite=False):
    """Export and validate a fixed-shape raw-head graph plus metadata."""
    output = Path(output)
    metadata_path = output.with_suffix(".meta.json")
    for path in (output, metadata_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                "%s already exists; pass --overwrite to replace it" % path
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(config)
    load_checkpoint(checkpoint, model, strict=True)
    model.eval()
    height, width = config["data"]["image_size"]
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            RawHeadExportWrapper(model),
            dummy,
            str(output),
            input_names=["image"],
            output_names=output_names(config),
            opset_version=opset,
            do_constant_folding=True,
            **legacy_export_options(),
        )

    import onnx

    onnx.checker.check_model(onnx.load(str(output)))
    fuse_logit = model.head.depth_fuse_logit
    metadata = {
        "format": "project_detection_raw_head_v1",
        "input_name": "image",
        "input_shape": [1, 3, height, width],
        "output_names": output_names(config),
        "depth_fuse_logit": (
            float(fuse_logit.detach().cpu()) if fuse_logit is not None else None
        ),
        "probabilistic_depth": bool(config["model"]["probabilistic_depth"]),
        "geometric_depth": bool(config["model"].get("geometric_depth", False)),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export FCOS3D raw-head ONNX model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Defaults to experiment output/model.onnx")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    output = args.output or (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
        / "model.onnx"
    )
    exported = export_model(
        config, args.checkpoint, output, args.opset, args.overwrite
    )
    print("Exported and checked: %s" % exported)


if __name__ == "__main__":
    main()
