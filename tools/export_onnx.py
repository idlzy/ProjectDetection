#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch import nn

from project_detection.config import load_config
from project_detection.engine import load_checkpoint
from project_detection.models import build_model


class ExportWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, image):
        flat = []
        for level in self.model(image):
            flat.extend([level["cls"], level["bbox"], level["direction"], level["attribute"], level["centerness"]])
            if level["depth_logits"] is not None: flat.append(level["depth_logits"])
            if level.get("geo_weight") is not None: flat.append(level["geo_weight"])
        return tuple(flat)


def main():
    parser = argparse.ArgumentParser(description="Export raw FCOS3D head outputs to ONNX")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None); parser.add_argument("--opset", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(); config = load_config(args.config, args.overrides)
    model = build_model(config); load_checkpoint(args.checkpoint, model, strict=True); model.eval()
    height, width = config["data"]["image_size"]
    dummy = torch.randn(1, 3, height, width)
    output_fields = ["cls", "bbox", "direction", "attribute", "centerness"]
    if config["model"]["probabilistic_depth"]: output_fields.append("depth_logits")
    if config["model"].get("geometric_depth", False): output_fields.append("geo_weight")
    names = ["p%d_%s" % (level, name) for level in range(3, 8)
             for name in output_fields]
    output = Path(args.output or (Path(config["experiment"]["output_dir"]) / config["experiment"]["name"] / "model.onnx"))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ExportWrapper(model), dummy, str(output), input_names=["image"], output_names=names,
                      opset_version=args.opset, do_constant_folding=True)
    print(output)


if __name__ == "__main__": main()
