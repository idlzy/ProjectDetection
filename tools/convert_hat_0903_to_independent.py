#!/usr/bin/env python3
"""Convert the 0903 HAT float checkpoint to the independent PGDA graph.

The target is the ordinary EfficientNetB0/BiFPN/FCOS3DHead model, not the
Legacy HAT implementation. Mappings follow graph nodes, never tensor shape
alone. Convolution biases removed by the independent graph are folded into
the following BatchNorm running means for equivalent evaluation behavior.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from project_detection.config import load_config
from project_detection.models import build_model


_BLOCK_LAYOUT = tuple(
    (stage, repeat)
    for stage, repeats in enumerate((1, 2, 2, 3, 3, 4, 1))
    for repeat in range(repeats)
)
_BN_FIELDS = ("weight", "bias", "running_var", "num_batches_tracked")


class _Mapper:
    def __init__(self, source, target):
        self.source = source
        self.target = {key: value.clone() for key, value in target.items()}
        self.mapped = {}
        self.folded_biases = set()

    def put(self, target_key, source_key, value=None):
        if target_key not in self.target:
            raise KeyError("Missing target tensor: %s" % target_key)
        if source_key not in self.source:
            raise KeyError("Missing source tensor: %s" % source_key)
        if target_key in self.mapped:
            raise ValueError("Target tensor mapped twice: %s" % target_key)
        if value is None:
            value = self.source[source_key]
        if value.shape != self.target[target_key].shape:
            raise ValueError(
                "Shape mismatch %s %s -> %s %s"
                % (source_key, tuple(value.shape), target_key,
                   tuple(self.target[target_key].shape))
            )
        self.target[target_key] = value.detach().clone()
        self.mapped[target_key] = source_key

    def batch_norm(self, target_prefix, source_prefix, removed_bias):
        for field in _BN_FIELDS:
            self.put(target_prefix + "." + field, source_prefix + "." + field)
        source_mean = source_prefix + ".running_mean"
        self.put(
            target_prefix + ".running_mean", source_mean,
            self.source[source_mean] - removed_bias,
        )

    def projection(self, target_prefix, source_prefix):
        source_conv = source_prefix + ".0"
        source_bn = source_prefix + ".1"
        self.put(target_prefix + ".0.weight", source_conv + ".weight")
        self.batch_norm(
            target_prefix + ".1", source_bn,
            self.source[source_conv + ".bias"],
        )
        self.folded_biases.add(source_conv + ".bias")

    def refinement(self, target_prefix, source_prefix):
        self.put(target_prefix + ".1.weight", source_prefix + ".1.weight")
        source_pointwise = source_prefix + ".2.0"
        self.put(target_prefix + ".2.weight", source_pointwise + ".weight")
        self.batch_norm(
            target_prefix + ".3", source_prefix + ".2.1",
            self.source[source_pointwise + ".bias"],
        )
        self.folded_biases.add(source_pointwise + ".bias")

    def head_block(self, target_prefix, source_prefix, level):
        source_conv = source_prefix + ".0"
        source_pointwise = source_prefix + ".1"
        source_bn = source_prefix + ".2"
        if level == 0:
            self.put(target_prefix + ".convolution.0.weight", source_conv + ".weight")
            self.put(
                target_prefix + ".convolution.1.weight",
                source_pointwise + ".weight",
            )
        pointwise_weight = self.source[source_pointwise + ".weight"]
        removed_bias = (
            pointwise_weight[:, :, 0, 0]
            .matmul(self.source[source_conv + ".bias"])
            + self.source[source_pointwise + ".bias"]
        )
        self.batch_norm(
            target_prefix + ".norms.%d" % level, source_bn, removed_bias
        )
        self.folded_biases.update((
            source_conv + ".bias", source_pointwise + ".bias",
        ))


def _map_backbone(mapper):
    for source_key in mapper.source:
        target_key = None
        match = re.match(r"^backbone\.mod1\.0\.(.+)$", source_key)
        if match:
            target_key = "backbone.stem." + match.group(1)
        match = re.match(
            r"^backbone\._blocks_conv\.(\d+)\."
            r"(_expand_conv|_depthwise_conv|_project_conv)\.(.+)$",
            source_key,
        )
        if match:
            block = int(match.group(1))
            if block >= len(_BLOCK_LAYOUT):
                raise ValueError("Unknown HAT backbone block %d" % block)
            stage, repeat = _BLOCK_LAYOUT[block]
            components = (
                {"_depthwise_conv": 0, "_project_conv": 1}
                if block == 0 else
                {"_expand_conv": 0, "_depthwise_conv": 1,
                 "_project_conv": 2}
            )
            target_key = "backbone.stages.%d.%d.block.%d.%s" % (
                stage, repeat, components[match.group(2)], match.group(3)
            )
        if target_key is not None:
            mapper.put(target_key, source_key)


def _map_neck(mapper, stacks):
    mapper.projection(
        "neck.p6_projection",
        "neck.extra_downsamples.0.resize_layer.0.lateral_conv",
    )
    first_layer_projections = (
        ("top_projections.2", 1),
        ("top_projections.1", 2),
        ("top_projections.0", 3),
        ("bottom_projections.0", 4),
        ("bottom_projections.1", 5),
    )
    for target_name, source_node in first_layer_projections:
        mapper.projection(
            "neck.layers.0." + target_name,
            "neck.bifpn_layers.0.all_nodes.%d.0.resize_layer.0.lateral_conv"
            % source_node,
        )
    for layer in range(stacks):
        for node in range(8):
            refinement_index = 3 if node < 4 or node == 7 else 4
            target_name = (
                "top_refinements.%d" % node if node < 4 else
                "bottom_refinements.%d" % (node - 4)
            )
            mapper.refinement(
                "neck.layers.%d.%s" % (layer, target_name),
                "neck.bifpn_layers.%d.all_nodes.%d.%d"
                % (layer, node, refinement_index),
            )


def _map_head_tower(mapper, target_name, source_name, depth):
    for block in range(depth):
        for level in range(5):
            mapper.head_block(
                "head.%s.blocks.%d" % (target_name, block),
                "head.%s.%d.%d" % (source_name, level, block),
                level,
            )


def _map_head(mapper, stacked_convs):
    _map_head_tower(mapper, "cls_tower", "cls_convs", stacked_convs)
    _map_head_tower(mapper, "reg_tower", "reg_convs", stacked_convs)
    _map_head_tower(mapper, "cls_branch", "conv_cls_prev", 1)
    for index in range(4):
        _map_head_tower(
            mapper, "reg_branches.%d" % index,
            "conv_reg_prevs.%d" % index, 1,
        )
    _map_head_tower(mapper, "direction_branch", "conv_dir_cls_prev", 1)
    _map_head_tower(mapper, "attribute_branch", "conv_attr_prev", 1)
    _map_head_tower(mapper, "centerness_branch", "conv_centerness_prev", 1)
    prediction_heads = (
        ("conv_cls", "conv_cls"),
        ("conv_dir", "conv_dir_cls"),
        ("conv_attr", "conv_attr"),
        ("conv_centerness", "conv_centerness"),
        ("conv_depth_prob", "conv_depth_prob"),
        *(("conv_regs.%d" % i, "conv_regs.%d" % i) for i in range(4)),
    )
    for target_name, source_name in prediction_heads:
        for field in ("weight", "bias"):
            mapper.put(
                "head.%s.%s" % (target_name, field),
                "head.%s.%s" % (source_name, field),
            )
    mapper.put(
        "head.depth_fuse_logit", "head.depth_fuse_logit",
        mapper.source["head.depth_fuse_logit"].reshape(()),
    )


def convert(source, target_model, config):
    model = config["model"]
    if not (
        model["backbone"] == "efficientnet_b0"
        and model["neck"] == "bifpn"
        and model.get("attribute_prediction_mode", "parallel") == "parallel"
        and model.get("probabilistic_depth", False)
        and not model.get("geometric_depth", False)
        and not model.get("camera_conditioning", False)
        and not model.get("local_geometric_constraints", False)
        and len(config["data"]["classes"]) == 12
    ):
        raise ValueError("Only the 12-class independent parallel PGDA graph is supported")
    mapper = _Mapper(source, target_model.state_dict())
    _map_backbone(mapper)
    _map_neck(mapper, model["bifpn_stacks"])
    _map_head(mapper, model["stacked_convs"])
    initialized = sorted(set(mapper.target) - set(mapper.mapped))
    expected = ["head.scales.%d" % index for index in range(5)]
    if initialized != expected:
        raise RuntimeError("Unmapped target tensors: %s" % initialized)
    target_model.load_state_dict(mapper.target, strict=True)
    report = {
        "target_tensors": len(mapper.target),
        "mapped_tensors": len(mapper.mapped),
        "initialized_tensors": initialized,
        "folded_source_biases": len(mapper.folded_biases),
        "mapped_by_module": {
            name: sum(key.startswith(name + ".") for key in mapper.mapped)
            for name in ("backbone", "neck", "head")
        },
        "source_to_target": {
            source_key: target_key
            for target_key, source_key in mapper.mapped.items()
        },
    }
    return mapper.target, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model = build_model(config)
    try:
        checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.input, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(source, dict):
        raise ValueError("Expected a HAT state_dict checkpoint")
    target, report = convert(source, model, config)
    report.update({
        "source": str(Path(args.input).resolve()),
        "config": str(Path(args.config).resolve()),
        "source_epoch": checkpoint.get("epoch"),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target, "conversion_report": report}, output)
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "report": str(report_path.resolve()),
        "mapped_tensors": report["mapped_tensors"],
        "initialized_tensors": report["initialized_tensors"],
    }, indent=2))


if __name__ == "__main__":
    main()
