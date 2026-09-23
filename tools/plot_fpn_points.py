#!/usr/bin/env python3
"""Plot the P3-P7 FCOS feature-point grids defined by an experiment config.

The figure shows candidate locations only. Predictions and GT assignments
require an image and a model/annotation, so they are not represented here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from project_detection.config import load_config
from project_detection.task.targets import feature_points


COLORS = ("#2878b5", "#e07b25", "#37a474", "#b65085", "#7354ae")


def build_grids(config):
    """Use the same feature-point definition as training and post-processing."""
    image_height, image_width = config["data"]["image_size"]
    grids = []
    for index, stride in enumerate(config["model"]["strides"]):
        feature_height = math.ceil(image_height / stride)
        feature_width = math.ceil(image_width / stride)
        points = feature_points(
            feature_height, feature_width, stride, torch.device("cpu")
        ).numpy()
        grids.append({
            "name": "P%d" % (index + 3),
            "stride": stride,
            "feature_height": feature_height,
            "feature_width": feature_width,
            "points": points,
            "color": COLORS[index],
        })
    return grids


def _setup_canvas(axis, width, height, title):
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_facecolor("#f8fafc")
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("x (input pixels)")
    axis.set_ylabel("y (input pixels)")
    axis.grid(color="#dce4eb", linewidth=0.5, zorder=0)


def plot_grids(config, output_path, zoom_size=256):
    image_height, image_width = config["data"]["image_size"]
    grids = build_grids(config)
    figure, axes = plt.subplots(3, 2, figsize=(17, 15), constrained_layout=True)

    for axis, grid in zip(axes.flat[:5], grids):
        _setup_canvas(
            axis, image_width, image_height,
            "%s  |  stride %s  |  %s x %s = %s points" % (
                grid["name"], grid["stride"], grid["feature_width"],
                grid["feature_height"], len(grid["points"]),
            ),
        )
        axis.scatter(
            grid["points"][:, 0], grid["points"][:, 1],
            s=5 if grid["stride"] <= 16 else 15,
            c=grid["color"], marker="o", linewidths=0, zorder=2,
            rasterized=True,
        )

    axis = axes.flat[5]
    zoom_width = min(zoom_size, image_width)
    zoom_height = min(zoom_size, image_height)
    left = (image_width - zoom_width) / 2
    top = (image_height - zoom_height) / 2
    _setup_canvas(axis, image_width, image_height, "Center crop  |  all levels")
    axis.set_xlim(left, left + zoom_width)
    axis.set_ylim(top + zoom_height, top)
    for grid in grids:
        points = grid["points"]
        inside = (
            (points[:, 0] >= left) & (points[:, 0] <= left + zoom_width)
            & (points[:, 1] >= top) & (points[:, 1] <= top + zoom_height)
        )
        axis.scatter(
            points[inside, 0], points[inside, 1],
            s=8 if grid["stride"] == 8 else 35,
            c=grid["color"], label=grid["name"],
            marker="o", linewidths=0, alpha=0.85, zorder=2,
        )
    axis.legend(loc="upper right", ncol=5, fontsize=9)

    figure.suptitle(
        "FCOS P3-P7 candidate locations  |  input %s x %s (W x H)" % (
            image_width, image_height,
        ), fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180)
    plt.close(figure)
    return grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument(
        "--output", default=None,
        help="Output PNG; default: outputs/fpn_points/<config-stem>.png",
    )
    parser.add_argument(
        "--zoom-size", type=int, default=256,
        help="Center crop side length in input pixels (default: 256)",
    )
    args = parser.parse_args()
    if args.zoom_size < 1:
        parser.error("--zoom-size must be positive")
    config = load_config(args.config)
    output = args.output or str(
        Path("outputs/fpn_points") / (Path(args.config).stem + ".png")
    )
    grids = plot_grids(config, output, args.zoom_size)
    print("Saved: %s" % output)
    for grid in grids:
        print(
            "%s: stride=%s feature=%sx%s points=%s" % (
                grid["name"], grid["stride"], grid["feature_width"],
                grid["feature_height"], len(grid["points"]),
            )
        )


if __name__ == "__main__":
    main()
