#!/usr/bin/env python3
"""Show zero-positive GT boxes, separating no-candidate and conflict losses."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.mw3d import Mw3dReadyDataset
from project_detection.task.targets import feature_points
from visualize_assignment_conflicts import (
    _crop_bounds, _draw, _input_canvas, _sample_conflict,
)


SIZE_BINS = ("0-8px", "8-16px", "16-24px")
NO_CANDIDATE_COLOR = "#ff9b44"


def _read_rows(path, class_name):
    pools = {(cause, size): [] for cause in ("no_candidate", "conflict_lost")
             for size in SIZE_BINS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] != class_name or row["size_bin"] not in SIZE_BINS:
                continue
            candidate = int(row["total_candidate_points"])
            assigned = int(row["total_assigned_points"])
            if assigned:
                continue
            cause = "no_candidate" if candidate == 0 else "conflict_lost"
            pools[(cause, row["size_bin"])].append(row)
    return pools


def _draw_no_candidate(ax, image, target, gt_index, bounds, stride, input_size):
    ax.imshow(image)
    box = target["boxes2d"][gt_index]
    x1, y1, x2, y2 = box.tolist()
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                           fill=False, edgecolor=NO_CANDIDATE_COLOR,
                           linewidth=2.5))
    center = target["centers2d"][gt_index]
    ax.scatter(float(center[0]), float(center[1]), marker="*", s=130,
               c=NO_CANDIDATE_COLOR, edgecolors="black", zorder=5)
    height, width = input_size
    points = feature_points((height + stride - 1) // stride,
                            (width + stride - 1) // stride,
                            stride, torch.device("cpu"))
    left, right, top, bottom = bounds
    visible = ((points[:, 0] >= left) & (points[:, 0] <= right)
               & (points[:, 1] >= top) & (points[:, 1] <= bottom))
    dots = points[visible].numpy()
    ax.scatter(dots[:, 0], dots[:, 1], c="white", s=13, alpha=0.9,
               edgecolors="black", linewidths=0.3, zorder=4)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])


def _render(path, image, target, row, config, center_radius):
    gt_index = int(row["gt_index"])
    input_size = tuple(config["data"]["image_size"])
    cause = row["cause"]
    if cause == "no_candidate":
        center = target["centers2d"][gt_index].tolist()
        bounds = _crop_bounds(target, [gt_index], center, input_size)
        def draw(ax):
            _draw_no_candidate(ax, image, target, gt_index, bounds,
                               config["model"]["strides"][0], input_size)
        detail = {"selected_gt_index": None, "point_xy": None,
                  "all_candidate_gt_indices_at_point": []}
    else:
        level, point, candidates, winner = _sample_conflict(
            target, config, random.Random(int(row["dataset_index"]) + gt_index),
            center_radius, lost_gt=gt_index)
        point = point.tolist()
        bounds = _crop_bounds(target, [gt_index, winner], point, input_size)
        def draw(ax):
            _draw(ax, image, target, [winner, gt_index], winner, point,
                  bounds=bounds)
        detail = {"selected_gt_index": winner, "point_xy": point,
                  "level": "P%d" % (level + 3),
                  "all_candidate_gt_indices_at_point": candidates}

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150,
                                gridspec_kw={"width_ratios": [1, 1]})
    axes[0].imshow(image)
    box = target["boxes2d"][gt_index].tolist()
    axes[0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                                box[3] - box[1], fill=False,
                                edgecolor=NO_CANDIDATE_COLOR, linewidth=2.5))
    axes[0].set_xlim(0, image.shape[1])
    axes[0].set_ylim(image.shape[0], 0)
    axes[0].axis("off")
    draw(axes[1])
    figure.suptitle("%s | image %s | GT %d | %s | %s candidates -> 0 assigned" % (
        config["data"]["classes"][int(target["labels"][gt_index])],
        row["dataset_index"], gt_index, row["size_bin"],
        row["total_candidate_points"]), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path)
    plt.close(figure)
    return draw, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="Obstacle")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty: %s" % args.output_dir)
    config = load_config(args.config)
    with (args.per_gt.parent / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if (summary["regress_ranges"] != config["model"]["regress_ranges"]
            or summary["strides"] != config["model"]["strides"]
            or summary["split"] != "train"):
        raise ValueError("The report does not match the training config")
    dataset = Mw3dReadyDataset(
        data_root=config["data"]["data_root"],
        ready_root=config["data"].get("ready_root"),
        split="train", classes=config["data"]["classes"],
        image_size=tuple(config["data"]["image_size"]),
    )
    pools = _read_rows(args.per_gt, args.class_name)
    rng = random.Random(args.seed)
    chosen = []
    used_images = set()
    for cause in ("no_candidate", "conflict_lost"):
        for size in SIZE_BINS:
            available = [row for row in pools[(cause, size)]
                         if int(row["dataset_index"]) not in used_images]
            if not available:
                raise ValueError("No unused GT for %s / %s" % (cause, size))
            row = dict(rng.choice(available), cause=cause)
            chosen.append(row)
            used_images.add(int(row["dataset_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)
    manifest = []
    for index, row in enumerate(chosen):
        dataset_index = int(row["dataset_index"])
        gt_index = int(row["gt_index"])
        _, target = dataset[dataset_index]
        image = _input_canvas(target, config["data"]["image_size"])
        filename = "zero_positive_%02d.png" % (index + 1)
        draw, detail = _render(
            args.output_dir / filename, image, target, row, config,
            summary["center_radius"])
        draw(axes.flat[index])
        axes.flat[index].set_title("%s | %s | %s -> 0" % (
            row["cause"].replace("_", " "), row["size_bin"],
            row["total_candidate_points"]), fontsize=10)
        manifest.append(dict(order=index + 1, dataset_index=dataset_index,
                             sample_token=target["sample_token"],
                             gt_index=gt_index, cause=row["cause"],
                             size_bin=row["size_bin"],
                             box_width_px=float(row["box_width_px"]),
                             box_height_px=float(row["box_height_px"]),
                             depth_m=float(row["depth_m"]),
                             candidate_points=int(row["total_candidate_points"]),
                             assigned_points=0, filename=filename, **detail))
    figure.suptitle("%s zero-positive GTs: orange = zero-positive box; white dots = P3 grid; stars = projected centers; X = conflict point" % args.class_name,
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output_dir / "contact_sheet.png")
    plt.close(figure)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(seed=args.seed, class_name=args.class_name,
                       source_report=str(args.per_gt.resolve()),
                       config=str(Path(args.config).resolve()),
                       pool_counts={"%s/%s" % key: len(value)
                                    for key, value in pools.items()},
                       examples=manifest), handle, ensure_ascii=False, indent=2)
    print("Saved %d %s zero-positive examples to %s" % (
        len(chosen), args.class_name, args.output_dir))


if __name__ == "__main__":
    main()
