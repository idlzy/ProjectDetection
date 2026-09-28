#!/usr/bin/env python3
"""Randomly visualize GT boxes competing for the same FCOS3D feature point."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.mw3d import Mw3dReadyDataset
from project_detection.task.targets import feature_points, target_assignment_diagnostics


WINNER_COLOR = "#20e878"
OTHER_COLOR = "#ff9b44"
POINT_COLOR = "#ffffff"


def _conflict_image_indices(per_gt_path, level_names):
    """A candidate-minus-assigned surplus proves a point had multiple GTs."""
    surplus = defaultdict(int)
    by_level = {level: defaultdict(int) for level in level_names}
    with per_gt_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_index = int(row["dataset_index"])
            surplus[image_index] += (
                int(row["total_candidate_points"])
                - int(row["total_assigned_points"])
            )
            for level in level_names:
                by_level[level][image_index] += (
                    int(row[level + "_candidate_points"])
                    - int(row[level + "_assigned_points"])
                )
    return (
        [index for index, count in surplus.items() if count > 0],
        {level: [index for index, count in counts.items() if count > 0]
         for level, counts in by_level.items()},
    )


def _zero_positive_gt_rows(per_gt_path, level_names):
    """Find GTs whose candidate points all lost to another GT."""
    rows = []
    by_level = {level: [] for level in level_names}
    candidate_counts = {}
    with per_gt_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (int(row["total_candidate_points"]) == 0
                    or int(row["total_assigned_points"]) != 0):
                continue
            item = (int(row["dataset_index"]), int(row["gt_index"]))
            rows.append(item)
            candidate_counts[item] = int(row["total_candidate_points"])
            for level in level_names:
                if int(row[level + "_candidate_points"]) > 0:
                    by_level[level].append(item)
    return rows, by_level, candidate_counts


def _sample_conflict(target, config, rng, center_radius, requested_level=None,
                     lost_gt=None):
    height, width = config["data"]["image_size"]
    conflicts = []
    for level, (stride, regress_range) in enumerate(zip(
        config["model"]["strides"], config["model"]["regress_ranges"]
    )):
        if requested_level is not None and level != requested_level:
            continue
        points = feature_points(
            (height + stride - 1) // stride,
            (width + stride - 1) // stride,
            stride,
            torch.device("cpu"),
        )
        candidates, matched = target_assignment_diagnostics(
            points, target, regress_range, stride, center_radius=center_radius
        )
        for point_index in torch.where(candidates.sum(dim=1) > 1)[0].tolist():
            if (lost_gt is not None
                    and (not bool(candidates[point_index, lost_gt])
                         or int(matched[point_index]) == lost_gt)):
                continue
            conflicts.append((level, points[point_index],
                              torch.where(candidates[point_index])[0].tolist(),
                              int(matched[point_index])))
    if not conflicts:
        raise RuntimeError("Report marks image as conflicting, but no conflict was found")
    return rng.choice(conflicts)


def _input_canvas(target, input_size):
    original = cv2.imread(target["image_path"], cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(target["image_path"])
    height, width = input_size
    scale = target["scale_factor"]
    resized_width = int(round(original.shape[1] * scale))
    resized_height = int(round(original.shape[0] * scale))
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = cv2.resize(
        original, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _crop_bounds(target, gt_indices, point, input_size):
    boxes = target["boxes2d"][gt_indices].numpy()
    height, width = input_size
    left = min(float(boxes[:, 0].min()), float(point[0]))
    top = min(float(boxes[:, 1].min()), float(point[1]))
    right = max(float(boxes[:, 2].max()), float(point[0]))
    bottom = max(float(boxes[:, 3].max()), float(point[1]))
    margin = max(24.0, 0.12 * max(right - left, bottom - top))
    return (
        max(0, left - margin), min(width, right + margin),
        max(0, top - margin), min(height, bottom + margin),
    )


def _draw(ax, image, target, gt_indices, winner, point, bounds=None, labels=False):
    ax.imshow(image)
    for gt_index in gt_indices:
        x1, y1, x2, y2 = target["boxes2d"][gt_index].tolist()
        color = WINNER_COLOR if gt_index == winner else OTHER_COLOR
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=2.5,
        ))
        center = target["centers2d"][gt_index]
        ax.scatter(float(center[0]), float(center[1]), marker="*", s=125,
                   c=color, edgecolors="black", linewidths=0.7, zorder=5)
        if labels:
            role = "winner" if gt_index == winner else "candidate"
            ax.text(x1, y1 - 3, "GT %d %s" % (gt_index, role),
                    fontsize=9, color="white", va="bottom", clip_on=True,
                    bbox=dict(facecolor=color, edgecolor="none", alpha=0.9,
                              boxstyle="round,pad=0.25"))
    ax.scatter(float(point[0]), float(point[1]), marker="X", s=140,
               c=POINT_COLOR, edgecolors="black", linewidths=1.5, zorder=7)
    if bounds is None:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    else:
        left, right, top, bottom = bounds
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])


def _description(target, classes, gt_indices, winner, lost_gt=None):
    lines = []
    for gt_index in gt_indices:
        box = target["boxes2d"][gt_index]
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        depth = float(target["boxes3d"][gt_index, 2])
        role = ("selected" if gt_index == winner else
                "lost all positives" if gt_index == lost_gt else "other")
        class_name = classes[int(target["labels"][gt_index])]
        lines.append("%s: GT %d %s, area %.0f px², depth %.1f m" % (
            role, gt_index, class_name, area, depth
        ))
    return lines


def _save_example(path, image, target, classes, conflict, input_size,
                  dataset_index, lost_gt=None, lost_candidate_count=None):
    level, point_tensor, gt_indices, winner = conflict
    shown_indices = ([winner, lost_gt] if lost_gt is not None else gt_indices)
    point = point_tensor.tolist()
    bounds = _crop_bounds(target, shown_indices, point, input_size)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.7), dpi=160,
                                gridspec_kw={"width_ratios": [1, 1]})
    _draw(axes[0], image, target, shown_indices, winner, point)
    axes[0].set_title("Full input image")
    _draw(axes[1], image, target, shown_indices, winner, point,
          bounds=bounds, labels=True)
    axes[1].set_title("Conflict crop")
    title = "Training image %d | P%d | feature point (%.1f, %.1f)" % (
        dataset_index, level + 3, point[0], point[1])
    if lost_gt is not None:
        title += " | GT %d: %d candidates -> 0 assigned" % (
            lost_gt, lost_candidate_count)
    figure.suptitle(title, fontsize=11)
    figure.text(0.02, 0.025, "   |   ".join(
        _description(target, classes, shown_indices, winner, lost_gt)
    ), fontsize=9)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.1,
                           wspace=0.04)
    figure.savefig(path)
    plt.close(figure)
    return bounds, shown_indices


def _save_contact_sheet(path, examples):
    columns = 2
    rows = (len(examples) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(13, 4.7 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    for ax, example in zip(axes, examples):
        _draw(ax, example["image"], example["target"], example["gt_indices"],
              example["winner"], example["point"], example["bounds"])
        title = "Sample %d · P%d · GT %d selected" % (
            example["order"], example["level"] + 3, example["winner"])
        if example["lost_gt"] is not None:
            title += " · GT %d: %d→0" % (
                example["lost_gt"], example["lost_candidate_count"])
        ax.set_title(title, fontsize=10)
    for ax in axes[len(examples):]:
        ax.axis("off")
    figure.suptitle("Random GT assignment conflicts: green = selected, orange = other; X = feature point, stars = projected centers", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stratify-levels", help="Comma-separated levels, e.g. P3,P4,P5")
    parser.add_argument("--zero-positive", action="store_true",
                        help="Sample only GTs that lost every candidate point")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
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
        split="train",
        classes=config["data"]["classes"],
        image_size=tuple(config["data"]["image_size"]),
    )
    level_names = ["P%d" % (index + 3)
                   for index in range(len(config["model"]["strides"]))]
    conflict_indices, by_level = _conflict_image_indices(args.per_gt, level_names)
    zero_positive_rows = []
    zero_candidate_counts = {}
    if args.zero_positive:
        zero_positive_rows, by_level, zero_candidate_counts = _zero_positive_gt_rows(
            args.per_gt, level_names)
    pool = zero_positive_rows if args.zero_positive else conflict_indices
    rng = random.Random(args.seed)
    if args.stratify_levels:
        requested = [item.strip() for item in args.stratify_levels.split(",")]
        if not requested or any(item not in level_names for item in requested):
            raise ValueError("--stratify-levels must contain known FPN levels")
        chosen = []
        used = set()
        for order in range(args.samples):
            level_name = requested[order % len(requested)]
            available = [item for item in by_level[level_name]
                         if (item[0] if args.zero_positive else item) not in used]
            if not available:
                raise ValueError("Not enough conflict images for %s" % level_name)
            item = rng.choice(available)
            dataset_index = item[0] if args.zero_positive else item
            chosen.append((item, level_names.index(level_name)))
            used.add(dataset_index)
    else:
        chosen = [(item, None) for item in rng.sample(
            pool, min(args.samples, len(pool))
        )]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty: %s" % args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    manifest = []
    for order, (item, requested_level) in enumerate(chosen, start=1):
        dataset_index, lost_gt = item if args.zero_positive else (item, None)
        _, target = dataset[dataset_index]
        conflict = _sample_conflict(target, config, rng, summary["center_radius"],
                                    requested_level=requested_level,
                                    lost_gt=lost_gt)
        level, point_tensor, gt_indices, winner = conflict
        image = _input_canvas(target, config["data"]["image_size"])
        filename = "conflict_%02d.png" % order
        lost_candidate_count = zero_candidate_counts.get(item)
        bounds, shown_indices = _save_example(
            args.output_dir / filename, image, target, config["data"]["classes"],
            conflict, config["data"]["image_size"], dataset_index,
            lost_gt, lost_candidate_count,
        )
        examples.append(dict(order=order, image=image, target=target,
                             gt_indices=shown_indices, winner=winner,
                             lost_gt=lost_gt,
                             lost_candidate_count=lost_candidate_count,
                             level=level,
                             point=point_tensor.tolist(), bounds=bounds))
        manifest.append(dict(
            order=order, dataset_index=dataset_index,
            sample_token=target["sample_token"], level="P%d" % (level + 3),
            point_xy=point_tensor.tolist(), candidate_gt_indices=gt_indices,
            selected_gt_index=winner, lost_all_positive_gt_index=lost_gt,
            lost_gt_candidate_points=lost_candidate_count,
            filename=filename,
            gt_details=_description(target, config["data"]["classes"],
                                    shown_indices, winner, lost_gt),
        ))
    _save_contact_sheet(args.output_dir / "contact_sheet.png", examples)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(seed=args.seed, config=str(Path(args.config).resolve()),
                       source_report=str(args.per_gt.resolve()),
                       stratify_levels=args.stratify_levels,
                       zero_positive=args.zero_positive,
                       zero_positive_gt_count=len(zero_positive_rows),
                       conflict_image_count=len(conflict_indices),
                       examples=manifest), handle, ensure_ascii=False, indent=2)
    print("Saved %d examples from %d conflict images to %s" % (
        len(chosen), len(conflict_indices), args.output_dir
    ))


if __name__ == "__main__":
    main()
