#!/usr/bin/env python3
"""Analyze FCOS3D positive-sample assignment for every dataset GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.mw3d import Mw3dReadyDataset
from project_detection.task.targets import (
    feature_points,
    target_assignment_diagnostics,
)


DEFAULT_SIZE_EDGES = (0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 192.0, 384.0, math.inf)
DEFAULT_DEPTH_EDGES = (0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, math.inf)


def _parse_edges(value):
    edges = []
    for item in value.split(","):
        item = item.strip().lower()
        edges.append(math.inf if item in ("inf", "+inf") else float(item))
    if len(edges) < 2 or edges[0] != 0.0:
        raise argparse.ArgumentTypeError("bin edges must start at 0 and contain at least two values")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError("bin edges must be strictly increasing")
    return tuple(edges)


def _format_number(value):
    if math.isinf(value):
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return format(value, "g")


def numeric_bin(value, edges, suffix=""):
    """Return a stable half-open bin label for a non-negative value."""
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper:
            if math.isinf(upper):
                return "%s%s+" % (_format_number(lower), suffix)
            return "%s-%s%s" % (
                _format_number(lower), _format_number(upper), suffix
            )
    return "out_of_range"


class AssignmentAccumulator:
    """Accumulate candidate and final-positive counts for one GT group."""

    def __init__(self, level_names):
        self.level_names = tuple(level_names)
        self.gt_count = 0
        self.zero_positive_count = 0
        self.single_positive_count = 0
        self.positive_buckets = OrderedDict(
            (("0", 0), ("1", 0), ("2-4", 0), ("5+", 0))
        )
        self.total_candidate_points = 0
        self.total_assigned_points = 0
        self.levels = {
            name: {
                "candidate_gt_count": 0,
                "assigned_gt_count": 0,
                "candidate_point_count": 0,
                "assigned_point_count": 0,
            }
            for name in self.level_names
        }

    def add(self, candidate_counts, assigned_counts):
        self.gt_count += 1
        total = int(sum(assigned_counts))
        self.total_candidate_points += int(sum(candidate_counts))
        self.total_assigned_points += total
        if total == 0:
            bucket = "0"
            self.zero_positive_count += 1
        elif total == 1:
            bucket = "1"
            self.single_positive_count += 1
        elif total <= 4:
            bucket = "2-4"
        else:
            bucket = "5+"
        self.positive_buckets[bucket] += 1

        for name, candidate, assigned in zip(
            self.level_names, candidate_counts, assigned_counts
        ):
            level = self.levels[name]
            candidate = int(candidate)
            assigned = int(assigned)
            level["candidate_point_count"] += candidate
            level["assigned_point_count"] += assigned
            level["candidate_gt_count"] += int(candidate > 0)
            level["assigned_gt_count"] += int(assigned > 0)

    def as_dict(self):
        denominator = max(self.gt_count, 1)
        return {
            "gt_count": self.gt_count,
            "zero_positive_count": self.zero_positive_count,
            "zero_positive_rate": self.zero_positive_count / denominator,
            "single_positive_count": self.single_positive_count,
            "single_positive_rate": self.single_positive_count / denominator,
            "positive_count_buckets": dict(self.positive_buckets),
            "total_candidate_points": self.total_candidate_points,
            "total_assigned_points": self.total_assigned_points,
            "mean_candidate_points_per_gt": self.total_candidate_points / denominator,
            "mean_assigned_points_per_gt": self.total_assigned_points / denominator,
            "levels": self.levels,
        }


def _add_group(groups, key, level_names, candidate_counts, assigned_counts):
    if key not in groups:
        groups[key] = AssignmentAccumulator(level_names)
    groups[key].add(candidate_counts, assigned_counts)


def _table_fieldnames(key_names, level_names):
    fields = list(key_names) + [
        "gt_count",
        "zero_positive_count",
        "zero_positive_rate",
        "single_positive_count",
        "single_positive_rate",
        "total_candidate_points",
        "total_assigned_points",
        "mean_candidate_points_per_gt",
        "mean_assigned_points_per_gt",
    ]
    for level in level_names:
        fields.extend(
            [
                level + "_candidate_gt_count",
                level + "_assigned_gt_count",
                level + "_candidate_point_count",
                level + "_assigned_point_count",
            ]
        )
    return fields


def _write_group_table(path, groups, key_names, level_names):
    fields = _table_fieldnames(key_names, level_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, accumulator in groups.items():
            keys = key if isinstance(key, tuple) else (key,)
            stats = accumulator.as_dict()
            row = dict(zip(key_names, keys))
            for name in fields[len(key_names):]:
                if "_candidate_" in name or "_assigned_" in name:
                    level_name, metric_suffix = name.split("_", 1)
                    if level_name in stats["levels"]:
                        row[name] = stats["levels"][level_name][metric_suffix]
                        continue
                row[name] = stats.get(name)
            writer.writerow(row)


def _group_json(groups):
    result = OrderedDict()
    for key, accumulator in groups.items():
        if isinstance(key, tuple):
            key = " | ".join(str(item) for item in key)
        result[str(key)] = accumulator.as_dict()
    return result


def _plot_level_heatmap(path, groups, level_names, title):
    labels = list(groups)
    if not labels:
        return
    matrix = np.asarray(
        [
            [groups[label].levels[level]["assigned_gt_count"] for level in level_names]
            for label in labels
        ],
        dtype=np.float64,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    ratios = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    figure_width = max(7.0, len(level_names) * 1.25)
    figure_height = max(4.5, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(ratios, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(level_names)), level_names)
    ax.set_yticks(range(len(labels)), [str(label) for label in labels])
    ax.set_xlabel("FPN level")
    ax.set_ylabel("GT group")
    ax.set_title(title)
    for row in range(ratios.shape[0]):
        for column in range(ratios.shape[1]):
            ax.text(
                column,
                row,
                "%.1f%%" % (100.0 * ratios[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if ratios[row, column] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, label="Share of level-covered GT assignments")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_positive_buckets(path, accumulator):
    buckets = accumulator.positive_buckets
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(list(buckets), list(buckets.values()), color="#2864B7")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Final positive points per GT")
    ax.set_ylabel("GT count")
    ax.set_title("Positive-sample count distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_bbox_width_height(path, widths, heights):
    """Plot every selected GT box in input-image pixel coordinates."""
    if not widths:
        return
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(
        widths,
        heights,
        s=3,
        alpha=0.12,
        color="#2864B7",
        edgecolors="none",
        rasterized=True,
    )
    maximum_width = max(widths)
    maximum_height = max(heights)
    diagonal_maximum = min(maximum_width, maximum_height)
    ax.plot(
        [0.0, diagonal_maximum],
        [0.0, diagonal_maximum],
        color="#D45656",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label="width = height",
    )
    ax.set_xlim(0.0, maximum_width * 1.03 if maximum_width else 1.0)
    ax.set_ylim(0.0, maximum_height * 1.03 if maximum_height else 1.0)
    ax.set_xlabel("BBox width in model-input pixels")
    ax.set_ylabel("BBox height in model-input pixels")
    ax.set_title("Width-height distribution of all selected GT boxes")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.98,
        "GT boxes: %d" % len(widths),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_dataset(config, split):
    data = config["data"]
    return Mw3dReadyDataset(
        data_root=data["data_root"],
        ready_root=data.get("ready_root"),
        split=split,
        classes=data["classes"],
        image_size=tuple(data["image_size"]),
        image_mean=data.get("image_mean", (128.0, 128.0, 128.0)),
        image_std=data.get("image_std", (128.0, 128.0, 128.0)),
        pad_value=data.get("pad_value", (0.0, 0.0, 0.0)),
    )


def analyze(args):
    config = load_config(args.config)
    dataset = _build_dataset(config, args.split)
    classes = config["data"]["classes"]
    selected_classes = set(args.classes or classes)
    unknown = selected_classes.difference(classes)
    if unknown:
        raise ValueError("Unknown classes: %s" % ", ".join(sorted(unknown)))

    indices = list(range(len(dataset)))
    if args.max_samples is not None and args.max_samples < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.max_samples))

    input_height, input_width = config["data"]["image_size"]
    strides = config["model"]["strides"]
    regress_ranges = config["model"]["regress_ranges"]
    level_names = tuple("P%d" % (index + 3) for index in range(len(strides)))
    points_by_level = []
    for stride in strides:
        height = (input_height + stride - 1) // stride
        width = (input_width + stride - 1) // stride
        points_by_level.append(
            feature_points(height, width, stride, torch.device("cpu"))
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(exist_ok=True)

    groups = {
        "size": OrderedDict(),
        "class": OrderedDict(),
        "depth": OrderedDict(),
        "focal": OrderedDict(),
        "class_size": OrderedDict(),
        "depth_size": OrderedDict(),
        "focal_size": OrderedDict(),
    }
    overall = AssignmentAccumulator(level_names)
    processed_images = 0
    skipped_gt = 0
    bbox_widths = []
    bbox_heights = []

    per_gt_fields = [
        "sample_token", "dataset_index", "gt_index", "class_name", "depth_m",
        "box_width_px", "box_height_px", "box_area_px", "box_max_side_px",
        "fx", "fy", "cx", "cy", "width_over_fx", "height_over_fy",
        "normalized_width_px", "normalized_height_px", "size_bin", "depth_bin",
        "focal_group", "center_inside_box", "touches_input_boundary",
        "candidate_levels", "assigned_levels", "total_candidate_points",
        "total_assigned_points", "has_positive", "has_single_positive",
    ]
    for level in level_names:
        per_gt_fields.extend([level + "_candidate_points", level + "_assigned_points"])

    per_gt_path = output_dir / "per_gt.csv"
    with per_gt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_gt_fields)
        writer.writeheader()
        with torch.no_grad():
            for position, dataset_index in enumerate(indices, start=1):
                _, target = dataset[dataset_index]
                num_gt = int(target["labels"].shape[0])
                if num_gt == 0:
                    processed_images += 1
                    continue

                candidate_by_level = []
                assigned_by_level = []
                for points, stride, regress_range in zip(
                    points_by_level, strides, regress_ranges
                ):
                    candidates, matched = target_assignment_diagnostics(
                        points,
                        target,
                        regress_range,
                        stride,
                        center_radius=args.center_radius,
                    )
                    candidate_by_level.append(candidates.sum(dim=0).cpu())
                    positive = matched >= 0
                    assigned_by_level.append(
                        torch.bincount(
                            matched[positive], minlength=num_gt
                        ).cpu()
                    )

                candidate_matrix = torch.stack(candidate_by_level, dim=1)
                assigned_matrix = torch.stack(assigned_by_level, dim=1)
                camera = target["camera_matrix"]
                fx, fy = float(camera[0, 0]), float(camera[1, 1])
                cx, cy = float(camera[0, 2]), float(camera[1, 2])
                focal_group = "fx~%d" % int(round(fx / args.focal_group_width) * args.focal_group_width)

                for gt_index in range(num_gt):
                    class_name = classes[int(target["labels"][gt_index])]
                    if class_name not in selected_classes:
                        skipped_gt += 1
                        continue
                    box = target["boxes2d"][gt_index]
                    center = target["centers2d"][gt_index]
                    width = max(float(box[2] - box[0]), 0.0)
                    height = max(float(box[3] - box[1]), 0.0)
                    bbox_widths.append(width)
                    bbox_heights.append(height)
                    max_side = max(width, height)
                    depth = float(target["boxes3d"][gt_index, 2])
                    size_bin = numeric_bin(max_side, args.size_edges, "px")
                    depth_bin = numeric_bin(depth, args.depth_edges, "m")
                    candidate_counts = candidate_matrix[gt_index].tolist()
                    assigned_counts = assigned_matrix[gt_index].tolist()
                    total_candidate = int(sum(candidate_counts))
                    total_assigned = int(sum(assigned_counts))
                    center_inside = bool(
                        box[0] < center[0] < box[2]
                        and box[1] < center[1] < box[3]
                    )
                    touches_boundary = bool(
                        box[0] <= args.boundary_tolerance
                        or box[1] <= args.boundary_tolerance
                        or box[2] >= input_width - 1 - args.boundary_tolerance
                        or box[3] >= input_height - 1 - args.boundary_tolerance
                    )

                    overall.add(candidate_counts, assigned_counts)
                    _add_group(groups["size"], size_bin, level_names, candidate_counts, assigned_counts)
                    _add_group(groups["class"], class_name, level_names, candidate_counts, assigned_counts)
                    _add_group(groups["depth"], depth_bin, level_names, candidate_counts, assigned_counts)
                    _add_group(groups["focal"], focal_group, level_names, candidate_counts, assigned_counts)
                    _add_group(groups["class_size"], (class_name, size_bin), level_names, candidate_counts, assigned_counts)
                    _add_group(groups["depth_size"], (depth_bin, size_bin), level_names, candidate_counts, assigned_counts)
                    _add_group(groups["focal_size"], (focal_group, size_bin), level_names, candidate_counts, assigned_counts)

                    row = {
                        "sample_token": target["sample_token"],
                        "dataset_index": dataset_index,
                        "gt_index": gt_index,
                        "class_name": class_name,
                        "depth_m": depth,
                        "box_width_px": width,
                        "box_height_px": height,
                        "box_area_px": width * height,
                        "box_max_side_px": max_side,
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "width_over_fx": width / fx if fx else None,
                        "height_over_fy": height / fy if fy else None,
                        "normalized_width_px": width * args.reference_focal / fx if fx else None,
                        "normalized_height_px": height * args.reference_focal / fy if fy else None,
                        "size_bin": size_bin,
                        "depth_bin": depth_bin,
                        "focal_group": focal_group,
                        "center_inside_box": int(center_inside),
                        "touches_input_boundary": int(touches_boundary),
                        "candidate_levels": "|".join(level for level, count in zip(level_names, candidate_counts) if count),
                        "assigned_levels": "|".join(level for level, count in zip(level_names, assigned_counts) if count),
                        "total_candidate_points": total_candidate,
                        "total_assigned_points": total_assigned,
                        "has_positive": int(total_assigned > 0),
                        "has_single_positive": int(total_assigned == 1),
                    }
                    for level, candidate, assigned in zip(level_names, candidate_counts, assigned_counts):
                        row[level + "_candidate_points"] = candidate
                        row[level + "_assigned_points"] = assigned
                    writer.writerow(row)

                processed_images += 1
                if args.log_every and position % args.log_every == 0:
                    print(
                        "Processed %d/%d images, %d selected GT"
                        % (position, len(indices), overall.gt_count),
                        flush=True,
                    )

    table_specs = (
        ("size_level.csv", "size", ("size_bin",)),
        ("class_level.csv", "class", ("class_name",)),
        ("depth_level.csv", "depth", ("depth_bin",)),
        ("focal_level.csv", "focal", ("focal_group",)),
        ("class_size_level.csv", "class_size", ("class_name", "size_bin")),
        ("depth_size_level.csv", "depth_size", ("depth_bin", "size_bin")),
        ("focal_size_level.csv", "focal_size", ("focal_group", "size_bin")),
    )
    for filename, group_name, key_names in table_specs:
        _write_group_table(
            output_dir / filename, groups[group_name], key_names, level_names
        )

    summary = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "dataset_images": len(dataset),
        "processed_images": processed_images,
        "selected_classes": sorted(selected_classes),
        "skipped_gt": skipped_gt,
        "input_size": [input_height, input_width],
        "strides": strides,
        "regress_ranges": regress_ranges,
        "center_radius": args.center_radius,
        "size_edges": [_format_number(value) for value in args.size_edges],
        "depth_edges": [_format_number(value) for value in args.depth_edges],
        "reference_focal": args.reference_focal,
        "focal_group_width": args.focal_group_width,
        "overall": overall.as_dict(),
        "groups": {name: _group_json(value) for name, value in groups.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    if not args.no_plots:
        _plot_positive_buckets(plots_dir / "positive_count_distribution.png", overall)
        _plot_bbox_width_height(
            plots_dir / "bbox_width_height_scatter.png",
            bbox_widths,
            bbox_heights,
        )
        _plot_level_heatmap(
            plots_dir / "assigned_gt_by_size_level.png",
            groups["size"], level_names, "Assigned GT distribution by object size",
        )
        _plot_level_heatmap(
            plots_dir / "assigned_gt_by_depth_level.png",
            groups["depth"], level_names, "Assigned GT distribution by depth",
        )

    print("Analysis complete: %s" % output_dir.resolve())
    print(
        "GT=%d, zero-positive=%d (%.2f%%), single-positive=%d (%.2f%%)"
        % (
            overall.gt_count,
            overall.zero_positive_count,
            100.0 * overall.zero_positive_count / max(overall.gt_count, 1),
            overall.single_positive_count,
            100.0 * overall.single_positive_count / max(overall.gt_count, 1),
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--class", dest="classes", action="append", help="Restrict to a class; repeat as needed")
    parser.add_argument("--center-radius", type=float, default=1.5)
    parser.add_argument("--reference-focal", type=float, default=660.0, help="Effective focal length used to normalize box sizes")
    parser.add_argument("--focal-group-width", type=float, default=25.0)
    parser.add_argument("--boundary-tolerance", type=float, default=1.0)
    parser.add_argument("--size-edges", type=_parse_edges, default=DEFAULT_SIZE_EDGES, help="Comma-separated max-side bins; use inf for the final edge")
    parser.add_argument("--depth-edges", type=_parse_edges, default=DEFAULT_DEPTH_EDGES, help="Comma-separated depth bins; use inf for the final edge")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.center_radius <= 0:
        raise ValueError("--center-radius must be positive")
    if args.reference_focal <= 0 or args.focal_group_width <= 0:
        raise ValueError("focal values must be positive")
    analyze(args)


if __name__ == "__main__":
    main()
