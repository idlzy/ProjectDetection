#!/usr/bin/env python3
"""Analyze FCOS3D positive-sample assignment for every dataset GT."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from project_detection.config import load_config
from project_detection.data.mw3d import Mw3dReadyDataset
from project_detection.task.targets import (
    feature_points,
    target_assignment_diagnostics,
)


DEFAULT_SIZE_EDGES = (0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 192.0, 384.0, math.inf)
DEFAULT_DEPTH_EDGES = (0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, math.inf)
BOX_DENSITY_EDGES = (0, 8, 16, 24, 32, 48, 64, 96, 144, 224, 352, 544, math.inf)
LTRB_EDGES = (0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, math.inf)


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


def _empty_ltrb_distribution(class_names, level_names):
    bins = len(LTRB_EDGES) - 1
    return {
        "definitions": {
            "box_center_gt": "One value per GT: max(box width, box height) / 2 in input pixels.",
            "assigned_points": "One value per final positive FPN point: max(left, top, right, bottom) to its matched GT box, in input pixels.",
        },
        "edges_px": [_format_number(edge) for edge in LTRB_EDGES],
        "box_center_gt": {
            "overall": [0] * bins,
            "by_class": {name: [0] * bins for name in class_names},
        },
        "assigned_points": {
            "overall": [0] * bins,
            "by_level": {name: [0] * bins for name in level_names},
            "by_class": {name: [0] * bins for name in class_names},
        },
    }


def _add_ltrb_values(counts, values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("GT max(ltrb) contains non-finite values")
    if np.any(values < 0):
        raise ValueError("GT max(ltrb) must be non-negative")
    histogram = np.histogram(values, bins=LTRB_EDGES)[0]
    for index, count in enumerate(histogram):
        counts[index] += int(count)


def _add_ltrb_value(counts, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("GT max(ltrb) must be finite and non-negative")
    counts[bisect.bisect_right(LTRB_EDGES, value) - 1] += 1


def _merge_ltrb_distributions(distributions):
    first = distributions[0]
    available = next((item["assigned_points"] for item in distributions if item["assigned_points"]), None)
    result = _empty_ltrb_distribution(
        first["box_center_gt"]["by_class"],
        available["by_level"] if available else (),
    )
    result["edges_px"] = first["edges_px"]
    for item in distributions:
        if item["edges_px"] != first["edges_px"]:
            raise ValueError("Cannot combine different max(ltrb) bin edges")
        for key in ("overall",):
            result["box_center_gt"][key] = [
                left + right for left, right in zip(
                    result["box_center_gt"][key], item["box_center_gt"][key]
                )
            ]
        for name, counts in item["box_center_gt"]["by_class"].items():
            target = result["box_center_gt"]["by_class"][name]
            result["box_center_gt"]["by_class"][name] = [
                left + right for left, right in zip(target, counts)
            ]
    if any(item["assigned_points"] is None for item in distributions):
        result["assigned_points"] = None
        result["assigned_points_note"] = (
            "Unavailable: at least one split predates per-point max(ltrb) statistics. "
            "Rerun assignment analysis to obtain this distribution."
        )
    else:
        for item in distributions:
            for key in ("overall",):
                result["assigned_points"][key] = [
                    left + right for left, right in zip(
                        result["assigned_points"][key], item["assigned_points"][key]
                    )
                ]
            for group in ("by_level", "by_class"):
                for name, counts in item["assigned_points"][group].items():
                    target = result["assigned_points"][group][name]
                    result["assigned_points"][group][name] = [
                        left + right for left, right in zip(target, counts)
                    ]
    return result


def _ltrb_from_existing_per_gt(path, class_names):
    result = _empty_ltrb_distribution(class_names, ())
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row["box_max_side_px"]) / 2.0
            if row["class_name"] not in result["box_center_gt"]["by_class"]:
                raise ValueError("Unexpected class in %s" % path)
            _add_ltrb_value(result["box_center_gt"]["overall"], value)
            _add_ltrb_value(
                result["box_center_gt"]["by_class"][row["class_name"]], value
            )
    result["assigned_points"] = None
    result["assigned_points_note"] = (
        "Unavailable in saved per_gt.csv; rerun assignment analysis to obtain "
        "per-point max(ltrb)."
    )
    return result


def _write_ltrb_report(output_dir, distribution, no_plots, class_name=None):
    report_dir = output_dir / "max_ltrb"
    report_dir.mkdir(parents=True, exist_ok=True)
    edges = distribution["edges_px"]
    center = distribution["box_center_gt"]
    center_counts = center["overall"] if class_name is None else center["by_class"][class_name]
    assigned = distribution["assigned_points"]
    series = {"box_center_gt": center_counts}
    if assigned is not None:
        series["assigned_points"] = (
            assigned["overall"] if class_name is None else assigned["by_class"][class_name]
        )
        if class_name is None:
            series.update(
                ("assigned_%s" % level, counts)
                for level, counts in assigned["by_level"].items()
            )
    with (report_dir / "distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("population", "bin_px", "count", "percentage"))
        for population, counts in series.items():
            total = sum(counts)
            for index, count in enumerate(counts):
                writer.writerow((
                    population, numeric_bin(float(edges[index]), LTRB_EDGES, "px"),
                    count, 100.0 * count / total if total else 0.0,
                ))
    if no_plots:
        return
    labels = [numeric_bin(float(edge), LTRB_EDGES, "px") for edge in edges[:-1]]
    for population, counts in series.items():
        fig, ax = plt.subplots(figsize=(11, 4.5))
        total = sum(counts)
        ax.bar(range(len(counts)), [100.0 * count / total if total else 0 for count in counts])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("GT max(ltrb): %s%s" % (
            population, " / " + class_name if class_name else "",
        ))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(report_dir / (population + ".png"), dpi=150)
        plt.close(fig)


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

    def merge_summary(self, summary):
        """Add previously serialized counts without averaging derived rates."""
        for key in (
            "gt_count", "zero_positive_count", "single_positive_count",
            "total_candidate_points", "total_assigned_points",
        ):
            setattr(self, key, getattr(self, key) + summary[key])
        for key in self.positive_buckets:
            self.positive_buckets[key] += summary["positive_count_buckets"][key]
        for level in self.level_names:
            for key in self.levels[level]:
                self.levels[level][key] += summary["levels"][level][key]


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


def _class_groups(groups, class_name):
    """Select the size/depth/focal groups belonging to one class."""
    return {
        group_name: OrderedDict(
            (group_key[1], accumulator)
            for group_key, accumulator in groups["class_" + group_name].items()
            if group_key[0] == class_name
        )
        for group_name in ("size", "depth", "focal")
    }


def _write_class_reports(
    output_dir, class_names, groups, level_names, widths, heights,
    summary_metadata, no_plots, ltrb_distribution,
):
    with ExitStack() as stack:
        writers = {}
        with (output_dir / "per_gt.csv").open("r", encoding="utf-8", newline="") as source:
            rows = csv.DictReader(source)
            for class_name in class_names:
                class_dir = output_dir / "classes" / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(
                    (class_dir / "per_gt.csv").open("w", encoding="utf-8", newline="")
                )
                writers[class_name] = csv.DictWriter(handle, fieldnames=rows.fieldnames)
                writers[class_name].writeheader()
            for row in rows:
                writers[row["class_name"]].writerow(row)

    for class_name in class_names:
        class_dir = output_dir / "classes" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_groups = _class_groups(groups, class_name)
        class_summary = dict(summary_metadata)
        class_summary.update({
            "class_name": class_name,
            "overall": groups["class"][class_name].as_dict(),
            "groups": {
                name: _group_json(values)
                for name, values in class_groups.items()
            },
        })
        with (class_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(class_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        _write_ltrb_report(class_dir, ltrb_distribution, no_plots, class_name)
        for name, key_name in (
            ("size", "size_bin"), ("depth", "depth_bin"),
            ("focal", "focal_group"),
        ):
            _write_group_table(
                class_dir / (name + "_level.csv"), class_groups[name],
                (key_name,), level_names,
            )
        if no_plots:
            continue
        plot_dir = class_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        _plot_positive_buckets(
            plot_dir / "positive_count_distribution.png",
            groups["class"][class_name],
        )
        _plot_bbox_width_height(
            plot_dir / "bbox_width_height_scatter.png",
            widths[class_name], heights[class_name],
        )
        _plot_bbox_width_height_density(
            plot_dir, widths[class_name], heights[class_name], class_name,
        )
        for name, title in (
            ("size", "Assigned GT by object size"),
            ("depth", "Assigned GT by depth"),
        ):
            _plot_level_heatmap(
                plot_dir / ("assigned_gt_by_%s_level.png" % name),
                class_groups[name], level_names,
                "%s: %s" % (class_name, title),
                sort_numeric_bins=True,
            )


def _merge_split_summaries(summaries):
    """Combine train/val/test counts while preserving per-split reports."""
    first = summaries[0]
    level_names = tuple("P%d" % (index + 3) for index in range(len(first["strides"])))
    overall = AssignmentAccumulator(level_names)
    group_accumulators = {name: OrderedDict() for name in first["groups"]}
    for summary in summaries:
        overall.merge_summary(summary["overall"])
        for name, entries in summary["groups"].items():
            for key, stats in entries.items():
                accumulator = group_accumulators[name].setdefault(
                    key, AssignmentAccumulator(level_names)
                )
                accumulator.merge_summary(stats)
    result = {
        key: first[key]
        for key in (
            "config", "selected_classes", "input_size", "strides",
            "regress_ranges", "center_radius", "size_edges", "depth_edges",
            "reference_focal", "focal_group_width",
        )
    }
    result.update({
        "split": "all",
        "splits": [item["split"] for item in summaries],
        "dataset_images": sum(item["dataset_images"] for item in summaries),
        "processed_images": sum(item["processed_images"] for item in summaries),
        "skipped_gt": sum(item["skipped_gt"] for item in summaries),
        "overall": overall.as_dict(),
        "groups": {
            name: _group_json(entries)
            for name, entries in group_accumulators.items()
        },
        "max_ltrb_distribution": _merge_ltrb_distributions(
            [item["max_ltrb_distribution"] for item in summaries]
        ),
    })
    return result


def _write_aggregate_class_reports(output_dir, summary, no_plots, dimensions=None):
    """Expose the combined per-class totals without duplicating per-GT rows."""
    level_names = tuple("P%d" % (index + 3) for index in range(len(summary["strides"])))
    for class_name, class_stats in summary["groups"]["class"].items():
        class_dir = output_dir / "classes" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_groups = {}
        for name in ("size", "depth", "focal"):
            prefix = class_name + " | "
            class_groups[name] = OrderedDict(
                (key[len(prefix):], stats)
                for key, stats in summary["groups"]["class_" + name].items()
                if key.startswith(prefix)
            )
        class_summary = {
            "config": summary["config"],
            "split": "all",
            "splits": summary["splits"],
            "class_name": class_name,
            "input_size": summary["input_size"],
            "strides": summary["strides"],
            "regress_ranges": summary["regress_ranges"],
            "center_radius": summary["center_radius"],
            "overall": class_stats,
            "groups": class_groups,
        }
        with (class_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(class_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        _write_ltrb_report(
            class_dir, summary["max_ltrb_distribution"], no_plots, class_name,
        )
        table_groups = {}
        for name, key_name in (
            ("size", "size_bin"), ("depth", "depth_bin"),
            ("focal", "focal_group"),
        ):
            accumulators = OrderedDict()
            for key, stats in class_groups[name].items():
                accumulator = AssignmentAccumulator(level_names)
                accumulator.merge_summary(stats)
                accumulators[key] = accumulator
            table_groups[name] = accumulators
            _write_group_table(
                class_dir / (name + "_level.csv"), accumulators,
                (key_name,), level_names,
            )
        if no_plots:
            continue
        plot_dir = class_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        accumulator = AssignmentAccumulator(level_names)
        accumulator.merge_summary(class_stats)
        _plot_positive_buckets(
            plot_dir / "positive_count_distribution.png", accumulator,
        )
        if dimensions is not None:
            _, _, class_widths, class_heights = dimensions
            _plot_bbox_width_height(
                plot_dir / "bbox_width_height_scatter.png",
                class_widths[class_name], class_heights[class_name],
            )
            _plot_bbox_width_height_density(
                plot_dir, class_widths[class_name], class_heights[class_name],
                class_name,
            )
        for name, title in (
            ("size", "Assigned GT by object size"),
            ("depth", "Assigned GT by depth"),
        ):
            _plot_level_heatmap(
                plot_dir / ("assigned_gt_by_%s_level.png" % name),
                table_groups[name], level_names,
                "%s: %s" % (class_name, title),
                sort_numeric_bins=True,
            )


def _load_box_dimensions(paths, class_names):
    widths, heights = [], []
    class_widths = {name: [] for name in class_names}
    class_heights = {name: [] for name in class_names}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                class_name = row["class_name"]
                width = float(row["box_width_px"])
                height = float(row["box_height_px"])
                if class_name not in class_widths or not (
                    math.isfinite(width) and math.isfinite(height)
                ):
                    raise ValueError("Invalid box dimensions or class in %s" % path)
                widths.append(width)
                heights.append(height)
                class_widths[class_name].append(width)
                class_heights[class_name].append(height)
    return widths, heights, class_widths, class_heights


def _plot_saved_split_heatmaps(split_dir, summary):
    """Refresh size/depth plots from saved counts without dataset inference."""
    level_names = tuple("P%d" % (index + 3) for index in range(len(summary["strides"])))
    for name, title in (
        ("size", "Assigned GT distribution by object size"),
        ("depth", "Assigned GT distribution by depth"),
    ):
        groups = OrderedDict()
        for key, stats in summary["groups"][name].items():
            accumulator = AssignmentAccumulator(level_names)
            accumulator.merge_summary(stats)
            groups[key] = accumulator
        _plot_level_heatmap(
            split_dir / "plots" / ("assigned_gt_by_%s_level.png" % name),
            groups, level_names, title, sort_numeric_bins=True,
        )
        for class_name in summary["selected_classes"]:
            class_groups = OrderedDict()
            class_plot_dir = split_dir / "classes" / class_name / "plots"
            class_plot_dir.mkdir(parents=True, exist_ok=True)
            prefix = class_name + " | "
            for key, stats in summary["groups"]["class_" + name].items():
                if key.startswith(prefix):
                    accumulator = AssignmentAccumulator(level_names)
                    accumulator.merge_summary(stats)
                    class_groups[key[len(prefix):]] = accumulator
            _plot_level_heatmap(
                class_plot_dir / ("assigned_gt_by_%s_level.png" % name),
                class_groups, level_names,
                "%s: Assigned GT by %s" % (
                    class_name, "object size" if name == "size" else "depth"
                ),
                sort_numeric_bins=True,
            )


def _write_aggregate_reports(output_dir, summary, no_plots, dimensions=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _write_ltrb_report(output_dir, summary["max_ltrb_distribution"], no_plots)
    level_names = tuple("P%d" % (index + 3) for index in range(len(summary["strides"])))
    groups = {}
    for name, entries in summary["groups"].items():
        groups[name] = OrderedDict()
        for key, stats in entries.items():
            accumulator = AssignmentAccumulator(level_names)
            accumulator.merge_summary(stats)
            groups[name][key] = accumulator
    for filename, name, key_names in (
        ("size_level.csv", "size", ("size_bin",)),
        ("class_level.csv", "class", ("class_name",)),
        ("depth_level.csv", "depth", ("depth_bin",)),
        ("focal_level.csv", "focal", ("focal_group",)),
        ("class_size_level.csv", "class_size", ("class_name", "size_bin")),
        ("class_depth_level.csv", "class_depth", ("class_name", "depth_bin")),
        ("class_focal_level.csv", "class_focal", ("class_name", "focal_group")),
        ("depth_size_level.csv", "depth_size", ("depth_bin", "size_bin")),
        ("focal_size_level.csv", "focal_size", ("focal_group", "size_bin")),
    ):
        converted = OrderedDict(
            (tuple(key.split(" | ")), value) if len(key_names) > 1 else (key, value)
            for key, value in groups[name].items()
        )
        _write_group_table(output_dir / filename, converted, key_names, level_names)
    if not no_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        overall = AssignmentAccumulator(level_names)
        overall.merge_summary(summary["overall"])
        _plot_positive_buckets(plot_dir / "positive_count_distribution.png", overall)
        for name, title in (
            ("size", "Assigned GT distribution by object size"),
            ("depth", "Assigned GT distribution by depth"),
        ):
            _plot_level_heatmap(
                plot_dir / ("assigned_gt_by_%s_level.png" % name),
                groups[name], level_names, title,
                sort_numeric_bins=True,
            )
        if dimensions is not None:
            widths, heights, _, _ = dimensions
            if len(widths) != summary["overall"]["gt_count"]:
                raise ValueError("Aggregate per-GT rows do not match the summary")
            _plot_bbox_width_height(
                plot_dir / "bbox_width_height_scatter.png", widths, heights,
            )
            _plot_bbox_width_height_density(plot_dir, widths, heights, "All splits")
    _write_aggregate_class_reports(output_dir, summary, no_plots, dimensions)


def _refresh_plots_from_existing(root, summaries, no_plots, refresh_splits=True):
    """Build new density plots and all-split reports from saved per-GT CSVs."""
    for summary in summaries:
        if "max_ltrb_distribution" not in summary:
            split_dir = root / summary["split"]
            summary["max_ltrb_distribution"] = _ltrb_from_existing_per_gt(
                split_dir / "per_gt.csv", summary["selected_classes"],
            )
            with (split_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        if refresh_splits:
            _write_ltrb_report(
                root / summary["split"], summary["max_ltrb_distribution"], no_plots,
            )
            for class_name in summary["selected_classes"]:
                _write_ltrb_report(
                    root / summary["split"] / "classes" / class_name,
                    summary["max_ltrb_distribution"], no_plots, class_name,
                )
    combined = _merge_split_summaries(summaries)
    if no_plots:
        _write_aggregate_reports(root / "all", combined, True)
        return
    class_names = summaries[0]["selected_classes"]
    if refresh_splits:
        for summary in summaries:
            split = summary["split"]
            split_dir = root / split
            dimensions = _load_box_dimensions([split_dir / "per_gt.csv"], class_names)
            widths, heights, class_widths, class_heights = dimensions
            if len(widths) != summary["overall"]["gt_count"]:
                raise ValueError("%s per-GT rows do not match the summary" % split)
            plot_dir = split_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            _plot_saved_split_heatmaps(split_dir, summary)
            _plot_bbox_width_height_density(plot_dir, widths, heights, split)
            for class_name in class_names:
                class_plot_dir = split_dir / "classes" / class_name / "plots"
                class_plot_dir.mkdir(parents=True, exist_ok=True)
                _plot_bbox_width_height_density(
                    class_plot_dir, class_widths[class_name],
                    class_heights[class_name], "%s / %s" % (split, class_name),
                )
    paths = [root / split / "per_gt.csv" for split in ("train", "val", "test")]
    dimensions = _load_box_dimensions(paths, class_names)
    _write_aggregate_reports(root / "all", combined, False, dimensions)


def _numeric_bin_lower_edge(label):
    try:
        return float(str(label).split("-", 1)[0].rstrip("pxm+"))
    except ValueError:
        return math.inf


def _plot_level_heatmap(path, groups, level_names, title, sort_numeric_bins=False):
    labels = list(groups)
    if sort_numeric_bins:
        labels.sort(key=_numeric_bin_lower_edge)
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


def _plot_bbox_width_height_density(plot_dir, widths, heights, title):
    """Show box counts for fixed width/height bins, including a 3D count axis."""
    if not widths:
        return
    edges = np.asarray(BOX_DENSITY_EDGES, dtype=np.float64)
    counts, _, _ = np.histogram2d(widths, heights, bins=(edges, edges))
    counts = counts.astype(np.int64)
    if int(counts.sum()) != len(widths):
        raise ValueError("BBox density bins did not cover every GT box")
    labels = [
        "%d-%d" % (edges[index], edges[index + 1])
        if math.isfinite(edges[index + 1]) else "%d+" % edges[index]
        for index in range(len(edges) - 1)
    ]
    with (plot_dir / "bbox_width_height_density.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("width_bin_px", "height_bin_px", "count", "percent"))
        for width_index, width_label in enumerate(labels):
            for height_index, height_label in enumerate(labels):
                count = int(counts[width_index, height_index])
                writer.writerow((
                    width_label, height_label, count, count / len(widths) * 100.0,
                ))

    display = np.ma.masked_equal(counts.T, 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        display, origin="lower", cmap="viridis", aspect="auto",
        norm=LogNorm(vmin=1, vmax=max(int(counts.max()), 2)),
    )
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("BBox width bin (input pixels)")
    ax.set_ylabel("BBox height bin (input pixels)")
    ax.set_title("%s: bbox width-height counts (n=%d)" % (title, len(widths)))
    for index in np.argsort(counts.ravel())[-12:]:
        width_index, height_index = np.unravel_index(index, counts.shape)
        count = counts[width_index, height_index]
        if count:
            ax.text(
                width_index, height_index, "%.1f%%" % (count / len(widths) * 100),
                ha="center", va="center", fontsize=7, color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )
    fig.colorbar(image, ax=ax, label="GT box count (log color scale)")
    fig.tight_layout()
    fig.savefig(plot_dir / "bbox_width_height_density_heatmap.png", dpi=180)
    plt.close(fig)

    width_index, height_index = np.nonzero(counts)
    heights_3d = counts[width_index, height_index].astype(float)
    colors = plt.cm.viridis(
        np.log1p(heights_3d) / np.log1p(max(int(counts.max()), 1))
    )
    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(111, projection="3d")
    axis.bar3d(
        width_index, height_index, np.zeros_like(heights_3d),
        0.8, 0.8, heights_3d, color=colors, shade=True,
    )
    ticks = list(range(0, len(labels), 2))
    axis.set_xticks([index + 0.4 for index in ticks])
    axis.set_xticklabels([str(int(edges[index])) for index in ticks])
    axis.set_yticks([index + 0.4 for index in ticks])
    axis.set_yticklabels([str(int(edges[index])) for index in ticks])
    axis.set_xlabel("BBox width bin lower edge (px)", labelpad=12)
    axis.set_ylabel("BBox height bin lower edge (px)", labelpad=12)
    axis.set_zlabel("GT box count", labelpad=10)
    axis.set_title("%s: bbox width-height count surface (n=%d)" % (title, len(widths)))
    axis.view_init(elev=28, azim=-60)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.92)
    fig.savefig(plot_dir / "bbox_width_height_density_3d.png", dpi=180)
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


def analyze(args, config=None, split=None, output_dir=None):
    config = config if config is not None else load_config(args.config)
    split = split or args.split
    dataset = _build_dataset(config, split)
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

    output_dir = Path(output_dir if output_dir is not None else args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(exist_ok=True)

    groups = {
        "size": OrderedDict(),
        "class": OrderedDict(
            (name, AssignmentAccumulator(level_names))
            for name in classes if name in selected_classes
        ),
        "depth": OrderedDict(),
        "focal": OrderedDict(),
        "class_size": OrderedDict(),
        "class_depth": OrderedDict(),
        "class_focal": OrderedDict(),
        "depth_size": OrderedDict(),
        "focal_size": OrderedDict(),
    }
    overall = AssignmentAccumulator(level_names)
    processed_images = 0
    skipped_gt = 0
    bbox_widths = []
    bbox_heights = []
    class_widths = {name: [] for name in selected_classes}
    class_heights = {name: [] for name in selected_classes}
    ltrb_distribution = _empty_ltrb_distribution(selected_classes, level_names)

    per_gt_fields = [
        "sample_token", "dataset_index", "gt_index", "class_name", "depth_m",
        "box_width_px", "box_height_px", "box_area_px", "box_max_side_px",
        "box_center_max_ltrb_px",
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
                for level_name, points, stride, regress_range in zip(
                    level_names, points_by_level, strides, regress_ranges
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
                    if bool(positive.any()):
                        positive_points = points[positive]
                        matched_gt = matched[positive]
                        matched_boxes = target["boxes2d"][matched_gt]
                        x, y = positive_points[:, 0], positive_points[:, 1]
                        max_ltrb = torch.stack((
                            x - matched_boxes[:, 0], y - matched_boxes[:, 1],
                            matched_boxes[:, 2] - x, matched_boxes[:, 3] - y,
                        ), dim=1).max(dim=1).values.numpy()
                        matched_classes = target["labels"][matched_gt].numpy()
                        for class_name in selected_classes:
                            class_index = classes.index(class_name)
                            class_values = max_ltrb[matched_classes == class_index]
                            _add_ltrb_values(
                                ltrb_distribution["assigned_points"]["by_class"][class_name],
                                class_values,
                            )
                            if len(class_values):
                                _add_ltrb_values(
                                    ltrb_distribution["assigned_points"]["overall"],
                                    class_values,
                                )
                                _add_ltrb_values(
                                    ltrb_distribution["assigned_points"]["by_level"][level_name],
                                    class_values,
                                )
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
                    class_widths[class_name].append(width)
                    class_heights[class_name].append(height)
                    max_side = max(width, height)
                    box_center_max_ltrb = max_side / 2.0
                    _add_ltrb_value(
                        ltrb_distribution["box_center_gt"]["overall"],
                        box_center_max_ltrb,
                    )
                    _add_ltrb_value(
                        ltrb_distribution["box_center_gt"]["by_class"][class_name],
                        box_center_max_ltrb,
                    )
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
                    _add_group(groups["class_depth"], (class_name, depth_bin), level_names, candidate_counts, assigned_counts)
                    _add_group(groups["class_focal"], (class_name, focal_group), level_names, candidate_counts, assigned_counts)
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
                        "box_center_max_ltrb_px": box_center_max_ltrb,
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
        ("class_depth_level.csv", "class_depth", ("class_name", "depth_bin")),
        ("class_focal_level.csv", "class_focal", ("class_name", "focal_group")),
        ("depth_size_level.csv", "depth_size", ("depth_bin", "size_bin")),
        ("focal_size_level.csv", "focal_size", ("focal_group", "size_bin")),
    )
    for filename, group_name, key_names in table_specs:
        _write_group_table(
            output_dir / filename, groups[group_name], key_names, level_names
        )

    summary = {
        "config": str(Path(args.config).resolve()),
        "split": split,
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
        "max_ltrb_distribution": ltrb_distribution,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _write_ltrb_report(output_dir, ltrb_distribution, args.no_plots)

    if not args.no_plots:
        _plot_positive_buckets(plots_dir / "positive_count_distribution.png", overall)
        _plot_bbox_width_height(
            plots_dir / "bbox_width_height_scatter.png",
            bbox_widths,
            bbox_heights,
        )
        _plot_bbox_width_height_density(
            plots_dir, bbox_widths, bbox_heights, split,
        )
        _plot_level_heatmap(
            plots_dir / "assigned_gt_by_size_level.png",
            groups["size"], level_names, "Assigned GT distribution by object size",
            sort_numeric_bins=True,
        )
        _plot_level_heatmap(
            plots_dir / "assigned_gt_by_depth_level.png",
            groups["depth"], level_names, "Assigned GT distribution by depth",
            sort_numeric_bins=True,
        )

    _write_class_reports(
        output_dir,
        [name for name in classes if name in selected_classes],
        groups, level_names, class_widths, class_heights,
        {
            "config": summary["config"], "split": split,
            "input_size": summary["input_size"],
            "strides": strides, "regress_ranges": regress_ranges,
            "center_radius": args.center_radius,
        },
        args.no_plots, ltrb_distribution,
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
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument(
        "--split", default="train", choices=("train", "val", "test", "all"),
        help="Dataset split; all writes train/, val/, test/ and all/summary.json",
    )
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
    parser.add_argument(
        "--from-existing", action="store_true",
        help="With --split all, rebuild reports from existing split summaries and per_gt.csv files",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.center_radius <= 0:
        raise ValueError("--center-radius must be positive")
    if args.reference_focal <= 0 or args.focal_group_width <= 0:
        raise ValueError("focal values must be positive")
    if args.from_existing and args.split != "all":
        raise ValueError("--from-existing requires --split all")
    if args.split == "all":
        config = load_config(args.config)
        root = Path(args.output_dir)
        summaries = []
        for split in ("train", "val", "test"):
            if args.from_existing:
                with (root / split / "summary.json").open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                if summary["config"] != str(Path(args.config).resolve()):
                    raise ValueError("Existing %s summary uses a different config" % split)
            else:
                summary = analyze(args, config=config, split=split, output_dir=root / split)
            summaries.append(summary)
        _refresh_plots_from_existing(
            root, summaries, args.no_plots, refresh_splits=args.from_existing,
        )
        print("Combined train/val/test reports: %s" % (root / "all").resolve())
    else:
        analyze(args)


if __name__ == "__main__":
    main()
