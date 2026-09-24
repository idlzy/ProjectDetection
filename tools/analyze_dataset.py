#!/usr/bin/env python3
"""Generate numerical and visual statistics for an MW3D-layout dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from project_detection.config import load_config


SPLIT_NAMES = ("train", "val", "test")
VALID_FRONT_LEFT_FLAGS = {"1", "true", "yes"}
MISSING_ANN_BATCH = "<missing>"
VISIBILITY_MISSING_LABEL = "<missing_label>"

# Default selection. Edit this tuple when a fixed subset of annotation batches
# is required, or pass repeated --ann-batch arguments for a one-off run.
SELECTED_ANN_BATCHES = (
    # "0727-D105-3D-国内城区道路-白天-878（258有效）-re重新回流",
    # "0727-D107-3D-国内城区道路-白天-1911（840有效）-re重新回流",
    # "0727-D108-3D-国内城区道路-白天-1253（1116有效）-re重新回流",
    # "0727-D109-3D-国内城区道路-白天-1187（530有效）-re重新回流",
    # "0727-D110-3D-国内城区道路-白天-1350（1147有效）-re重新回流",
    # "0727-D111-3D-国内城区道路-白天-2338（1573有效）-re重新回流",
    # "0727-D112-3D-国内城区道路-白天-2107（1442有效）-re重新回流",
    # "0727-D113-3D-国内城区道路-白天-1044（605有效）-re重新回流",
    # "0727-D114-3D-国内城区道路-白天-2531（857有效）",
    # "0727-D115-3D-国内城区道路-白天-595（229有效）",
    # "0731-D116-3D行车-国内园区-白天-1177（975有效）",
    # "0804-D117-3d-国内测试二轮车-园区白天-798（721有效）",
    # "0804-D119-D120-3d-测试二轮车四轮车-城区-园区道路-白天-1249（1223有效）",
    "0807-D201-3D-国外道路-夜晚白天-1550（1014有效）",
    "0807-D202-3D-国外道路-白天-1183（816有效）",
    "0828-D210-3D-国外城区道路-白天-1976（1642有效）",
    "0911-D203-3D行车-国外乡村道路-白天-968（660有效）",
    "0911-D204-3D行车-国外乡村道路-白天-1938（1636有效）",
    "0911-D208-D209-D210-3D行车-国外城区道路-白天-1355（1086有效）",
    "0911-D212-3D行车-国外道路-白天-2697（1836有效）",
    "0911-D218-3D行车-国外道路-白天-2020（1718有效）",
    "0914-D211-D213-3D行车-国外城区道路-白天-1071（836有效）",
    # "D100-1725_re",
    # "D103-1357_re",
    # "D103-642_re",
    # "KITTI",
    # "ONCE_cam01",
    # "ONCE_cam03",
    # ONCE records in the current manifest do not contain ann_batch.
    # MISSING_ANN_BATCH,
)

DISTANCE_EDGES = np.asarray([0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, np.inf])
DISTANCE_LABELS = ("0-10m", "10-20m", "20-30m", "30-50m", "50-80m", "80-120m", "120m+")
COLORS = {"train": "#2864B7", "val": "#F28E2B", "test": "#2E9D69"}
SPLIT_DISPLAY_NAMES = {"train": "训练集", "val": "验证集", "test": "测试集"}
DISTANCE_DISPLAY_LABELS = ("0–10米", "10–20米", "20–30米", "30–50米", "50–80米", "80–120米", "120米以上")
TEXT = "#25364A"
GRID = "#D9E2EC"

CLASS_DISPLAY_NAMES = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "Special_vehicle": "特殊车辆",
    "Tricycle": "三轮车",
    "two_wheeled_vehicle": "二轮车",
    "Pedestrian": "行人",
    "Animal": "动物",
    "Obstacle": "障碍物",
    "Uncertain": "不确定",
    "Dense_object": "密集",
    "Other_obstacles": "其他障碍物",
}
CLASS_ORDER = tuple(CLASS_DISPLAY_NAMES)
SUBCLASS_DISPLAY_NAMES = {
    "car": {
        "car": "轿车", "suv": "SUV", "van": "面包车", "mpv": "MPV",
        "Iveco": "依维柯", "ambulance": "救护车", "police_car": "警车",
        "Other_car": "其他车辆",
    },
    "truck": {
        "Small_truck": "小型卡车/货车", "Pickup": "皮卡",
        "Medium_trucks": "中型卡车/货车", "Large_trucks": "大型卡车/货车",
        "Large_trucks_front": "大型卡车车头", "Large_trucks_rear": "大型卡车车尾",
        "Special_trucks": "特殊卡车", "fire_engine": "消防车",
    },
    "bus": {"Largebuses": "大型客车/公交车", "Minibuses": "小型客车"},
    "Special_vehicle": {"excavator": "挖掘机等履带式车辆", "trailer": "拖车", "Other": "其他", "Other_truck": "其他"},
    "two_wheeled_vehicle": {
        "Manned_motorcycle": "摩托车（有人）", "Unmanned_motorcycles": "摩托车（无人）",
        "Manned_bicycle": "自行车（有人）", "Unmanned_bicycles": "自行车（无人）",
        "Manned_electric_vehicle": "电动车（有人）",
        "Unmanned_electric_vehicles": "电动车（无人）",
    },
    "Pedestrian": {
        "adult": "成人", "children": "儿童", "Squatting_adult": "成人（蹲坐）",
        "Adults_pushing_carts": "成人（携带附件）",
    },
    "Animal": {"cats": "猫", "dogs": "狗"},
    "Obstacle": {
        "Cone_barrel": "锥桶", "Intersection_marker_post": "道口标柱",
        "Isolation_stone_pile": "隔离石桩", "Crash_barrels": "防撞桶",
        "Water_horse": "水马", "Triangle_tiles": "三角牌", "fence": "栅栏",
    },
    "Dense_object": {
        "A_dense_group_of_bicycles": "密集的自行车群",
        "A_dense_group_of_electric_vehicles": "密集的电动车群",
    },
    "Other_obstacles": {"Other_obstacles": "其他障碍物"},
}
SUBCLASS_ORDER = {
    class_name: tuple(subtypes) for class_name, subtypes in SUBCLASS_DISPLAY_NAMES.items()
}
SUBCLASS_GROUP_COLORS = (
    "#386CB0", "#F07C34", "#2F9C69", "#8C63B8", "#D45656", "#168C8C",
    "#C39428", "#6487A8", "#9A6571", "#6B8E23", "#5573A8", "#A76539",
)


def _visibility_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _visibility_key(number):
    return str(int(number)) if number.is_integer() else format(number, ".12g")


class VisibilityAccumulator:
    """Accumulate visibility quality for all objects and per class."""

    def __init__(self):
        self.total = 0
        self.missing = 0
        self.invalid = 0
        self.values = Counter()
        self.invalid_values = Counter()
        self.value_sum = 0.0
        self.by_class = {}

    def add(self, obj, include_class=True):
        self.total += 1
        if "visibility" not in obj:
            self.missing += 1
        else:
            raw_value = obj["visibility"]
            number = _visibility_number(raw_value)
            if number is None:
                self.invalid += 1
                self.invalid_values[repr(raw_value)] += 1
            else:
                self.values[_visibility_key(number)] += 1
                self.value_sum += number
        if include_class:
            class_name = str(obj.get("label") or VISIBILITY_MISSING_LABEL)
            self.by_class.setdefault(class_name, VisibilityAccumulator()).add(
                obj, include_class=False
            )

    def as_dict(self):
        valid = sum(self.values.values())
        result = {
            "total_objects": self.total,
            "valid_visibility": valid,
            "missing_visibility": self.missing,
            "invalid_visibility": self.invalid,
            "valid_ratio": valid / self.total if self.total else 0.0,
            "mean_visibility": self.value_sum / valid if valid else None,
            "distribution": {
                value: {
                    "count": count,
                    "ratio_of_total": count / self.total if self.total else 0.0,
                    "ratio_of_valid": count / valid if valid else 0.0,
                }
                for value, count in sorted(
                    self.values.items(), key=lambda item: float(item[0])
                )
            },
        }
        if self.invalid_values:
            result["invalid_values"] = dict(sorted(self.invalid_values.items()))
        if self.by_class:
            result["by_class"] = {
                name: accumulator.as_dict()
                for name, accumulator in sorted(self.by_class.items())
            }
        return result


def _register_cjk_font():
    """Register a CJK font explicitly; Matplotlib's cache may miss TTC files."""
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            return font_manager.FontProperties(fname=candidate).get_name()
    return "DejaVu Sans"


PLOT_FONT_FAMILY = _register_cjk_font()

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F9FC",
        "axes.edgecolor": "#BAC7D5",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": "#52657A",
        "ytick.color": "#52657A",
        "font.size": 10,
        "font.family": PLOT_FONT_FAMILY,
        "font.sans-serif": [PLOT_FONT_FAMILY, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
    }
)


def _record_batch(record):
    return record.get("ann_batch") or MISSING_ANN_BATCH


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON at %s:%d" % (path, line_number)) from error
            if not isinstance(record, dict) or not record.get("ann_rel"):
                raise ValueError("Invalid manifest record at %s:%d" % (path, line_number))
            records.append(record)
    return records


def _read_objects(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("objects")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Annotation must contain a list of objects: %s" % path)
    return payload


def _split_integrity(records_by_split):
    overlaps = {}
    has_leakage = False
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair = "%s/%s" % (left, right)
        overlaps[pair] = {}
        for key in ("sample_token", "split_group"):
            left_values = {
                record.get(key) for record in records_by_split[left]
                if record.get(key) is not None
            }
            right_values = {
                record.get(key) for record in records_by_split[right]
                if record.get(key) is not None
            }
            shared = sorted(left_values & right_values)
            overlaps[pair][key] = {
                "count": len(shared),
                "examples": shared[:20],
            }
            has_leakage |= bool(shared)
    return {
        "has_split_leakage": has_leakage,
        "manifest_frames": {
            split: len(records_by_split[split]) for split in SPLIT_NAMES
        },
        "ann_batch_counts": {
            split: len({_record_batch(record) for record in records_by_split[split]})
            for split in SPLIT_NAMES
        },
        "overlaps": overlaps,
    }


def _is_front_left(obj):
    value = obj.get("source", {}).get("pinhole_left", 0)
    return str(value).strip().lower() in VALID_FRONT_LEFT_FLAGS


def _finite_xyz(mapping):
    try:
        values = tuple(float(mapping[axis]) for axis in "xyz")
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _subclass(obj, class_name):
    value = obj.get(class_name)
    if value is None or isinstance(value, (dict, list)) or not str(value).strip():
        return "<unspecified>"
    return str(value).strip()


def _describe(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: None for key in ("count", "mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")}
    percentiles = np.percentile(array, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(np.max(array)),
    }


def _percentage(count, total):
    return round(float(count) / float(total) * 100.0, 6) if total else 0.0


def _distribution_percentages(counts):
    total = sum(counts.values())
    return {
        name: _percentage(count, total) for name, count in counts.items()
    }


def collect_statistics(data_root, selected_batches):
    selected_batch_order = list(dict.fromkeys(selected_batches))
    selected_batches = set(selected_batch_order)
    split_frames = Counter()
    split_clips = Counter()
    split_missing_clip_ids = Counter()
    split_boxes = Counter()
    class_by_split = defaultdict(Counter)
    subclass_by_split = defaultdict(Counter)
    batch_frames = Counter()
    batch_boxes = Counter()
    invalid_geometry = Counter()
    values = defaultdict(list)
    value_splits = []
    value_classes = []

    records_by_split = {}
    for split in SPLIT_NAMES:
        manifest = data_root / "splits" / (split + "_frames.jsonl")
        if not manifest.is_file():
            raise FileNotFoundError("Missing split manifest: %s" % manifest)
        records_by_split[split] = _read_jsonl(manifest)

    integrity = _split_integrity(records_by_split)
    visibility_accumulators = {
        split: {
            "all": VisibilityAccumulator(),
            "front_left": VisibilityAccumulator(),
        }
        for split in SPLIT_NAMES
    }

    for split in SPLIT_NAMES:
        records = records_by_split[split]
        selected_records = [
            record for record in records
            if _record_batch(record) in selected_batches
        ]
        split_frames[split] = len(selected_records)
        clip_ids = {
            str(record["split_group"])
            for record in selected_records
            if record.get("split_group") not in (None, "")
        }
        split_clips[split] = len(clip_ids)
        split_missing_clip_ids[split] = sum(
            record.get("split_group") in (None, "")
            for record in selected_records
        )
        for index, record in enumerate(records, 1):
            batch = _record_batch(record)
            selected_record = batch in selected_batches
            if selected_record:
                batch_frames[(batch, split)] += 1
            annotation = data_root / record["ann_rel"]
            if not annotation.is_file():
                raise FileNotFoundError("Missing annotation: %s" % annotation)
            for obj in _read_objects(annotation):
                visibility_accumulators[split]["all"].add(obj)
                is_front_left = _is_front_left(obj)
                if is_front_left:
                    visibility_accumulators[split]["front_left"].add(obj)
                if not selected_record or not is_front_left:
                    continue
                class_name = str(obj.get("label") or "<missing_label>")
                split_boxes[split] += 1
                class_by_split[class_name][split] += 1
                subclass_by_split[(class_name, _subclass(obj, class_name))][split] += 1
                batch_boxes[(batch, split)] += 1

                center = _finite_xyz(obj.get("center", {}))
                size = _finite_xyz(obj.get("size", {}))
                if center is None:
                    invalid_geometry["invalid_center"] += 1
                    continue
                if size is None or any(dimension <= 0.0 for dimension in size):
                    invalid_geometry["invalid_size"] += 1
                    continue
                length, width, height = size
                x, y, z = center
                values["length"].append(length)
                values["width"].append(width)
                values["height"].append(height)
                values["volume"].append(length * width * height)
                values["bev_distance"].append(math.hypot(x, y))
                values["center_distance_3d"].append(math.sqrt(x * x + y * y + z * z))
                value_splits.append(split)
                value_classes.append(class_name)
            if index % 5000 == 0:
                print(
                    "%s: processed %d/%d frames" % (split, index, len(records)),
                    flush=True,
                )

    if sum(split_frames.values()) == 0:
        raise ValueError("No manifest records match the selected ann_batch values")

    arrays = {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}
    split_array = np.asarray(value_splits, dtype=object)
    class_array = np.asarray(value_classes, dtype=object)
    class_names = sorted(class_by_split, key=lambda name: -sum(class_by_split[name].values()))
    subtype_names = sorted(
        subclass_by_split,
        key=lambda item: -sum(subclass_by_split[item].values()),
    )

    distance_by_split = {}
    distance_by_class = {}
    for split in SPLIT_NAMES:
        counts, _ = np.histogram(arrays["bev_distance"][split_array == split], bins=DISTANCE_EDGES)
        distance_by_split[split] = dict(zip(DISTANCE_LABELS, map(int, counts)))
    for class_name in class_names:
        counts, _ = np.histogram(arrays["bev_distance"][class_array == class_name], bins=DISTANCE_EDGES)
        distance_by_class[class_name] = dict(zip(DISTANCE_LABELS, map(int, counts)))
    distance_percentage_by_split = {
        split: _distribution_percentages(counts)
        for split, counts in distance_by_split.items()
    }
    distance_percentage_by_class = {
        class_name: _distribution_percentages(counts)
        for class_name, counts in distance_by_class.items()
    }

    geometry_summary = {
        name: _describe(array) for name, array in arrays.items()
    }
    geometry_by_split = {
        split: {
            name: _describe(array[split_array == split]) for name, array in arrays.items()
        }
        for split in SPLIT_NAMES
    }
    geometry_by_class = {
        class_name: {
            name: _describe(array[class_array == class_name]) for name, array in arrays.items()
        }
        for class_name in class_names
    }

    report = {
        "schema_version": 3,
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_root": str(data_root),
        "integrity": integrity,
        "scope": {
            "object_filter": "source.pinhole_left in [1, true, yes]",
            "bbox_size": "annotation size.x/y/z interpreted as length/width/height in metres",
            "distance": "vehicle-coordinate BEV distance sqrt(center.x^2 + center.y^2) in metres",
            "note": "Counts are annotation-level FrontLeft boxes; image projection clipping is not applied.",
        },
        "selected_ann_batches": selected_batch_order,
        "splits": {
            split: {
                "frames": split_frames[split],
                "clips": split_clips[split],
                "missing_clip_ids": split_missing_clip_ids[split],
                "bboxes": split_boxes[split],
            }
            for split in SPLIT_NAMES
        },
        "totals": {
            "frames": sum(split_frames.values()),
            "clips": sum(split_clips.values()),
            "missing_clip_ids": sum(split_missing_clip_ids.values()),
            "bboxes": sum(split_boxes.values()),
            "geometry_valid_bboxes": len(split_array),
            "invalid_geometry": dict(invalid_geometry),
        },
        "class_bbox_counts": {
            name: {
                **{
                    key: value
                    for split in SPLIT_NAMES
                    for key, value in (
                        (split, class_by_split[name][split]),
                        (
                            "%s_percentage" % split,
                            _percentage(
                                class_by_split[name][split], split_boxes[split]
                            ),
                        ),
                    )
                },
                "total": sum(class_by_split[name].values()),
                "total_percentage": _percentage(
                    sum(class_by_split[name].values()),
                    sum(split_boxes.values()),
                ),
            }
            for name in class_names
        },
        "subclass_bbox_counts": {
            class_name: {
                subtype: {
                    **{split: subclass_by_split[(class_name, subtype)][split] for split in SPLIT_NAMES},
                    "total": sum(subclass_by_split[(class_name, subtype)].values()),
                }
                for current_class, subtype in subtype_names if current_class == class_name
            }
            for class_name in class_names
        },
        "ann_batch_counts": {
            batch: {
                split: {
                    "frames": batch_frames[(batch, split)],
                    "bboxes": batch_boxes[(batch, split)],
                }
                for split in SPLIT_NAMES
            }
            for batch in selected_batch_order
        },
        "geometry": {
            "overall": geometry_summary,
            "by_split": geometry_by_split,
            "by_class": geometry_by_class,
        },
        "distance_bins": {
            "labels": list(DISTANCE_LABELS),
            "by_split": distance_by_split,
            "by_split_percentages": distance_percentage_by_split,
            "by_class": distance_by_class,
            "by_class_percentages": distance_percentage_by_class,
            "percentage_denominator": (
                "geometry-valid FrontLeft boxes in each split or class"
            ),
        },
        "visibility": {
            "scope_notes": {
                "all": "All objects present in annotation files.",
                "front_left": (
                    "Objects whose source.pinhole_left matches the dataset loader; "
                    "geometry filters are not applied."
                ),
            },
            "by_split": {
                split: {
                    scope: accumulator.as_dict()
                    for scope, accumulator in visibility_accumulators[split].items()
                }
                for split in SPLIT_NAMES
            },
        },
    }
    return report, arrays, split_array, class_array


def _decorate(axis, grid_axis="y"):
    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _save(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_split_summary(report, path):
    frames = [report["splits"][split]["frames"] for split in SPLIT_NAMES]
    clips = [report["splits"][split]["clips"] for split in SPLIT_NAMES]
    boxes = [report["splits"][split]["bboxes"] for split in SPLIT_NAMES]
    x = np.arange(len(SPLIT_NAMES))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, values, title, label in (
        (axes[0], frames, "各数据集帧数", "帧数"),
        (axes[1], clips, "各数据集片段数", "Clip数量"),
        (axes[2], boxes, "各数据集左前相机3D框数量", "3D框数量"),
    ):
        bars = axis.bar(x, values, color=[COLORS[name] for name in SPLIT_NAMES], width=0.62)
        axis.set_xticks(x, [SPLIT_DISPLAY_NAMES[name] for name in SPLIT_NAMES])
        axis.set_ylabel(label)
        axis.set_title(title, loc="left")
        _decorate(axis)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    figure.suptitle("数据集划分概览", x=0.04, ha="left", fontsize=16, fontweight="bold", color=TEXT)
    figure.subplots_adjust(top=0.82, wspace=0.28)
    _save(figure, path)


def _plot_class_counts(report, path):
    counts = report["class_bbox_counts"]
    names = list(counts)
    display_names = [CLASS_DISPLAY_NAMES.get(name, name) for name in names]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(11.5, max(5.5, 0.48 * len(names) + 1.8)))
    left = np.zeros(len(names))
    for split in SPLIT_NAMES:
        values = np.asarray([counts[name][split] for name in names])
        axis.barh(positions, values, left=left, color=COLORS[split], label=SPLIT_DISPLAY_NAMES[split], height=0.68)
        left += values
    axis.set_yticks(positions, display_names)
    axis.invert_yaxis()
    axis.set_xlabel("左前相机3D框数量")
    axis.set_title("各大类在数据集中的分布", loc="left", pad=14)
    axis.legend(loc="lower right")
    _decorate(axis, "x")
    grand_total = max(float(left.sum()), 1.0)
    for y, total in enumerate(left):
        axis.text(
            total,
            y,
            "  %s (%.2f%%)" % (f"{int(total):,}", total / grand_total * 100),
            va="center",
            fontsize=8,
        )
    _save(figure, path)


def _subclass_plot_rows(report):
    """Return rows in stable taxonomy order with Chinese display labels."""
    counts = report["subclass_bbox_counts"]
    class_rank = {name: index for index, name in enumerate(CLASS_ORDER)}
    class_names = sorted(counts, key=lambda name: (class_rank.get(name, len(class_rank)), name))
    rows = []
    for group_index, class_name in enumerate(class_names):
        subtypes = counts[class_name]
        subtype_rank = {
            name: index for index, name in enumerate(SUBCLASS_ORDER.get(class_name, ()))
        }
        subtype_names = sorted(
            subtypes,
            key=lambda name: (subtype_rank.get(name, len(subtype_rank)), name),
        )
        class_label = CLASS_DISPLAY_NAMES.get(class_name, class_name)
        for subtype in subtype_names:
            if subtype == "<unspecified>":
                subtype_label = ""
            else:
                subtype_label = SUBCLASS_DISPLAY_NAMES.get(class_name, {}).get(subtype, subtype)
            rows.append({
                "total": subtypes[subtype]["total"],
                "class_label": class_label,
                "subclass_label": subtype_label,
                "class_name": class_name,
                "group_index": group_index,
            })
    return rows


def _plot_subclasses(report, path):
    rows = _subclass_plot_rows(report)
    figure, axis = plt.subplots(figsize=(13, max(7, 0.38 * len(rows) + 1.8)))
    positions = np.arange(len(rows))
    values = [row["total"] for row in rows]
    colors = [SUBCLASS_GROUP_COLORS[row["group_index"] % len(SUBCLASS_GROUP_COLORS)] for row in rows]

    group_ranges = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["class_name"] != rows[start]["class_name"]:
            group_ranges.append((start, index - 1, rows[start]))
            start = index
    for group_number, (first, last, _) in enumerate(group_ranges):
        if group_number % 2:
            axis.axhspan(first - 0.5, last + 0.5, color="#EAF0F6", alpha=0.58, zorder=0)

    bars = axis.barh(positions, values, color=colors, height=0.65)
    axis.set_yticks(positions)
    axis.set_yticklabels([])
    axis.tick_params(axis="y", length=0)
    axis.invert_yaxis()
    axis.set_xlabel("左前相机3D框数量")
    axis.set_title("小类分布（按大类分组）", loc="left", pad=24)
    _decorate(axis, "x")

    label_transform = axis.get_yaxis_transform()
    axis.text(-0.235, 1.012, "大类", transform=axis.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold", color="#66788A")
    axis.text(-0.018, 1.012, "小类", transform=axis.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color="#66788A")
    axis.plot([-0.145, -0.145], [-0.5, len(rows) - 0.5], transform=label_transform, color="#C1CCD7", linewidth=1.0, clip_on=False)
    axis.plot([0.0, 0.0], [-0.5, len(rows) - 0.5], transform=label_transform, color="#AEBBC8", linewidth=1.0, clip_on=False)
    for first, last, row in group_ranges:
        center = (first + last) / 2.0
        axis.text(-0.235, center, row["class_label"], transform=label_transform, ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT, clip_on=False)
        if first:
            axis.plot([-0.29, 1.0], [first - 0.5, first - 0.5], transform=label_transform, color="#AAB7C4", linewidth=1.05, clip_on=False)
    for position, row in zip(positions, rows):
        if row["subclass_label"]:
            axis.text(-0.018, position, row["subclass_label"], transform=label_transform, ha="right", va="center", fontsize=9.5, color="#43586E", clip_on=False)
    for bar, value in zip(bars, values):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f"  {value:,}", va="center", fontsize=7.5)
    figure.subplots_adjust(left=0.31, right=0.965, top=0.94, bottom=0.07)
    _save(figure, path)


def _plot_geometry_distributions(arrays, split_array, path):
    fields = (("length", "长度（米）"), ("width", "宽度（米）"), ("height", "高度（米）"), ("volume", "体积（立方米）"))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (field, label) in zip(axes.flat, fields):
        values = arrays[field]
        upper = max(float(np.percentile(values, 99.5)), 1e-6)
        bins = np.linspace(0.0, upper, 45)
        for split in SPLIT_NAMES:
            clipped = np.minimum(values[split_array == split], upper)
            axis.hist(clipped, bins=bins, histtype="step", linewidth=2.0, color=COLORS[split], label=SPLIT_DISPLAY_NAMES[split])
        axis.set_xlabel(label + "（截断至 P99.5）")
        axis.set_ylabel("3D框数量")
        axis.set_title(label + "分布", loc="left")
        _decorate(axis)
    axes[0, 0].legend()
    figure.suptitle("3D框尺寸分布", x=0.04, ha="left", fontsize=16, fontweight="bold", color=TEXT)
    figure.subplots_adjust(top=0.91, hspace=0.3, wspace=0.24)
    _save(figure, path)


def _plot_distance(arrays, split_array, path):
    values = arrays["bev_distance"]
    upper = max(float(np.percentile(values, 99.5)), 1.0)
    bins = np.linspace(0.0, upper, 60)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for split in SPLIT_NAMES:
        split_values = values[split_array == split]
        axes[0].hist(np.minimum(split_values, upper), bins=bins, histtype="step", linewidth=2.0, color=COLORS[split], label=SPLIT_DISPLAY_NAMES[split])
        ordered = np.sort(split_values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / max(len(ordered), 1), color=COLORS[split], linewidth=2.0, label=SPLIT_DISPLAY_NAMES[split])
    axes[0].set_xlabel("BEV距离（米，截断至 P99.5）")
    axes[0].set_ylabel("3D框数量")
    axes[0].set_title("距离直方图", loc="left")
    axes[1].set_xlabel("BEV距离（米）")
    axes[1].set_ylabel("累计比例")
    axes[1].set_xlim(0, upper)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_title("距离累计分布", loc="left")
    for axis in axes:
        _decorate(axis)
        axis.legend()
    figure.suptitle("左前相机目标距离分布", x=0.04, ha="left", fontsize=16, fontweight="bold", color=TEXT)
    figure.subplots_adjust(top=0.84, wspace=0.25)
    _save(figure, path)


def _plot_distance_heatmap(report, path):
    matrix = report["distance_bins"]["by_class"]
    names = list(matrix)
    labels = report["distance_bins"]["labels"]
    counts = np.asarray([[matrix[name][label] for label in labels] for name in names], dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True)
    ratios = np.divide(counts, totals, out=np.zeros_like(counts, dtype=np.float64), where=totals > 0)
    figure, axis = plt.subplots(figsize=(12, max(5.8, 0.48 * len(names) + 2)))
    image = axis.imshow(ratios, cmap="YlGnBu", vmin=0, vmax=max(float(ratios.max()), 1e-6), aspect="auto")
    axis.set_xticks(np.arange(len(labels)), DISTANCE_DISPLAY_LABELS, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(names)), [CLASS_DISPLAY_NAMES.get(name, name) for name in names])
    axis.set_xlabel("车辆坐标系 BEV 距离")
    axis.set_title("各类别距离分布（按行归一化）", loc="left", pad=14)
    for row in range(len(names)):
        for column in range(len(labels)):
            value = ratios[row, column]
            if counts[row, column]:
                axis.text(column, row, f"{value:.0%}\n{counts[row, column]:,}", ha="center", va="center", fontsize=6.8, color="white" if value > ratios.max() * 0.48 else TEXT)
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.025, label="类内比例")
    _save(figure, path)


def _plot_class_geometry(report, path):
    by_class = report["geometry"]["by_class"]
    names = list(by_class)
    display_names = [CLASS_DISPLAY_NAMES.get(name, name) for name in names]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(14, max(5.5, 0.47 * len(names) + 1.8)))
    for axis, field, title, xlabel, color in (
        (axes[0], "volume", "各类别3D框体积", "体积（立方米）", "#7B61A8"),
        (axes[1], "bev_distance", "各类别BEV距离", "距离（米）", "#168C8C"),
    ):
        medians = np.asarray([by_class[name][field]["p50"] or 0.0 for name in names])
        lower = np.asarray([by_class[name][field]["p25"] or 0.0 for name in names])
        upper = np.asarray([by_class[name][field]["p75"] or 0.0 for name in names])
        axis.errorbar(medians, positions, xerr=np.vstack([medians - lower, upper - medians]), fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.6)
        axis.set_yticks(positions, display_names)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel + "（中位数及四分位距）")
        axis.set_title(title, loc="left")
        _decorate(axis, "x")
    figure.subplots_adjust(wspace=0.32)
    _save(figure, path)


def _plot_ann_batches(report, path):
    batches = report["ann_batch_counts"]
    rows = []
    for batch, split_values in batches.items():
        frames = sum(item["frames"] for item in split_values.values())
        boxes = sum(item["bboxes"] for item in split_values.values())
        rows.append((frames, boxes, batch))
    rows.sort(reverse=True)
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(17, max(8, 0.37 * len(rows) + 1.8)), sharey=True)
    labels = [_ann_batch_plot_label(row[2]) for row in rows]
    for axis, index, title, color in ((axes[0], 0, "帧数", "#3F6FB6"), (axes[1], 1, "左前相机3D框数量", "#CA6C3C")):
        values = [row[index] for row in rows]
        axis.barh(positions, values, color=color, height=0.64)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.set_title("各 ann_batch " + title, loc="left")
        _decorate(axis, "x")
    figure.subplots_adjust(wspace=0.12)
    _save(figure, path)


def _plot_visibility(report, path):
    by_split = report["visibility"]["by_split"]
    keys = sorted(
        {
            value
            for split in SPLIT_NAMES
            for value in by_split[split]["front_left"]["distribution"]
        },
        key=float,
    )
    labels = ["可见度 %s" % value for value in keys] + ["缺失", "无效"]
    positions = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 1.25), 5.6))
    for split_index, split in enumerate(SPLIT_NAMES):
        statistics = by_split[split]["front_left"]
        values = [
            statistics["distribution"].get(key, {}).get("count", 0)
            for key in keys
        ] + [statistics["missing_visibility"], statistics["invalid_visibility"]]
        offset = (split_index - 1) * width
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            color=COLORS[split],
            label=SPLIT_DISPLAY_NAMES[split],
        )
        for bar, value in zip(bars, values):
            if value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:,}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("左前相机目标数量")
    axis.set_title("标注可见度分布", loc="left", pad=14)
    axis.legend()
    _decorate(axis)
    _save(figure, path)


def _ann_batch_plot_label(batch):
    """Keep long Chinese batch names readable without losing batch identity."""
    if batch == MISSING_ANN_BATCH:
        return "缺少 ann_batch"
    if batch == "KITTI" or re.fullmatch(r"D\d+(?:-\d+)*_re", batch):
        return batch
    batch_ids = re.findall(r"D\d+", batch)
    valid = re.search(r"[（(](\d+)有效[）)]", batch)
    if batch_ids:
        label = "-".join(dict.fromkeys(batch_ids))
        if valid:
            label += " · %s帧有效" % valid.group(1)
        return label
    return batch if len(batch) <= 42 else batch[:39] + "..."


def _write_csv_files(report, output_dir):
    with (output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "frames", "clips", "missing_clip_ids", "bboxes"]
        )
        for split in SPLIT_NAMES:
            values = report["splits"][split]
            writer.writerow(
                [
                    split,
                    values["frames"],
                    values["clips"],
                    values["missing_clip_ids"],
                    values["bboxes"],
                ]
            )
    with (output_dir / "class_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        columns = ["class"]
        for split in SPLIT_NAMES:
            columns.extend([split, "%s_percentage" % split])
        columns.extend(["total", "total_percentage"])
        writer.writerow(columns)
        for name, values in report["class_bbox_counts"].items():
            row = [name]
            for split in SPLIT_NAMES:
                row.extend(
                    [values[split], values["%s_percentage" % split]]
                )
            row.extend([values["total"], values["total_percentage"]])
            writer.writerow(row)
    with (output_dir / "subclass_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "subclass", *SPLIT_NAMES, "total"])
        for class_name, subtypes in report["subclass_bbox_counts"].items():
            for subtype, values in subtypes.items():
                writer.writerow([class_name, subtype, *(values[split] for split in SPLIT_NAMES), values["total"]])
    with (output_dir / "geometry_by_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "metric", "count", "mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"])
        for class_name, metrics in report["geometry"]["by_class"].items():
            for metric, values in metrics.items():
                writer.writerow([class_name, metric, *(values[key] for key in ("count", "mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"))])
    with (output_dir / "ann_batch_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ann_batch", *("%s_frames" % split for split in SPLIT_NAMES), *("%s_bboxes" % split for split in SPLIT_NAMES), "total_frames", "total_bboxes"])
        for batch, split_values in report["ann_batch_counts"].items():
            frames = [split_values[split]["frames"] for split in SPLIT_NAMES]
            boxes = [split_values[split]["bboxes"] for split in SPLIT_NAMES]
            writer.writerow([batch, *frames, *boxes, sum(frames), sum(boxes)])
    with (output_dir / "distance_bins.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        labels = report["distance_bins"]["labels"]
        columns = ["scope", "name"]
        for label in labels:
            columns.extend([label, "%s_percentage" % label])
        columns.append("total")
        writer.writerow(columns)
        for split, values in report["distance_bins"]["by_split"].items():
            percentages = report["distance_bins"]["by_split_percentages"][split]
            row = ["split", split]
            for label in labels:
                row.extend([values[label], percentages[label]])
            writer.writerow([*row, sum(values.values())])
        for class_name, values in report["distance_bins"]["by_class"].items():
            percentages = report["distance_bins"]["by_class_percentages"][
                class_name
            ]
            row = ["class", class_name]
            for label in labels:
                row.extend([values[label], percentages[label]])
            writer.writerow([*row, sum(values.values())])
    with (output_dir / "split_integrity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split_pair", "key", "overlap_count", "examples"])
        for pair, values in report["integrity"]["overlaps"].items():
            for key, overlap in values.items():
                writer.writerow([
                    pair,
                    key,
                    overlap["count"],
                    json.dumps(overlap["examples"], ensure_ascii=False),
                ])
    with (output_dir / "visibility.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "split", "scope", "class", "total", "valid", "missing",
            "invalid", "valid_ratio", "mean", "distribution",
        ])
        for split, scopes in report["visibility"]["by_split"].items():
            for scope, statistics in scopes.items():
                rows = [("<all_classes>", statistics)] + list(
                    statistics.get("by_class", {}).items()
                )
                for class_name, values in rows:
                    writer.writerow([
                        split,
                        scope,
                        class_name,
                        values["total_objects"],
                        values["valid_visibility"],
                        values["missing_visibility"],
                        values["invalid_visibility"],
                        values["valid_ratio"],
                        values["mean_visibility"],
                        json.dumps(values["distribution"], ensure_ascii=False),
                    ])


def write_report(report, arrays, split_array, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_statistics.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    _write_csv_files(report, output_dir)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    # Remove the obsolete Top-30 plot so an updated report cannot be mistaken
    # for the previous truncated visualization.
    legacy_subclass_plot = plots / "subclass_counts_top30.png"
    if legacy_subclass_plot.is_file():
        legacy_subclass_plot.unlink()
    plotters = {
        "split_summary.png": lambda path: _plot_split_summary(report, path),
        "class_counts.png": lambda path: _plot_class_counts(report, path),
        "subclass_counts.png": lambda path: _plot_subclasses(report, path),
        "bbox_size_distributions.png": lambda path: _plot_geometry_distributions(arrays, split_array, path),
        "distance_distribution.png": lambda path: _plot_distance(arrays, split_array, path),
        "distance_by_class.png": lambda path: _plot_distance_heatmap(report, path),
        "class_geometry.png": lambda path: _plot_class_geometry(report, path),
        "ann_batch_counts.png": lambda path: _plot_ann_batches(report, path),
        "visibility_distribution.png": lambda path: _plot_visibility(report, path),
    }
    for filename, plotter in plotters.items():
        plotter(plots / filename)
    index = {
        "report": str(report_path),
        "csv": [str(path) for path in sorted(output_dir.glob("*.csv"))],
        "plots": [str(plots / name) for name in plotters],
    }
    (plots / "plots_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate MW3D splits and generate dataset, geometry and visibility "
            "statistics in one scan"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Read data.ready_root from a project config")
    source.add_argument("--data-root", type=Path, help="Dataset root containing splits/")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: outputs/dataset_statistics/<dataset-name>)",
    )
    parser.add_argument(
        "--ann-batch", action="append", default=None,
        help="Only include this ann_batch; repeat for multiple batches. Defaults to the fixed list in the script.",
    )
    parser.add_argument(
        "--allow-split-leakage",
        action="store_true",
        help="Write the report but return success when split overlap is detected",
    )
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
        data_root = Path(config["data"]["ready_root"]).expanduser().resolve()
    elif args.data_root:
        data_root = args.data_root.expanduser().resolve()
    else:
        data_root = Path(
            "/home/gy-zb-a-luziyang/datasets/mw3d_with_kitti"
        ).resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (Path("outputs/dataset_statistics") / data_root.name).resolve()
    )
    selected_batches = tuple(args.ann_batch) if args.ann_batch else SELECTED_ANN_BATCHES
    if not data_root.is_dir():
        parser.error("Dataset root does not exist: %s" % data_root)
    report, arrays, split_array, _ = collect_statistics(data_root, selected_batches)
    report_path = write_report(report, arrays, split_array, output_dir)
    print(json.dumps({
        "report": str(report_path),
        "output_dir": str(output_dir),
        "totals": report["totals"],
        "splits": report["splits"],
        "integrity": report["integrity"],
        "visibility": {
            split: {
                scope: {
                    key: values[key]
                    for key in (
                        "total_objects", "valid_visibility", "missing_visibility",
                        "invalid_visibility", "valid_ratio", "mean_visibility",
                    )
                }
                for scope, values in scopes.items()
            }
            for split, scopes in report["visibility"]["by_split"].items()
        },
    }, ensure_ascii=False, indent=2))
    if report["integrity"]["has_split_leakage"] and not args.allow_split_leakage:
        parser.exit(2, "error: dataset split leakage detected; see report integrity section\n")


if __name__ == "__main__":
    main()
