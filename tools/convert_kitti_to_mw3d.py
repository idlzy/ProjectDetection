#!/usr/bin/env python3
"""Convert KITTI object detection data to the local MW3D-ready layout."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import struct
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_MAP = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Cyclist": "two_wheeled_vehicle",
    "Tram": "bus",
}
SPLIT_NAMES = ("train", "val", "test")
CAMERA_TO_VEHICLE = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def camera_rotation_to_vehicle_yaw(rotation_y: float) -> float:
    """Convert KITTI camera-Y rotation to vehicle-frame Z yaw."""
    yaw = -float(rotation_y) - math.pi / 2.0
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def parse_calibration(path: Path) -> dict[str, np.ndarray]:
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, raw = line.partition(":")
            if separator:
                values[key] = np.asarray(
                    [float(item) for item in raw.split()], dtype=np.float64
                )
    if "P2" not in values or values["P2"].size != 12:
        raise ValueError("Missing 3x4 P2 matrix in %s" % path)
    projection = values["P2"].reshape(3, 4)
    intrinsic = projection[:, :3]
    if not np.allclose(
        intrinsic,
        [[intrinsic[0, 0], 0.0, intrinsic[0, 2]],
         [0.0, intrinsic[1, 1], intrinsic[1, 2]],
         [0.0, 0.0, 1.0]],
        atol=1e-3,
    ):
        raise ValueError("P2 has unsupported skew or scale in %s" % path)
    # P2 @ [X, 1] == K @ (X + camera_shift). Keeping this small shift makes
    # projections made by the MW3D loader agree with KITTI's rectified P2.
    camera_shift = np.linalg.solve(intrinsic, projection[:, 3])
    return {
        "projection": projection,
        "intrinsic": intrinsic,
        "camera_shift": camera_shift,
    }


def calibration_json(intrinsic: np.ndarray, width: int, height: int) -> dict:
    return {
        "front_left": {
            "camera_location": "front_left",
            "resolution_px": {"width": width, "height": height},
            "intrinsics": {
                "focal_length_px": {
                    "fx": float(intrinsic[0, 0]),
                    "fy": float(intrinsic[1, 1]),
                },
                "principal_point_px": {
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                },
                "distortion_coefficients": {
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                    "k3": 0.0,
                    "k4": 0.0,
                    "k5": 0.0,
                    "k6": 0.0,
                },
            },
            # Rz(-pi/2) @ Rx(-pi/2) maps KITTI camera coordinates
            # (right, down, forward) to vehicle coordinates (forward, left, up).
            "extrinsics": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": -math.pi / 2.0,
                "pitch": 0.0,
                "yaw": -math.pi / 2.0,
            },
        }
    }


def parse_annotations(path: Path, camera_shift: np.ndarray) -> tuple[list[dict], Counter]:
    objects = []
    counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for object_id, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 15:
                raise ValueError("Malformed KITTI annotation in %s: %s" % (path, line.rstrip()))
            source_label = fields[0]
            target_label = CLASS_MAP.get(source_label)
            counts[source_label] += 1
            if target_label is None:
                continue

            truncated = float(fields[1])
            occluded = int(fields[2])
            height, width, length = (float(value) for value in fields[8:11])
            bottom_center = np.asarray(
                [float(fields[11]), float(fields[12]), float(fields[13])],
                dtype=np.float64,
            )
            center_camera = bottom_center.copy()
            center_camera[1] -= height * 0.5
            center_camera += camera_shift
            center_vehicle = CAMERA_TO_VEHICLE @ center_camera
            rotation_y = float(fields[14])
            yaw_vehicle = camera_rotation_to_vehicle_yaw(rotation_y)
            objects.append(
                {
                    "id": object_id,
                    "label": target_label,
                    "center": dict(zip("xyz", center_vehicle.tolist())),
                    "size": {"x": length, "y": width, "z": height},
                    "rotation": {"x": 0.0, "y": 0.0, "z": yaw_vehicle},
                    "source": {"pinhole_left": "1"},
                    "visibility": max(0, 3 - occluded),
                    "kitti": {
                        "type": source_label,
                        "truncated": truncated,
                        "occluded": occluded,
                        "alpha": float(fields[3]),
                        "bbox": [float(value) for value in fields[4:8]],
                        "rotation_y": rotation_y,
                    },
                }
            )
    return objects, counts


def assign_splits(stems: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("split ratios must be non-negative and have a positive sum")
    normalized = np.asarray(ratios, dtype=np.float64) / sum(ratios)
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    raw_counts = normalized * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(-(raw_counts - counts))[: len(shuffled) - int(counts.sum())]:
        counts[index] += 1
    destinations = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts.tolist()):
        for stem in shuffled[start : start + count]:
            destinations[stem] = split_name
        start += count
    return destinations


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def link_or_copy(source: Path, target: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(source, target)
    else:
        os.symlink(source.resolve(), target)


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header: %s" % path)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def convert(args) -> dict:
    source_root = args.kitti_root.resolve()
    output_root = args.output_root.resolve()
    image_source = source_root / "image_2"
    label_source = source_root / "label_2"
    calibration_source = source_root / "calib"
    for path in (image_source, label_source, calibration_source):
        if not path.is_dir():
            raise FileNotFoundError("Missing KITTI directory: %s" % path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Output directory is not empty: %s" % output_root)

    image_output = output_root / "images" / "KITTI"
    annotation_output = output_root / "annotation" / "KITTI"
    calibration_output = output_root / "calibration" / "KITTI"
    split_output = output_root / "splits"
    for path in (image_output, annotation_output, calibration_output, split_output):
        path.mkdir(parents=True, exist_ok=True)

    images = sorted(image_source.glob("*.png"))
    if not images:
        raise ValueError("No PNG images found in %s" % image_source)
    stems = [path.stem for path in images]
    split_for_stem = assign_splits(stems, tuple(args.split_ratios), args.seed)
    records = []
    source_counts = Counter()
    target_counts = Counter()

    for image_path in images:
        stem = image_path.stem
        label_path = label_source / (stem + ".txt")
        source_calibration_path = calibration_source / (stem + ".txt")
        if not label_path.is_file() or not source_calibration_path.is_file():
            raise FileNotFoundError("Missing label or calibration for KITTI frame %s" % stem)
        width, height = read_png_size(image_path)
        calibration = parse_calibration(source_calibration_path)
        objects, frame_counts = parse_annotations(label_path, calibration["camera_shift"])
        source_counts.update(frame_counts)
        target_counts.update(obj["label"] for obj in objects)

        target_image = image_output / image_path.name
        link_or_copy(image_path, target_image, args.copy_images)
        write_json(annotation_output / (stem + ".json"), objects)
        write_json(
            calibration_output / (stem + ".json"),
            calibration_json(calibration["intrinsic"], width, height),
        )
        split_name = split_for_stem[stem]
        records.append(
            {
                "split_group": "KITTI__" + stem,
                "sample_token": "kitti__" + stem,
                "stem": stem,
                "image_rel": "images/KITTI/%s" % image_path.name,
                "ann_rel": "annotation/KITTI/%s.json" % stem,
                "calib_rel": "calibration/KITTI/%s.json" % stem,
                "ann_batch": "KITTI__" + stem,
                "source_dataset": "kitti",
                "split": split_name,
            }
        )

    with (output_root / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    split_counts = {}
    for split_name in SPLIT_NAMES:
        selected = [record for record in records if record["split"] == split_name]
        split_counts[split_name] = len(selected)
        with (split_output / (split_name + "_frames.jsonl")).open(
            "w", encoding="utf-8"
        ) as handle:
            for record in selected:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "images": len(images),
        "image_mode": "copy" if args.copy_images else "absolute_symlink",
        "seed": args.seed,
        "split_ratios": dict(zip(SPLIT_NAMES, args.split_ratios)),
        "split_counts": split_counts,
        "source_class_counts": dict(sorted(source_counts.items())),
        "converted_class_counts": dict(sorted(target_counts.items())),
        "ignored_class_counts": {
            name: count for name, count in sorted(source_counts.items()) if name not in CLASS_MAP
        },
        "class_map": CLASS_MAP,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert KITTI training data to the MW3D-ready JSON layout"
    )
    parser.add_argument(
        "--kitti-root",
        type=Path,
        default=Path("/home/gy-zb-a-luziyang/datasets/kitti/training"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/gy-zb-a-luziyang/datasets/kitti_mw3d"),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating absolute symbolic links",
    )
    args = parser.parse_args()
    print(json.dumps(convert(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
