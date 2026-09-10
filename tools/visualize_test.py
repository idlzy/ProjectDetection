#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.geometry import project_distorted, vehicle_box_corners
from project_detection.engine import build_loader, load_checkpoint
from project_detection.models import build_model
from project_detection.task import FCOS3DPostProcessor
from project_detection.visualization import (
    draw_bev as draw_shared_bev,
    draw_camera_view as draw_shared_camera_view,
)


def color_for_class(class_id):
    hue = int((class_id * 47) % 180)
    pixel = np.uint8([[[hue, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def draw_label(image, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.48, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(y, height + baseline + 2)
    cv2.rectangle(image, (x, y - height - baseline - 2), (x + width + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - baseline - 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def camera_box_corners(
    box, camera_to_vehicle_rotation=None, camera_to_vehicle_translation=None
):
    """Build camera-frame corners for the project's gravity-aligned 3D box.

    The stored yaw is derived from a vehicle-frame, ground-aligned heading.  A
    camera on the vehicle is generally rolled/pitched, so constructing a box
    by rotating around camera Y makes its vertical edges disagree with the
    road plane.  When calibration is supplied, recover the vehicle heading,
    construct the cuboid there, and transform all corners back to the camera.

    The calibration-free branch is retained for callers using conventional
    camera-frame boxes.
    """
    x, y, z, length, height, width, yaw = [
        float(value) for value in box[:7]
    ]
    if (camera_to_vehicle_rotation is None) != (
        camera_to_vehicle_translation is None
    ):
        raise ValueError("camera rotation and translation must be supplied together")
    if camera_to_vehicle_rotation is not None:
        r_c2v = np.asarray(camera_to_vehicle_rotation, dtype=np.float64)
        t_c2v = np.asarray(camera_to_vehicle_translation, dtype=np.float64)
        center_c = np.asarray([x, y, z], dtype=np.float64)
        center_v = r_c2v @ center_c + t_c2v

        # build_targets stores theta=-camera_yaw, where
        # theta=atan2(heading_camera.z, heading_camera.x).  Invert that exact
        # 2-D direction mapping instead of assuming a level camera.
        heading_projection = np.asarray(
            [
                [r_c2v[0, 0], r_c2v[1, 0]],
                [r_c2v[0, 2], r_c2v[1, 2]],
            ],
            dtype=np.float64,
        )
        try:
            heading_v = np.linalg.solve(
                heading_projection,
                np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64),
            )
        except np.linalg.LinAlgError:
            # Degenerate mounting geometry is unlikely, but least-squares
            # gives a stable diagnostic visualization instead of crashing.
            heading_v = np.linalg.lstsq(
                heading_projection,
                np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64),
                rcond=None,
            )[0]
        yaw_v = math.atan2(float(heading_v[1]), float(heading_v[0]))
        corners_v = vehicle_box_corners(
            center_v,
            np.asarray([length, width, height], dtype=np.float64),
            yaw_v,
        )
        return (r_c2v.T @ (corners_v - t_c2v.reshape(1, 3)).T).T

    local = np.asarray(
        [[-length / 2, -height / 2, -width / 2],
         [length / 2, -height / 2, -width / 2],
         [length / 2, -height / 2, width / 2],
         [-length / 2, -height / 2, width / 2],
         [-length / 2, height / 2, -width / 2],
         [length / 2, height / 2, -width / 2],
         [length / 2, height / 2, width / 2],
         [-length / 2, height / 2, width / 2]],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated_x = cosine * local[:, 0] - sine * local[:, 2]
    rotated_z = sine * local[:, 0] + cosine * local[:, 2]
    return np.stack(
        [rotated_x + x, local[:, 1] + y, rotated_z + z], axis=1
    )


CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def project_camera_box(box, target):
    corners = camera_box_corners(
        box,
        target["camera_to_vehicle_rotation"].numpy(),
        target["camera_to_vehicle_translation"].numpy(),
    )
    camera_matrix = target["camera_matrix"].numpy().astype(np.float64)
    distortion = target["distortion"].numpy().astype(np.float64)
    pixels, depth = project_distorted(corners, camera_matrix, distortion)
    return pixels / float(target["scale_factor"]), depth


def draw_projected_cuboid(image, pixels, depth, color, thickness=2):
    valid = (depth > 0.1) & np.isfinite(pixels).all(axis=1)
    safe_pixels = np.nan_to_num(pixels, nan=0.0, posinf=1e6, neginf=-1e6)
    rounded = np.clip(safe_pixels, -1e6, 1e6).round().astype(np.int32)
    for start, end in CUBOID_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                image, tuple(rounded[start]), tuple(rounded[end]),
                color, thickness, cv2.LINE_AA,
            )
    return rounded, valid


def draw_camera_view(
    image, target, result, classes, max_detections, draw_ground_truth=True
):
    if draw_ground_truth:
        for box, label in zip(target["boxes3d"].numpy(), target["labels"]):
            pixels, depth = project_camera_box(box, target)
            rounded, valid = draw_projected_cuboid(
                image, pixels, depth, (255, 120, 0), 3
            )
            if valid.any():
                x, y = rounded[valid].min(axis=0).tolist()
                draw_label(image, "GT " + classes[int(label)], (max(x, 0), max(y, 0)), (255, 120, 0))

    boxes = result["boxes3d"][:max_detections].cpu()
    scores = result["scores"][:max_detections].cpu()
    labels = result["labels"][:max_detections].cpu()
    depths = result["boxes3d"][:max_detections, 2].cpu()
    valid = result.get("geometry_valid")
    weights = result.get("depth_fusion_weight")
    for index, (box, score, label, depth_value) in enumerate(
        zip(boxes, scores, labels, depths)
    ):
        color = color_for_class(int(label))
        pixels, corner_depth = project_camera_box(box.numpy(), target)
        rounded, projected_valid = draw_projected_cuboid(
            image, pixels, corner_depth, color, 2
        )
        if not projected_valid.any():
            continue
        x, y = rounded[projected_valid].min(axis=0).tolist()
        suffix = ""
        if valid is not None and bool(valid[index]):
            suffix = " G w=%.2f" % float(weights[index])
        text = "%s %.2f z=%.1fm%s" % (
            classes[int(label)], float(score), float(depth_value), suffix
        )
        draw_label(image, text, (max(x, 0), max(y, 0)), color)
    return image


def bev_corners(box):
    x, z, length, width, yaw = [float(value) for value in box[[0, 2, 3, 5, 6]]]
    local = np.asarray(
        [[-length / 2, -width / 2], [length / 2, -width / 2],
         [length / 2, width / 2], [-length / 2, width / 2]],
        dtype=np.float32,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return local @ np.asarray([[cosine, -sine], [sine, cosine]]).T + [x, z]


def draw_bev(target, result, classes, max_detections, size=(700, 700)):
    height, width = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x_limit, z_limit = 40.0, 80.0

    def project(points):
        pixels = np.empty_like(points)
        pixels[:, 0] = width / 2 + points[:, 0] / x_limit * (width / 2 - 30)
        pixels[:, 1] = height - 30 - points[:, 1] / z_limit * (height - 60)
        return pixels.round().astype(np.int32)

    for distance in range(10, 81, 10):
        y = int(height - 30 - distance / z_limit * (height - 60))
        cv2.line(canvas, (25, y), (width - 25, y), (210, 210, 210), 1)
        cv2.putText(canvas, "%dm" % distance, (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (80, 80, 80), 1)
    cv2.line(canvas, (width // 2, 10), (width // 2, height - 15), (160, 160, 160), 1)
    cv2.circle(canvas, (width // 2, height - 30), 5, (0, 0, 0), -1)

    for box in target["boxes3d"].numpy():
        cv2.polylines(canvas, [project(bev_corners(box))], True, (255, 120, 0), 2)
    for box, score, label in zip(
        result["boxes3d"][:max_detections].cpu().numpy(),
        result["scores"][:max_detections].cpu().numpy(),
        result["labels"][:max_detections].cpu().numpy(),
    ):
        points = project(bev_corners(box))
        color = color_for_class(int(label))
        cv2.polylines(canvas, [points], True, color, 2)
        cv2.putText(canvas, "%.2f" % score, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    cv2.putText(canvas, "GT=cyan, prediction=class color", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1)
    return canvas


def evenly_spaced_indices(length, count):
    if count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def sequential_indices(length, start, count):
    start = max(0, min(int(start), length))
    end = length if count is None else min(length, start + max(0, int(count)))
    return list(range(start, end))


def safe_group_name(value):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(value)).strip() or "unknown"


def result_to_json(result, classes, max_detections):
    count = min(max_detections, len(result["scores"]))
    predictions = []
    for index in range(count):
        item = {
            "class": classes[int(result["labels"][index])],
            "score": float(result["scores"][index]),
            "box2d": result["boxes2d"][index].cpu().tolist(),
            "box3d": result["boxes3d"][index].cpu().tolist(),
        }
        if "depth_local" in result:
            item.update(
                depth_local=float(result["depth_local"][index]),
                depth_geometric=float(result["depth_geometric"][index]),
                depth_fusion_weight=float(result["depth_fusion_weight"][index]),
                geometry_valid=bool(result["geometry_valid"][index]),
            )
        predictions.append(item)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions on test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sampling", choices=("sequential", "evenly"), default="sequential")
    parser.add_argument("--layout", choices=("grouped", "paired"), default="grouped")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-bev", action="store_true")
    parser.add_argument("--draw-ground-truth", action="store_true")
    parser.add_argument("--skip-summary-json", action="store_true")
    parser.add_argument("--set", nargs="+", action="append", default=[])
    args = parser.parse_args()
    overrides = [item for group in args.set for item in group]
    config = load_config(args.config, overrides)
    device = torch.device(
        "cuda" if config["runtime"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, strict=True)
    model.eval()
    processor = FCOS3DPostProcessor(model, config)
    loader = build_loader(config, "test", max_samples=None)
    dataset = loader.dataset
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        config["experiment"]["output_dir"]
    ) / config["experiment"]["name"] / "visualizations" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sampling == "evenly":
        count = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
        indices = evenly_spaced_indices(len(dataset), count)
    else:
        indices = sequential_indices(len(dataset), args.start_index, args.max_images)
    if not indices:
        raise ValueError("No test frames selected")
    summary, errors = [], []
    generated = skipped = failed = 0
    for sequence, index in enumerate(indices):
        try:
            image_tensor, target = dataset[index]
            image_stem = Path(target["image_path"]).stem
            if args.layout == "grouped":
                group_dir = output_dir / safe_group_name(target["split_group"])
                camera_path = group_dir / (image_stem + "_pred.jpg")
                bev_path = group_dir / (image_stem + "_bev.jpg")
            else:
                stem = "%03d_%s" % (sequence, image_stem)
                group_dir = output_dir
                camera_path = group_dir / (stem + "_camera.jpg")
                bev_path = group_dir / (stem + "_bev.jpg")
            if camera_path.is_file() and not args.overwrite:
                skipped += 1
                print("[%d/%d] skip existing: %s" % (sequence + 1, len(indices), camera_path))
                continue
            with torch.no_grad():
                result = processor(model(image_tensor.unsqueeze(0).to(device)), [target])[0]
            original = cv2.imread(target["image_path"], cv2.IMREAD_COLOR)
            camera = draw_shared_camera_view(
                original.copy(), target, result, config["data"]["classes"],
                args.max_detections, args.draw_ground_truth,
            )
            group_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(camera_path), camera):
                raise OSError("Cannot write visualization: %s" % camera_path)
            if args.save_bev:
                bev = draw_shared_bev(
                    target,
                    result,
                    config["data"]["classes"],
                    args.max_detections,
                )
                if not cv2.imwrite(str(bev_path), bev):
                    raise OSError("Cannot write BEV visualization: %s" % bev_path)
            generated += 1
            summary.append(
                {
                    "dataset_index": index,
                    "sample_token": target["sample_token"],
                    "split_group": target["split_group"],
                    "image_path": target["image_path"],
                    "visualization": str(camera_path),
                    "bev_visualization": str(bev_path) if args.save_bev else None,
                    "predictions": result_to_json(
                        result, config["data"]["classes"], args.max_detections
                    ),
                }
            )
            print("[%d/%d] manifest#%d %s -> %s boxes=%d" % (
                sequence + 1, len(indices), index, Path(target["image_path"]).name,
                camera_path, min(args.max_detections, len(result["scores"])),
            ))
        except Exception as error:
            failed += 1
            errors.append({"dataset_index": index, "error": repr(error)})
            print("[%d/%d] ERROR manifest#%d: %s" % (sequence + 1, len(indices), index, error))
    if errors:
        error_path = output_dir / "errors.jsonl"
        error_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors),
            encoding="utf-8",
        )
        print("Errors: %s" % error_path.resolve())
    summary_path = output_dir / "predictions.json"
    if not args.skip_summary_json:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Visualizations: %s" % output_dir.resolve())
    if not args.skip_summary_json:
        print("Predictions: %s" % summary_path.resolve())
    print("Done: generated=%d, existing=%d, failed=%d" % (generated, skipped, failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
