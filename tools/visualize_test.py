#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.geometry import project_distorted
from project_detection.engine import build_loader, load_checkpoint
from project_detection.models import build_model
from project_detection.task import FCOS3DPostProcessor


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


def camera_box_corners(box):
    x, y, z, length, height, width, yaw = [
        float(value) for value in box[:7]
    ]
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
    corners = camera_box_corners(box)
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


def draw_camera_view(image, target, result, classes, max_detections):
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
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--max-detections", type=int, default=30)
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
    summary = []
    for sequence, index in enumerate(
        evenly_spaced_indices(len(dataset), min(args.max_images, len(dataset)))
    ):
        image_tensor, target = dataset[index]
        with torch.no_grad():
            result = processor(model(image_tensor.unsqueeze(0).to(device)), [target])[0]
        original = cv2.imread(target["image_path"], cv2.IMREAD_COLOR)
        camera = draw_camera_view(
            original.copy(), target, result, config["data"]["classes"], args.max_detections
        )
        bev = draw_bev(target, result, config["data"]["classes"], args.max_detections)
        stem = "%03d_%s" % (sequence, Path(target["image_path"]).stem)
        camera_path = output_dir / (stem + "_camera.jpg")
        bev_path = output_dir / (stem + "_bev.jpg")
        cv2.imwrite(str(camera_path), camera)
        cv2.imwrite(str(bev_path), bev)
        summary.append(
            {
                "dataset_index": index,
                "sample_token": target["sample_token"],
                "image_path": target["image_path"],
                "camera_visualization": str(camera_path),
                "bev_visualization": str(bev_path),
                "predictions": result_to_json(
                    result, config["data"]["classes"], args.max_detections
                ),
            }
        )
        print("[%d/%d] %s" % (sequence + 1, min(args.max_images, len(dataset)), stem))
    summary_path = output_dir / "predictions.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Visualizations: %s" % output_dir.resolve())
    print("Predictions: %s" % summary_path.resolve())


if __name__ == "__main__":
    main()
