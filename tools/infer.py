#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.geometry import load_front_left_calibration
from project_detection.data.preprocessing import normalize_image
from project_detection.engine import load_checkpoint
from project_detection.models import build_model, forward_with_targets
from project_detection.task import FCOS3DPostProcessor
from project_detection.visualization import draw_bev, draw_camera_view


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def collect_images(input_path: Path, recursive: bool):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    candidates = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        path for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError("No supported images found under %s" % input_path)
    return images


def bev_path_for(camera_path: Path):
    stem = camera_path.stem
    if stem.endswith("_pred"):
        stem = stem[:-5]
    return camera_path.with_name(stem + "_bev" + camera_path.suffix)


def head_report_path(camera_path: Path):
    return camera_path.with_name(camera_path.stem + "_heads.json")


def _tensor_value(result, key, index):
    value = result.get(key)
    if value is None:
        return None
    item = value[index].detach().cpu()
    if item.ndim == 0:
        if item.dtype == torch.bool:
            return bool(item.item())
        return float(item.item())
    return [float(number) for number in item.tolist()]


def build_head_report(result, class_names, strides, scale_factor=1.0):
    source_levels = result.get("source_levels")
    if source_levels is None:
        source_levels = torch.full_like(result["labels"], -1)
    source_levels = source_levels.detach().cpu()
    candidate_counts = result.get("level_candidate_counts", [])
    heads = []
    for level_index, stride in enumerate(strides):
        indices = torch.where(source_levels == level_index)[0].tolist()
        detections = []
        for index in indices:
            class_id = int(result["labels"][index].item())
            box3d = _tensor_value(result, "boxes3d", index)
            bbox2d = _tensor_value(result, "boxes2d", index)
            item = {
                "detection_index": int(index),
                "class_id": class_id,
                "class_name": class_names[class_id],
                "score": _tensor_value(result, "scores", index),
                "bbox2d_model_input_xyxy": bbox2d,
                "bbox2d_original_image_xyxy": [
                    value / float(scale_factor) for value in bbox2d
                ],
                "center_xyz": box3d[:3],
                "dimensions_lhw": box3d[3:6],
                "yaw": box3d[6],
                "velocity_xz": box3d[7:9],
            }
            for key in (
                "depth_confidence", "depth_local", "depth_geometric",
                "depth_fusion_weight", "geometry_valid",
            ):
                value = _tensor_value(result, key, index)
                if value is not None:
                    item[key] = value
            detections.append(item)
        heads.append({
            "level_index": level_index,
            "head_name": "P%d" % (level_index + 3),
            "stride": int(stride),
            "candidates_after_threshold": (
                int(candidate_counts[level_index])
                if level_index < len(candidate_counts) else None
            ),
            "detections_after_nms": len(detections),
            "detections": detections,
        })
    return {
        "candidate_stage": "per-level nms_pre and score-threshold filtering",
        "detection_stage": "cross-level class-wise rotated BEV NMS and max_per_image",
        "total_detections_after_nms": int(len(result["scores"])),
        "heads": heads,
    }


def print_head_report(report):
    print("Detection head details:")
    for head in report["heads"]:
        print(
            "  %s stride=%d candidates_after_threshold=%s detections_after_nms=%d"
            % (
                head["head_name"], head["stride"],
                head["candidates_after_threshold"],
                head["detections_after_nms"],
            )
        )
        for detection in head["detections"]:
            print("    " + json.dumps(detection, ensure_ascii=False))


def output_paths(image_path, input_path, output, multiple):
    if not multiple:
        camera_path = output or Path("outputs/inference.jpg")
        return camera_path, bev_path_for(camera_path)

    output_dir = output or Path("outputs/inference")
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            "--output must be a directory when --image is a directory: %s"
            % output_dir
        )
    relative_parent = image_path.parent.relative_to(input_path)
    result_dir = output_dir / relative_parent
    return (
        result_dir / (image_path.stem + "_pred.jpg"),
        result_dir / (image_path.stem + "_bev.jpg"),
    )


def prepare_image(image_path, calibration_path, config, extrinsic_path=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)
    original_h, original_w = image.shape[:2]
    target_h, target_w = config["data"]["image_size"]
    scale = min(target_w / original_w, target_h / original_h)
    width = int(round(original_w * scale))
    height = int(round(original_h * scale))
    canvas = np.empty((target_h, target_w, 3), dtype=np.float32)
    canvas[...] = np.asarray(
        config["data"].get("pad_value", [0, 0, 0]), dtype=np.float32
    )
    canvas[:height, :width] = cv2.resize(image, (width, height))

    calibration = load_front_left_calibration(
        calibration_path, (original_h, original_w), extrinsic_path=extrinsic_path
    )
    camera_matrix = calibration["k"].copy()
    camera_matrix[0] *= scale
    camera_matrix[1] *= scale
    target = {
        "camera_matrix": torch.from_numpy(camera_matrix.astype(np.float32)),
        "distortion": torch.from_numpy(calibration["dist"].astype(np.float32)),
        "camera_to_vehicle_rotation": torch.from_numpy(
            calibration["r_c2v"].astype(np.float32)
        ),
        "camera_to_vehicle_translation": torch.from_numpy(
            calibration["t_c2v"].astype(np.float32)
        ),
        "image_size": torch.tensor([target_h, target_w], dtype=torch.int64),
        "scale_factor": float(scale),
    }
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float()
    tensor = normalize_image(
        tensor,
        config["data"].get("image_mean", [128, 128, 128]),
        config["data"].get("image_std", [128, 128, 128]),
    )
    return image, tensor, target


def main():
    parser = argparse.ArgumentParser(
        description="Infer and draw FCOS3D detections"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--image", required=True,
        help="An image file or a directory containing images",
    )
    parser.add_argument(
        "--calib", required=True,
        help="Calibration file shared by all input images",
    )
    parser.add_argument(
        "--extrinsic",
        help="Optional HAT-format vehicle-to-camera extrinsic shared by all inputs",
    )
    parser.add_argument(
        "--output",
        help="Output image for one input, or output directory for a folder input",
    )
    parser.add_argument(
        "--save-bev", action="store_true",
        help="Also save a bird's-eye-view image for every input",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively find images when --image is a directory",
    )
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    input_path = Path(args.image)
    calibration_path = Path(args.calib)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    extrinsic_path = Path(args.extrinsic) if args.extrinsic else None
    if extrinsic_path is not None and not extrinsic_path.is_file():
        raise FileNotFoundError(extrinsic_path)
    images = collect_images(input_path, args.recursive)
    multiple = input_path.is_dir()
    output = Path(args.output) if args.output else None
    max_detections = args.max_detections or config["evaluation"][
        "max_per_image"
    ]

    requested_device = config["runtime"].get("device", "cuda")
    device = torch.device(
        "cuda"
        if requested_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    processor = FCOS3DPostProcessor(model, config)

    for sequence, image_path in enumerate(images, start=1):
        original, tensor, target = prepare_image(
            image_path, calibration_path, config, extrinsic_path
        )
        with torch.no_grad():
            outputs = forward_with_targets(
                model, tensor.unsqueeze(0).to(device), [target]
            )
            result = processor(outputs, [target])[0]

        camera_path, bev_path = output_paths(
            image_path, input_path, output, multiple
        )
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        camera = draw_camera_view(
            original.copy(), target, result, config["data"]["classes"],
            max_detections, draw_ground_truth=False,
        )
        if not cv2.imwrite(str(camera_path), camera):
            raise OSError("Cannot write visualization: %s" % camera_path)
        if args.save_bev:
            bev_path.parent.mkdir(parents=True, exist_ok=True)
            bev = draw_bev(
                target, result, config["data"]["classes"], max_detections,
                draw_ground_truth=False,
            )
            if not cv2.imwrite(str(bev_path), bev):
                raise OSError("Cannot write BEV visualization: %s" % bev_path)

        if not multiple:
            report = build_head_report(
                result, config["data"]["classes"], config["model"]["strides"],
                target["scale_factor"],
            )
            report.update({
                "image": str(image_path),
                "checkpoint": str(Path(args.checkpoint)),
                "score_threshold": config["evaluation"]["score_threshold"],
                "nms_pre": config["evaluation"]["nms_pre"],
                "nms_threshold": config["evaluation"]["nms_threshold"],
                "nms_mode": config["evaluation"].get("nms_mode", "global"),
                "max_per_image": config["evaluation"]["max_per_image"],
            })
            details_path = head_report_path(camera_path)
            details_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print_head_report(report)
            print("Detection head JSON: %s" % details_path)

        print(
            "[%d/%d] %s -> %s%s boxes=%d"
            % (
                sequence,
                len(images),
                image_path,
                camera_path,
                " + " + str(bev_path) if args.save_bev else "",
                min(max_detections, len(result["scores"])),
            )
        )


if __name__ == "__main__":
    main()
