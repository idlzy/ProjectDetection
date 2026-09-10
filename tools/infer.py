#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.geometry import load_front_left_calibration
from project_detection.data.preprocessing import normalize_image
from project_detection.engine import load_checkpoint
from project_detection.models import build_model
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


def prepare_image(image_path, calibration_path, config):
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
        calibration_path, (original_h, original_w)
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
            image_path, calibration_path, config
        )
        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0).to(device))
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
