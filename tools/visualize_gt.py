#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from project_detection.config import load_config
from project_detection.engine import build_loader
from project_detection.visualization import draw_bev, draw_camera_view


def empty_result():
    return {
        "boxes3d": torch.empty((0, 9), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
    }


def output_path_for(target, data_root, output_dir):
    image_path = Path(target["image_path"])
    try:
        relative = image_path.relative_to(data_root)
    except ValueError:
        relative = Path(image_path.name)
    if relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    return output_dir / relative.parent / (relative.stem + "_gt.jpg")


def main():
    parser = argparse.ArgumentParser(
        description="Draw ground-truth 3D boxes without running a model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", default="kitti_gt")
    parser.add_argument("--save-bev", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    loader = build_loader(
        config,
        args.split,
        world_size=1,
        max_samples=args.max_images,
    )
    dataset = loader.dataset
    data_root = Path(config["data"]["data_root"]).resolve()
    output_dir = Path(args.output_dir)
    classes = config["data"]["classes"]
    blank = empty_result()
    generated = skipped = total_boxes = 0

    for index in range(len(dataset)):
        _, target = dataset[index]
        camera_path = output_path_for(target, data_root, output_dir)
        bev_path = camera_path.with_name(
            camera_path.stem[:-3] + "_gt_bev.jpg"
            if camera_path.stem.endswith("_gt")
            else camera_path.stem + "_bev.jpg"
        )
        if camera_path.is_file() and not args.overwrite:
            skipped += 1
            print(
                "[%d/%d] skip existing: %s"
                % (index + 1, len(dataset), camera_path)
            )
            continue

        image = cv2.imread(target["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(target["image_path"])
        box_count = len(target["boxes3d"])
        rendered = draw_camera_view(
            image,
            target,
            blank,
            classes,
            max_detections=0,
            draw_ground_truth=True,
        )
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(camera_path), rendered):
            raise OSError("Cannot write GT visualization: %s" % camera_path)
        if args.save_bev:
            bev = draw_bev(
                target,
                blank,
                classes,
                max_detections=0,
                draw_ground_truth=True,
            )
            if not cv2.imwrite(str(bev_path), bev):
                raise OSError("Cannot write GT BEV visualization: %s" % bev_path)
        generated += 1
        total_boxes += box_count
        print(
            "[%d/%d] %s -> %s boxes=%d%s"
            % (
                index + 1,
                len(dataset),
                target["sample_token"],
                camera_path,
                box_count,
                " bev=" + str(bev_path) if args.save_bev else "",
            )
        )

    print(
        "done | split=%s frames=%d generated=%d skipped=%d boxes=%d output=%s"
        % (
            args.split,
            len(dataset),
            generated,
            skipped,
            total_boxes,
            output_dir.resolve(),
        )
    )


if __name__ == "__main__":
    main()
