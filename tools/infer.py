#!/usr/bin/env python3
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


def main():
    parser = argparse.ArgumentParser(description="Infer and draw FCOS3D detections")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True); parser.add_argument("--calib", required=True)
    parser.add_argument("--output", default="outputs/inference.jpg")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides); image = cv2.imread(args.image)
    if image is None: raise FileNotFoundError(args.image)
    original_h, original_w = image.shape[:2]; target_h, target_w = config["data"]["image_size"]
    scale = min(target_w/original_w, target_h/original_h); width, height = round(original_w*scale), round(original_h*scale)
    canvas = np.empty((target_h, target_w, 3), np.float32)
    canvas[...] = np.asarray(config["data"].get("pad_value", [0, 0, 0]), np.float32)
    canvas[:height, :width] = cv2.resize(image, (width, height))
    calibration = load_front_left_calibration(Path(args.calib), (original_h, original_w)); k = calibration["k"].copy(); k[0] *= scale; k[1] *= scale
    target = {
        "camera_matrix": torch.from_numpy(k.astype(np.float32)),
        "distortion": torch.from_numpy(calibration["dist"].astype(np.float32)),
        "camera_to_vehicle_rotation": torch.from_numpy(calibration["r_c2v"].astype(np.float32)),
        "camera_to_vehicle_translation": torch.from_numpy(calibration["t_c2v"].astype(np.float32)),
        "image_size": torch.tensor([target_h, target_w], dtype=torch.int64),
    }
    tensor = torch.from_numpy(canvas.transpose(2,0,1).copy()).float().unsqueeze(0)
    tensor = normalize_image(
        tensor,
        config["data"].get("image_mean", [128, 128, 128]),
        config["data"].get("image_std", [128, 128, 128]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device); load_checkpoint(args.checkpoint, model); model.eval()
    with torch.no_grad(): result = FCOS3DPostProcessor(model, config)(model(tensor.to(device)), [target])[0]
    for box, score, label in zip(result.get("boxes2d", []), result["scores"], result["labels"]):
        x1,y1,x2,y2 = (box.cpu().numpy()/scale).astype(int).tolist()
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, "%s %.2f" % (config["data"]["classes"][int(label)], float(score)), (x1,max(y1-5,10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), image); print(output)


if __name__ == "__main__": main()
