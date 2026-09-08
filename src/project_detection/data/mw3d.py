from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import build_targets, load_front_left_calibration
from .preprocessing import normalize_image


class Mw3dReadyDataset(Dataset):
    def __init__(self, data_root: str, ready_root: Optional[str], split: str, classes: List[str],
                 image_size=(512, 896), max_samples: Optional[int] = None,
                 image_mean=(128.0, 128.0, 128.0), image_std=(128.0, 128.0, 128.0),
                 pad_value=(0.0, 0.0, 0.0)) -> None:
        self.data_root = Path(data_root)
        # Local datasets may keep frames.jsonl, splits/, images/ and annotation/
        # under one root. The separate ready_root remains supported for the
        # existing remote dataset layout.
        self.ready_root = Path(ready_root) if ready_root else self.data_root
        self.split = split
        self.classes = list(classes)
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}
        self.image_size = tuple(image_size)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.pad_value = tuple(pad_value)
        manifest = self.ready_root / "splits" / (split + "_frames.jsonl")
        if not manifest.is_file():
            raise FileNotFoundError("Missing split manifest: %s" % manifest)
        with manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.available_samples = len(self.records)
        required_fields = ("image_rel", "ann_rel", "calib_rel")
        for index, record in enumerate(self.records):
            missing = [key for key in required_fields if not record.get(key)]
            if missing:
                raise ValueError(
                    "Invalid record %d in %s; missing: %s"
                    % (index + 1, manifest, ", ".join(missing))
                )
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self._calibration_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_root / record["image_rel"]
        annotation_path = self.data_root / record["ann_rel"]
        calibration_path = self.data_root / record["calib_rel"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Cannot read image: %s" % image_path)
        original_h, original_w = image.shape[:2]
        cache_key = "%s:%dx%d" % (calibration_path, original_h, original_w)
        if cache_key not in self._calibration_cache:
            self._calibration_cache[cache_key] = load_front_left_calibration(calibration_path, (original_h, original_w))
        calibration = self._calibration_cache[cache_key]
        with annotation_path.open("r", encoding="utf-8") as handle:
            objects = json.load(handle)
        targets = build_targets(objects, calibration, (original_h, original_w), self.class_to_id)

        target_h, target_w = self.image_size
        scale = min(target_w / original_w, target_h / original_h)
        resized_w, resized_h = int(round(original_w * scale)), int(round(original_h * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.empty((target_h, target_w, 3), dtype=np.float32)
        canvas[...] = np.asarray(self.pad_value, dtype=np.float32)
        canvas[:resized_h, :resized_w] = resized
        targets["boxes2d"] *= scale
        targets["centers2d"] *= scale
        input_k = calibration["k"].copy()
        input_k[0] *= scale; input_k[1] *= scale
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float()
        tensor = normalize_image(tensor, self.image_mean, self.image_std)
        target = {key: torch.from_numpy(value) for key, value in targets.items()}
        target["camera_matrix"] = torch.from_numpy(input_k.astype(np.float32))
        target["distortion"] = torch.from_numpy(calibration["dist"].astype(np.float32))
        target["camera_to_vehicle_rotation"] = torch.from_numpy(
            calibration["r_c2v"].astype(np.float32)
        )
        target["camera_to_vehicle_translation"] = torch.from_numpy(
            calibration["t_c2v"].astype(np.float32)
        )
        target["image_size"] = torch.tensor([target_h, target_w], dtype=torch.int64)
        target["sample_token"] = record.get("sample_token", record.get("stem", str(index)))
        target["ann_batch"] = record.get("ann_batch", record.get("split_group", "unknown"))
        target["image_path"] = str(image_path)
        target["scale_factor"] = float(scale)
        return tensor, target


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
