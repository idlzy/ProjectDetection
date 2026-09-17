from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def load_vehicle_to_camera_extrinsic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read HAT's per-frame vehicle-to-camera extrinsic text format.

    The file contains three rotation rows followed by one translation row;
    comments and empty lines are ignored.  Dataset targets use camera-to-
    vehicle transforms internally, so callers invert this transform.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing per-frame extrinsic: %s" % path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = [float(value) for value in stripped.split()]
            except ValueError as error:
                raise ValueError(
                    "Invalid numeric value in %s:%d" % (path, line_number)
                ) from error
            if len(row) != 3:
                raise ValueError(
                    "Expected 3 values in %s:%d, got %d"
                    % (path, line_number, len(row))
                )
            rows.append(row)
    if len(rows) != 4:
        raise ValueError(
            "Expected 4 numeric rows in %s, got %d" % (path, len(rows))
        )
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value in per-frame extrinsic: %s" % path)
    return values[:3], values[3]


def load_front_left_calibration(
    path: Path,
    image_hw: Tuple[int, int],
    extrinsic_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    param = next(
        (value for value in raw.values() if isinstance(value, dict) and value.get("camera_location") == "front_left"),
        None,
    )
    if param is None:
        raise KeyError("front_left camera is missing in %s" % path)
    intr = param["intrinsics"]
    focal, principal = intr["focal_length_px"], intr["principal_point_px"]
    k = np.asarray(
        [[float(focal["fx"]), 0.0, float(principal["cx"])],
         [0.0, float(focal["fy"]), float(principal["cy"])],
         [0.0, 0.0, 1.0]], dtype=np.float64,
    )
    resolution = param.get("resolution_px", {})
    calib_h = float(resolution.get("height", image_hw[0]))
    calib_w = float(resolution.get("width", image_hw[1]))
    k[0] *= image_hw[1] / calib_w
    k[1] *= image_hw[0] / calib_h
    d = intr.get("distortion_coefficients", {})
    dist = np.asarray([d.get("k1", 0), d.get("k2", 0), d.get("p1", 0), d.get("p2", 0),
                       d.get("k3", 0), d.get("k4", 0), d.get("k5", 0), d.get("k6", 0)], dtype=np.float64)
    if extrinsic_path is not None:
        # Match HAT: a declared per-frame file stores vehicle -> camera and
        # takes precedence over the batch-level transform embedded in JSON.
        r_v2c, t_v2c = load_vehicle_to_camera_extrinsic(extrinsic_path)
        r_c2v = r_v2c.T
        t_c2v = -r_c2v @ t_v2c
    else:
        ext = param["extrinsics"]
        r_c2v = rot_z(float(ext.get("yaw", 0))) @ rot_y(float(ext.get("pitch", 0))) @ rot_x(float(ext.get("roll", 0)))
        t_c2v = np.asarray([ext.get("x", 0), ext.get("y", 0), ext.get("z", 0)], dtype=np.float64)
    return {"k": k, "dist": dist, "r_c2v": r_c2v, "t_c2v": t_c2v}


def project_distorted(points_cam: np.ndarray, k: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth = points_cam[:, 2]
    x = points_cam[:, 0] / np.clip(depth, 1e-6, None)
    y = points_cam[:, 1] / np.clip(depth, 1e-6, None)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist.tolist()
    r2 = x * x + y * y
    radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / (1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2)
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return np.stack([k[0, 0]*xd + k[0, 2], k[1, 1]*yd + k[1, 2]], axis=1), depth


def vehicle_box_corners(center: np.ndarray, size_lwh: np.ndarray, yaw: float) -> np.ndarray:
    length, width, height = size_lwh.tolist()
    signs = np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float64)
    corners = signs * np.asarray([length, width, height], dtype=np.float64) / 2
    return corners @ rot_z(yaw).T + center.reshape(1, 3)


def camera_yaw(yaw_vehicle: float, r_c2v: np.ndarray) -> float:
    heading = r_c2v.T @ np.asarray([math.cos(yaw_vehicle), math.sin(yaw_vehicle), 0.0])
    return math.atan2(-float(heading[2]), float(heading[0]))


def build_targets(objects, calibration: Dict[str, np.ndarray], image_hw: Tuple[int, int], class_to_id: Dict[str, int]):
    h, w = image_hw
    k, dist, r_c2v, t_c2v = (calibration[key] for key in ("k", "dist", "r_c2v", "t_c2v"))
    boxes2d, centers2d, boxes3d, labels = [], [], [], []
    for obj in objects:
        try:
            visibility = float(obj.get("visibility", 0))
        except (TypeError, ValueError):
            visibility = 0.0
        if not math.isfinite(visibility) or visibility <= 0:
            continue
        source_flag = str(obj.get("source", {}).get("pinhole_left", 0)).lower()
        if source_flag not in ("1", "true", "yes") or obj.get("label") not in class_to_id:
            continue
        center_v = np.asarray([obj.get("center", {}).get(axis, 0) for axis in "xyz"], dtype=np.float64)
        center_c = r_c2v.T @ (center_v - t_c2v)
        if center_c[2] <= 0.1:
            continue
        size = np.asarray([max(float(obj.get("size", {}).get(axis, 0)), 1e-3) for axis in "xyz"])
        yaw_v = float(obj.get("rotation", {}).get("z", 0))
        corners_v = vehicle_box_corners(center_v, size, yaw_v)
        corners_c = (r_c2v.T @ (corners_v - t_c2v.reshape(1, 3)).T).T
        uv, depth = project_distorted(corners_c, k, dist)
        valid = (depth > 0.1) & np.isfinite(uv).all(axis=1)
        center_uv, _ = project_distorted(center_c.reshape(1, 3), k, dist)
        if valid.sum() < 2 or not np.isfinite(center_uv).all():
            continue
        visible = uv[valid]
        box = np.asarray([visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()])
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        yaw_c = camera_yaw(yaw_v, r_c2v)
        boxes2d.append(box); centers2d.append(center_uv[0])
        boxes3d.append([center_c[0], center_c[1], center_c[2], size[0], size[2], size[1], -yaw_c, 0, 0])
        labels.append(class_to_id[obj["label"]])
    return {
        "boxes2d": np.asarray(boxes2d, dtype=np.float32).reshape(-1, 4),
        "centers2d": np.asarray(centers2d, dtype=np.float32).reshape(-1, 2),
        "boxes3d": np.asarray(boxes3d, dtype=np.float32).reshape(-1, 9),
        "labels": np.asarray(labels, dtype=np.int64),
    }
