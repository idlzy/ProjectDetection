#!/usr/bin/env python3
"""Standalone ONNX inference for the 0903 legacy HAT FCOS3D checkpoint.

Requires only numpy, opencv-python and onnxruntime. The exported ONNX embeds
its preprocessing, class, depth-fusion and postprocessing contract. This file
does not import project_detection, PyTorch or another local script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
FIELDS = ("cls", "bbox", "direction", "attribute", "centerness", "depth_logits")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def _sigmoid(value):
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float32)))


def _available_path(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s exists; pass --overwrite to replace it" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(source, recursive=False, max_images=None):
    source = Path(source)
    if source.is_file():
        images = [source]
    elif source.is_dir():
        candidates = source.rglob("*") if recursive else source.glob("*")
        images = sorted(path for path in candidates if path.is_file()
                        and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(source)
    if not images:
        raise FileNotFoundError("No supported images under %s" % source)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        images = images[:max_images]
    return images


def resolve_extrinsic(path, image, image_source):
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    image, image_source = Path(image), Path(image_source)
    relative = image.relative_to(image_source) if image_source.is_dir() else Path(image.name)
    for candidate in (path / relative.with_suffix(".txt"),
                      path / relative.parent / "extrinsic.txt",
                      path / (image.stem + ".txt")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No per-frame extrinsic for %s in %s" % (image, path))


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def load_calibration(path, image_hw, extrinsic_path=None):
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    param = next((value for value in raw.values() if isinstance(value, dict)
                  and value.get("camera_location") == "front_left"), None)
    if param is None:
        raise KeyError("front_left camera is missing in %s" % path)
    intr = param["intrinsics"]
    focal, principal = intr["focal_length_px"], intr["principal_point_px"]
    k = np.asarray(((float(focal["fx"]), 0, float(principal["cx"])),
                    (0, float(focal["fy"]), float(principal["cy"])),
                    (0, 0, 1)), dtype=np.float64)
    resolution = param.get("resolution_px", {})
    k[0] *= image_hw[1] / float(resolution.get("width", image_hw[1]))
    k[1] *= image_hw[0] / float(resolution.get("height", image_hw[0]))
    distortion = intr.get("distortion_coefficients", {})
    dist = np.asarray([distortion.get(key, 0) for key in (
        "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    )], dtype=np.float64)
    if extrinsic_path is None:
        ext = param["extrinsics"]
        r_c2v = (_rot_z(float(ext.get("yaw", 0))) @
                 _rot_y(float(ext.get("pitch", 0))) @
                 _rot_x(float(ext.get("roll", 0))))
        t_c2v = np.asarray([ext.get(axis, 0) for axis in "xyz"], dtype=np.float64)
    else:
        rows = []
        for number, line in enumerate(Path(extrinsic_path).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 3:
                raise ValueError("Expected 3 values in %s:%d" % (extrinsic_path, number))
            rows.append(values)
        if len(rows) != 4:
            raise ValueError("Expected 4 numeric rows in %s" % extrinsic_path)
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite extrinsic in %s" % extrinsic_path)
        r_c2v = values[:3].T
        t_c2v = -r_c2v @ values[3]
    return {"k": k, "dist": dist, "r_c2v": r_c2v, "t_c2v": t_c2v}


def prepare_sample(image_path, calibration_path, config, extrinsic_path=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)
    height, width = image.shape[:2]
    target_height, target_width = config["data"]["image_size"]
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    canvas = np.empty((target_height, target_width, 3), np.float32)
    canvas[...] = config["data"].get("pad_value", (0, 0, 0))
    canvas[:new_height, :new_width] = cv2.resize(image, (new_width, new_height))
    if config["data"].get("color_space", "bgr").lower() == "rgb":
        canvas = canvas[..., ::-1]
    mean = np.asarray(config["data"].get("image_mean", (128, 128, 128)), np.float32)
    std = np.asarray(config["data"].get("image_std", (128, 128, 128)), np.float32)
    if np.any(std == 0):
        raise ValueError("Image normalization std must be nonzero")
    tensor = np.ascontiguousarray(((canvas - mean) / std).transpose(2, 0, 1)[None],
                                  dtype=np.float32)
    target = load_calibration(calibration_path, (height, width), extrinsic_path)
    k = target["k"].copy()
    k[0] *= scale
    k[1] *= scale
    target["camera_matrix"] = k.astype(np.float32)
    target["scale_factor"] = scale
    return image, tensor, target


def load_contract(session, model_path, metadata_path=None):
    embedded = session.get_modelmeta().custom_metadata_map.get("project_detection_inference")
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    elif embedded:
        metadata = json.loads(embedded)
    else:
        sidecar = Path(model_path).with_suffix(".meta.json")
        if not sidecar.is_file():
            raise ValueError("ONNX has no embedded inference contract; provide --metadata")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "project_detection_raw_head_v1":
        raise ValueError("Unsupported ONNX export format")
    config = metadata.get("inference_config")
    if config is None:
        raise ValueError("Metadata lacks inference_config; re-export the 0903 model")
    if (config["model"].get("depth_mode") != "exp" or
            not config["model"].get("probabilistic_depth") or
            config["model"].get("geometric_depth")):
        raise ValueError("This script requires legacy exp+PGD without geometric depth")
    if metadata.get("depth_fuse_logit") is None:
        raise ValueError("Learned depth_fuse_logit is missing")
    expected = ["p%d_%s" % (level, field) for level in range(3, 8) for field in FIELDS]
    actual = [item.name for item in session.get_outputs()]
    if metadata.get("output_names") != expected or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("ONNX output names do not match the 0903 raw-head contract")
    shapes = {item.name: item.shape for item in session.get_outputs()}
    if any(isinstance(shapes["p%d_bbox" % level][1], int) and
           shapes["p%d_bbox" % level][1] != 9 for level in range(3, 8)):
        raise ValueError("0903 legacy bbox output must have 9 channels")
    return config, metadata


def _undistort(points, k, dist):
    xd = (points[:, 0] - k[0, 2]) / k[0, 0]
    yd = (points[:, 1] - k[1, 2]) / k[1, 1]
    x, y = xd.copy(), yd.copy()
    k1, k2, p1, p2, k3, k4, k5, k6 = dist
    for _ in range(6):
        r2 = x*x + y*y
        radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / np.maximum(
            1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2, 1e-6)
        x = (xd - (2*p1*x*y + p2*(r2 + 2*x*x))) / np.maximum(radial, 1e-6)
        y = (yd - (p1*(r2 + 2*y*y) + 2*p2*x*y)) / np.maximum(radial, 1e-6)
    return x, y


def _depth(raw, logits, config, metadata):
    centers = np.arange(0, config["model"]["depth_bin_max"] +
                        config["model"]["depth_bin_unit"],
                        config["model"]["depth_bin_unit"], dtype=np.float32)
    if logits.shape[1] != len(centers):
        raise ValueError("Depth-bin count differs from export metadata")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    weight = float(_sigmoid(metadata["depth_fuse_logit"]))
    depth = (weight * np.exp(raw[:, 2]) +
             (1 - weight) * (probabilities * centers).sum(axis=1))
    confidence = np.partition(probabilities, -2, axis=1)[:, -2:].mean(axis=1)
    return np.maximum(depth, 0.1), confidence


def _rotated_iou(first, second):
    a = ((float(first[0]), float(first[1])), (float(first[2]), float(first[3])),
         math.degrees(float(first[4])))
    b = ((float(second[0]), float(second[1])), (float(second[2]), float(second[3])),
         math.degrees(float(second[4])))
    intersection_type, points = cv2.rotatedRectangleIntersection(a, b)
    if intersection_type == cv2.INTERSECT_NONE or points is None:
        return 0.0
    intersection = abs(float(cv2.contourArea(points)))
    union = first[2]*first[3] + second[2]*second[3] - intersection
    return intersection / union if union > 1e-7 else 0.0


def rotated_bev_nms(boxes, scores, threshold):
    if not len(scores):
        return np.empty(0, np.int64)
    bev = boxes[:, [0, 2, 3, 5, 6]].astype(np.float32)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        rest = []
        for candidate in order[1:]:
            if _rotated_iou(bev[current], bev[int(candidate)]) <= threshold:
                rest.append(int(candidate))
        order = np.asarray(rest, np.int64)
    return np.asarray(keep, np.int64)


def _empty_result(level_counts):
    return {"boxes3d": np.empty((0, 9), np.float32),
            "boxes2d": np.empty((0, 4), np.float32),
            "scores": np.empty(0, np.float32), "labels": np.empty(0, np.int64),
            "depth_confidence": np.empty(0, np.float32),
            "source_levels": np.empty(0, np.int64),
            "level_candidate_counts": level_counts}


def postprocess(outputs, target, config, metadata):
    """Match the repository's FCOS3DPostProcessor._legacy_hat semantics."""
    evaluation = config["evaluation"]
    class_names = config["data"]["classes"]
    selected_names = evaluation.get("classes")
    selected_ids = list(range(len(class_names))) if selected_names is None else sorted(
        class_names.index(name) for name in selected_names)
    if not selected_ids:
        raise ValueError("No evaluation classes selected")
    all_boxes, all_image_boxes, all_scores, all_rank, all_levels = [], [], [], [], []
    all_confidences = []
    level_counts = []
    k, dist = target["camera_matrix"], target["dist"]
    for level_index, stride in enumerate(config["model"]["strides"]):
        level = level_index + 3
        cls_map = outputs["p%d_cls" % level]
        raw_map = outputs["p%d_bbox" % level]
        direction_map = outputs["p%d_direction" % level]
        center_map = outputs["p%d_centerness" % level]
        depth_map = outputs["p%d_depth_logits" % level]
        if raw_map.shape[1] != 9 or direction_map.shape[1] != 2:
            raise ValueError("P%d is not a 0903 legacy HAT output" % level)
        height, width = cls_map.shape[-2:]
        cls = _sigmoid(cls_map[0]).transpose(1, 2, 0).reshape(-1, len(class_names))
        centerness = _sigmoid(center_map[0]).reshape(-1)
        raw = raw_map[0].transpose(1, 2, 0).reshape(-1, 9)
        direction = direction_map[0].transpose(1, 2, 0).reshape(-1, 2).argmax(axis=1)
        logits = depth_map[0].transpose(1, 2, 0).reshape(-1, depth_map.shape[1])
        depth, confidence = _depth(raw, logits, config, metadata)
        location_rank = (cls[:, selected_ids] * centerness[:, None] *
                         confidence[:, None]).max(axis=1)
        nms_pre = int(evaluation["nms_pre"])
        if nms_pre > 0 and len(location_rank) > nms_pre:
            selected = np.argpartition(-location_rank, nms_pre - 1)[:nms_pre]
            selected = selected[np.argsort(-location_rank[selected], kind="stable")]
            cls, centerness, raw = cls[selected], centerness[selected], raw[selected]
            direction, depth, confidence = direction[selected], depth[selected], confidence[selected]
        else:
            selected = np.arange(len(raw))
        points = np.column_stack((((selected % width) + 0.5) * stride,
                                  ((selected // width) + 0.5) * stride))
        centers = points - raw[:, :2] * stride
        norm_x, norm_y = _undistort(centers, k, dist)
        dimensions = np.exp(raw[:, 3:6])
        local_yaw = np.remainder(raw[:, 6] - 0.7854, math.pi) + 0.7854 + math.pi*direction
        yaw = local_yaw + np.arctan(norm_x)
        boxes = np.column_stack((norm_x*depth, norm_y*depth, depth, dimensions,
                                 yaw, raw[:, 7:9]))
        half_w = k[0, 0] * dimensions[:, 2] / np.maximum(depth, 1e-3) / 2
        half_h = k[1, 1] * dimensions[:, 1] / np.maximum(depth, 1e-3) / 2
        image_boxes = np.column_stack((centers[:, 0]-half_w, centers[:, 1]-half_h,
                                       centers[:, 0]+half_w, centers[:, 1]+half_h))
        det_scores = cls * centerness[:, None]
        rank_scores = det_scores * confidence[:, None]
        level_counts.append(int((det_scores[:, selected_ids] >= evaluation["score_threshold"]).sum()))
        all_boxes.append(boxes)
        all_image_boxes.append(image_boxes)
        all_scores.append(det_scores)
        all_rank.append(rank_scores)
        all_confidences.append(confidence)
        all_levels.append(np.full(len(boxes), level_index, np.int64))

    boxes = np.concatenate(all_boxes)
    image_boxes = np.concatenate(all_image_boxes)
    scores = np.concatenate(all_scores)
    ranking = np.concatenate(all_rank)
    levels = np.concatenate(all_levels)
    confidences = np.concatenate(all_confidences)
    class_boxes, class_image_boxes, class_scores = [], [], []
    class_ranks, class_labels, class_levels = [], [], []
    class_confidences = []
    for label in selected_ids:
        valid = scores[:, label] >= evaluation["score_threshold"]
        if not valid.any():
            continue
        class_boxes.append(boxes[valid])
        class_image_boxes.append(image_boxes[valid])
        class_scores.append(scores[valid, label])
        class_ranks.append(ranking[valid, label])
        class_labels.append(np.full(int(valid.sum()), label, np.int64))
        class_levels.append(levels[valid])
        class_confidences.append(confidences[valid])
    if not class_boxes:
        return _empty_result(level_counts)
    boxes = np.concatenate(class_boxes)
    image_boxes = np.concatenate(class_image_boxes)
    scores = np.concatenate(class_scores)
    ranking = np.concatenate(class_ranks)
    labels = np.concatenate(class_labels)
    levels = np.concatenate(class_levels)
    confidences = np.concatenate(class_confidences)
    dimensions = boxes[:, 3:6]
    valid = (np.isfinite(boxes).all(axis=1) & np.isfinite(image_boxes).all(axis=1) &
             np.isfinite(ranking) & (boxes[:, 2] >= evaluation.get("min_valid_depth", 0.1)) &
             (boxes[:, 2] <= evaluation.get("max_valid_depth", 1000.0)) &
             (np.abs(boxes[:, :3]).max(axis=1) <= evaluation.get("max_abs_position", 1000.0)) &
             (dimensions >= evaluation.get("min_valid_dimension", 1e-3)).all(axis=1) &
             (dimensions <= evaluation.get("max_valid_dimension", 100.0)).all(axis=1) &
             (np.abs(boxes[:, 6]) <= evaluation.get("max_abs_yaw", 100.0*math.pi)))
    if not valid.any():
        return _empty_result(level_counts)
    boxes, image_boxes, scores = boxes[valid], image_boxes[valid], scores[valid]
    ranking, labels, levels = ranking[valid], labels[valid], levels[valid]
    confidences = confidences[valid]
    limit = int(evaluation.get("nms_max_candidates", 1000))
    threshold = float(evaluation["nms_threshold"])
    mode = evaluation.get("nms_mode", "global")
    if mode == "global":
        selected = np.argsort(-ranking, kind="stable")[:limit]
        kept = selected[rotated_bev_nms(boxes[selected], ranking[selected], threshold)]
    else:
        selected_groups = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            selected = indices[np.argsort(-ranking[indices], kind="stable")[:limit]]
            selected_groups.extend(selected[rotated_bev_nms(boxes[selected], ranking[selected], threshold)])
        kept = np.asarray(selected_groups, np.int64)
    kept = kept[np.argsort(-ranking[kept], kind="stable")[:int(evaluation["max_per_image"])]]
    return {"boxes3d": boxes[kept].astype(np.float32),
            "boxes2d": image_boxes[kept].astype(np.float32),
            "scores": scores[kept].astype(np.float32), "labels": labels[kept],
            "depth_confidence": confidences[kept].astype(np.float32),
            "source_levels": levels[kept], "level_candidate_counts": level_counts}


def _camera_corners(box, target):
    x, y, z, length, height, width, yaw = map(float, box[:7])
    rotation, translation = target["r_c2v"], target["t_c2v"]
    center_vehicle = rotation @ np.asarray((x, y, z)) + translation
    projection = np.asarray(((rotation[0, 0], rotation[1, 0]),
                             (rotation[0, 2], rotation[1, 2])))
    heading = np.asarray((math.cos(yaw), math.sin(yaw)))
    try:
        heading_vehicle = np.linalg.solve(projection, heading)
    except np.linalg.LinAlgError:
        heading_vehicle = np.linalg.lstsq(projection, heading, rcond=None)[0]
    yaw_vehicle = math.atan2(float(heading_vehicle[1]), float(heading_vehicle[0]))
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), np.float64)
    corners = signs * np.asarray((length, width, height)) / 2
    corners = corners @ _rot_z(yaw_vehicle).T + center_vehicle
    return (rotation.T @ (corners - translation).T).T


def _project(points, k, dist):
    depth = points[:, 2]
    x = points[:, 0] / np.clip(depth, 1e-6, None)
    y = points[:, 1] / np.clip(depth, 1e-6, None)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist
    r2 = x*x + y*y
    radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / (
        1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2)
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return np.column_stack((k[0, 0]*xd+k[0, 2], k[1, 1]*yd+k[1, 2])), depth


def _color(label):
    pixel = np.uint8([[[(int(label)*47) % 180, 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])



def draw_adaptive_labels(image, labels, bbox_regions=()):
    """Place compact labels against their own box, changing sides on collision.

    Each label is (lines, anchor, leader_color, text_color[, own_bbox]).
    Small boxes prefer exterior edges, large boxes prefer interior corners.
    Keep this helper self-contained for standalone deployment scripts.
    """
    if not labels:
        return []
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return []
    font = cv2.FONT_HERSHEY_SIMPLEX
    gap, padding = 2, 2
    occupied, panels = [], []
    reserved = np.asarray(bbox_regions, dtype=np.float64).reshape(-1, 4)
    reserved = reserved[np.isfinite(reserved).all(axis=1)]
    reserved[:, [0, 2]] = np.clip(reserved[:, [0, 2]], 0, width)
    reserved[:, [1, 3]] = np.clip(reserved[:, [1, 3]], 0, height)
    reserved = reserved[(reserved[:, 2] > reserved[:, 0]) &
                        (reserved[:, 3] > reserved[:, 1])]

    def intersection_areas(starts, ends, rectangles, margin=0):
        if not len(rectangles):
            return np.zeros(len(starts), dtype=np.float64)
        intersection = np.maximum(
            0, np.minimum(ends[:, None, :] + margin, rectangles[None, :, 2:]) -
            np.maximum(starts[:, None, :] - margin, rectangles[None, :, :2])
        )
        return (intersection[..., 0] * intersection[..., 1]).sum(axis=1)

    for label in labels:
        text, anchor, color, text_color = label[:4]
        raw_lines = text.split("\n") if isinstance(text, str) else list(text)
        ax = int(np.clip(anchor[0], padding, width - padding))
        ay = int(np.clip(anchor[1], padding, height - padding))
        own = None
        if len(label) > 4 and np.isfinite(label[4]).all():
            own = np.asarray(label[4], dtype=np.float64).copy()
            own[[0, 2]] = np.clip(own[[0, 2]], 0, width)
            own[[1, 3]] = np.clip(own[[1, 3]], 0, height)
        neighbors = reserved
        if own is not None and len(reserved):
            matches = np.flatnonzero(np.isclose(reserved, own).all(axis=1))
            if len(matches):
                neighbors = np.delete(reserved, matches[0], axis=0)
        best = None
        for scale in (0.32, 0.28, 0.26):
            max_text_width = min(width - 4 * padding, max(100, width // 5))
            lines = []
            for line in raw_lines:
                current = ""
                for char in str(line):
                    trial = current + char
                    if current and cv2.getTextSize(trial, font, scale, 1)[0][0] > max_text_width:
                        lines.append(current)
                        current = char
                    else:
                        current = trial
                lines.append(current)
            (_, glyph_height), baseline = cv2.getTextSize("Ag", font, scale, 1)
            line_height = glyph_height + baseline + 1
            pw = min(width - 2 * padding, max(
                cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines
            ) + 2 * padding)
            ph = len(lines) * line_height + 2 * padding
            if ph > height - 2 * padding:
                continue
            candidates, preferences = [], []
            if own is not None:
                bx, by, bx2, by2 = own
                bw, bh = bx2 - bx, by2 - by
                large = bw >= 1.5 * pw and bh >= 2.5 * ph
                # All exterior candidates touch a side; slide at most 12px.
                for shift in (0, -6, 6, -12, 12):
                    xs = (bx + shift, bx2 - pw + shift, (bx + bx2 - pw) / 2 + shift)
                    ys = (by + shift, by2 - ph + shift, (by + by2 - ph) / 2 + shift)
                    outside = [(x, by - ph - gap) for x in xs]
                    outside += [(x, by2 + gap) for x in xs]
                    outside += [(bx - pw - gap, y) for y in ys]
                    outside += [(bx2 + gap, y) for y in ys]
                    candidates.extend(outside)
                    preferences.extend([1 if large else 0] * len(outside))
                if bw >= pw + 2 * gap and bh >= ph + 2 * gap:
                    inside = [(x, y) for y in (by + gap, by2 - ph - gap)
                              for x in (bx + gap, bx2 - pw - gap)]
                    candidates.extend(inside)
                    preferences.extend([0 if large else 1] * len(inside))
            else:
                # Legacy callers without a box retain anchor-based placement.
                candidates = [(ax, ay - ph - gap), (ax, ay + gap)]
                candidates.extend((x, y) for y in range(padding, height - ph - padding + 1, ph + gap)
                                  for x in range(padding, width - pw - padding + 1, pw + gap))
                preferences = [0] * len(candidates)
            positions = np.asarray(candidates, dtype=np.int32)
            positions[:, 0] = np.clip(positions[:, 0], padding, width - padding - pw)
            positions[:, 1] = np.clip(positions[:, 1], padding, height - padding - ph)
            ends = positions + [pw, ph]
            label_overlap = intersection_areas(
                positions, ends, np.asarray(occupied).reshape(-1, 4), gap
            )
            box_overlap = intersection_areas(positions, ends, neighbors, 1)
            own_overlap = np.zeros(len(positions))
            if own is not None:
                own_overlap = intersection_areas(positions, ends, own.reshape(1, 4))
                inside = ((positions >= own[:2] + gap).all(axis=1) &
                          (ends <= own[2:] - gap).all(axis=1))
                own_overlap[inside] = 0
            nearest = np.maximum(positions, np.minimum(ends, [ax, ay]))
            distance = ((nearest - [ax, ay]).astype(np.float64) ** 2).sum(axis=1)
            crowding = np.zeros(len(positions))
            if own is not None and len(neighbors):
                center = (own[:2] + own[2:]) / 2
                offsets = (neighbors[:, :2] + neighbors[:, 2:]) / 2 - center
                nearby = ((np.abs(offsets[:, 0]) <= max(bw, pw) * 2) &
                          (np.abs(offsets[:, 1]) <= max(bh, ph) * 2))
                offsets = offsets[nearby]
                if len(offsets):
                    pressure = (offsets / np.maximum(
                        (offsets ** 2).sum(axis=1, keepdims=True), 1
                    )).sum(axis=0)
                    crowding = ((positions + ends) / 2 - center) @ pressure
            order = np.lexsort((distance, crowding, preferences, own_overlap,
                                box_overlap, label_overlap))
            index = int(order[0])
            key = (float(label_overlap[index]), float(box_overlap[index]),
                   float(own_overlap[index]), int(preferences[index]),
                   float(crowding[index]), float(distance[index]))
            choice = (key, positions[index], pw, ph, lines, scale, glyph_height, line_height)
            if best is None or key < best[0]:
                best = choice
            if key[:3] == (0.0, 0.0, 0.0) and key[3] == 0:
                best = choice
                break
        if best is None:
            continue
        _, position, pw, ph, lines, scale, glyph_height, line_height = best
        x, y = map(int, position)
        rect = (x, y, x + pw, y + ph)
        occupied.append(rect)
        panels.append((rect, lines, scale, glyph_height, line_height, own,
                       (ax, ay), color, text_color))
    # A short connector is only needed if clipping shifts an exterior label.
    for rect, _, _, _, _, own, anchor, color, _ in panels:
        endpoint = (int(np.clip(anchor[0], rect[0], rect[2])),
                    int(np.clip(anchor[1], rect[1], rect[3])))
        if own is not None:
            anchor = (int(np.clip(endpoint[0], own[0], own[2])),
                      int(np.clip(endpoint[1], own[1], own[3])))
        if (anchor[0] - endpoint[0]) ** 2 + (anchor[1] - endpoint[1]) ** 2 > 16:
            cv2.line(image, anchor, endpoint, color, 1, cv2.LINE_AA)
    for rect, lines, scale, glyph_height, line_height, _, _, color, text_color in panels:
        x, y, _, _ = rect
        for index, line in enumerate(lines):
            origin = (x + padding, y + padding + glyph_height + index * line_height)
            outline = (0, 0, 0) if sum(text_color) > 384 else (255, 255, 255)
            cv2.putText(image, line, origin, font, scale, outline, 2, cv2.LINE_AA)
            cv2.putText(image, line, origin, font, scale, text_color, 1, cv2.LINE_AA)
    return occupied


def draw_camera(image, result, target, classes):
    label_panels = []
    bbox_regions = []
    confidences = result.get("depth_confidence")
    for index, (box, score, label) in enumerate(zip(result["boxes3d"], result["scores"], result["labels"])):
        pixels, depth = _project(_camera_corners(box, target),
                                 target["camera_matrix"], target["dist"])
        pixels /= target["scale_factor"]
        valid = (depth > 0.1) & np.isfinite(pixels).all(axis=1)
        if not valid.any():
            continue
        points = np.clip(np.nan_to_num(pixels, nan=0, posinf=1e6, neginf=-1e6),
                         -1e6, 1e6).round().astype(np.int32)
        color = _color(label)
        for start, end in EDGES:
            if valid[start] and valid[end]:
                cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
        x, y = points[valid].min(axis=0)
        bbox_regions.append((x, y, *(points[valid].max(axis=0) + 1)))
        score_text = "score=%.2f" % score
        if confidences is not None:
            score_text += "/%.2f" % float(confidences[index])
        label_panels.append((
            [classes[int(label)], score_text, "z=%.1fm" % box[2]],
            (x, y), color, (255, 255, 255),
            bbox_regions[-1],
        ))
    draw_adaptive_labels(image, label_panels, bbox_regions)
    return image


def draw_bev(result, classes, size=(700, 700)):
    height, width = size
    canvas = np.full((height, width, 3), 245, np.uint8)
    for distance in range(10, 81, 10):
        y = int(height-30-distance/80*(height-60))
        cv2.line(canvas, (25, y), (width-25, y), (210, 210, 210), 1)
    cv2.line(canvas, (width//2, 10), (width//2, height-15), (160, 160, 160), 1)
    label_panels = []
    bbox_regions = []
    confidences = result.get("depth_confidence")
    for index, (box, score, label) in enumerate(zip(result["boxes3d"], result["scores"], result["labels"])):
        rectangle = ((float(box[0]), float(box[2])),
                     (float(box[3]), float(box[5])), math.degrees(float(box[6])))
        points = cv2.boxPoints(rectangle)
        points[:, 0] = width/2 + points[:, 0]/40*(width/2-30)
        points[:, 1] = height-30-points[:, 1]/80*(height-60)
        if not np.isfinite(points).all() or np.abs(points).max() > 1e6:
            continue
        points = points.round().astype(np.int32)
        color = _color(label)
        cv2.polylines(canvas, [points], True, color, 2)
        bbox_regions.append((*points.min(axis=0), *(points.max(axis=0) + 1)))
        score_text = "score=%.2f" % score
        if confidences is not None:
            score_text += "/%.2f" % float(confidences[index])
        label_panels.append((
            [classes[int(label)], score_text],
            tuple(points[0]), color, (255, 255, 255),
            bbox_regions[-1],
        ))
    draw_adaptive_labels(canvas, label_panels, bbox_regions)
    return canvas


def _image_paths(image, source, output):
    image, source = Path(image), Path(source)
    if source.is_file():
        path = Path(output or "outputs/onnx_0903_pred.jpg")
        return path, path.with_name(path.stem + "_bev" + path.suffix)
    root = Path(output or "outputs/onnx_0903")
    if root.is_file():
        raise ValueError("--output must be a directory for folder inference")
    relative = image.relative_to(source)
    path = root / relative.parent / (image.stem + "_pred.jpg")
    return path, path.with_name(image.stem + "_bev.jpg")


def _validate_input(session, tensor):
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)":
        raise ValueError("Expected one float32 ONNX input")
    for expected, actual in zip(inputs[0].shape, tensor.shape):
        if isinstance(expected, int) and expected != actual:
            raise ValueError("ONNX input shape %s differs from %s" % (
                inputs[0].shape, list(tensor.shape)))
    return inputs[0].name


def infer(args):
    import onnxruntime as ort
    if not Path(args.model).is_file() or not Path(args.calib_path).is_file():
        raise FileNotFoundError("Model and --calib-path must exist")
    available = ort.get_available_providers()
    providers = args.providers or (["CUDAExecutionProvider", "CPUExecutionProvider"]
                                   if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
    missing = [item for item in providers if item not in available]
    if missing:
        raise ValueError("Unavailable ONNX Runtime providers: %s" % missing)
    started = perf_counter()
    session = ort.InferenceSession(args.model, providers=providers)
    config, metadata = load_contract(session, args.model, args.metadata)
    for key, value in (("score_threshold", args.score_threshold),
                       ("nms_threshold", args.nms_threshold),
                       ("nms_pre", args.nms_pre), ("max_per_image", args.max_per_image)):
        if value is not None:
            config["evaluation"][key] = value
    evaluation = config["evaluation"]
    if not 0 <= evaluation["score_threshold"] <= 1 or not 0 <= evaluation["nms_threshold"] <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    if evaluation["nms_pre"] <= 0 or evaluation["max_per_image"] <= 0:
        raise ValueError("nms-pre and max-per-image must be positive")
    if args.save_bev and args.no_visualize:
        raise ValueError("--save-bev requires visualization")
    images = collect_images(args.image, args.recursive, args.max_images)
    names = [item.name for item in session.get_outputs()]
    print("Initialization %.2f ms | ORT=%s | CPU postprocess" % (
        (perf_counter()-started)*1000, session.get_providers()), flush=True)
    records = []
    for number, image in enumerate(images, 1):
        started = perf_counter()
        extrinsic = resolve_extrinsic(args.extri_path, image, args.image)
        original, tensor, target = prepare_sample(image, args.calib_path, config, extrinsic)
        input_name = _validate_input(session, tensor)
        preprocess_ms = (perf_counter()-started)*1000
        started = perf_counter()
        values = session.run(None, {input_name: tensor})
        inference_ms = (perf_counter()-started)*1000
        if any(not np.isfinite(value).all() for value in values):
            raise FloatingPointError("Non-finite ONNX output for %s" % image)
        started = perf_counter()
        result = postprocess(dict(zip(names, values)), target, config, metadata)
        postprocess_ms = (perf_counter()-started)*1000
        camera_path, bev_path, visualization_ms = None, None, None
        if not args.no_visualize:
            started = perf_counter()
            camera_path, bev_path = _image_paths(image, args.image, args.output)
            camera_path = _available_path(camera_path, args.overwrite)
            if not cv2.imwrite(str(camera_path), draw_camera(original.copy(), result, target,
                                                              config["data"]["classes"])):
                raise OSError("Cannot save %s" % camera_path)
            if args.save_bev:
                bev_path = _available_path(bev_path, args.overwrite)
                if not cv2.imwrite(str(bev_path), draw_bev(result, config["data"]["classes"])):
                    raise OSError("Cannot save %s" % bev_path)
            else:
                bev_path = None
            visualization_ms = (perf_counter()-started)*1000
        timing = {"preprocess_ms": preprocess_ms, "inference_ms": inference_ms,
                  "postprocess_ms": postprocess_ms, "visualization_ms": visualization_ms,
                  "total_ms": preprocess_ms+inference_ms+postprocess_ms+(visualization_ms or 0)}
        records.append({"image": str(image), "detections": len(result["scores"]),
                        "extrinsic": str(extrinsic) if extrinsic else None,
                        "visualization": str(camera_path) if camera_path else None,
                        "bev": str(bev_path) if bev_path else None, "timing": timing})
        print("[%d/%d] %s boxes=%d pre=%.2f ms onnx=%.2f ms post=%.2f ms viz=%s" % (
            number, len(images), image, len(result["scores"]), preprocess_ms,
            inference_ms, postprocess_ms,
            "%.2f ms" % visualization_ms if visualization_ms is not None else "off"),
            flush=True)
    average = {key: float(np.mean([record["timing"][key] for record in records]))
               for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")}
    if not args.no_visualize:
        average["visualization_ms"] = float(np.mean([
            record["timing"]["visualization_ms"] for record in records]))
    summary = {"model": str(Path(args.model).resolve()), "providers": session.get_providers(),
               "num_images": len(records), "average_timing": average, "images": records}
    if args.summary:
        _available_path(args.summary, args.overwrite).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Average over %d images | pre=%.2f ms onnx=%.2f ms post=%.2f ms "
          "viz=%s total=%.2f ms" % (
              len(records), average["preprocess_ms"], average["inference_ms"],
              average["postprocess_ms"],
              "%.2f ms" % average["visualization_ms"] if not args.no_visualize else "off",
              average["total_ms"],
          ))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", help="Optional export .meta.json if ONNX metadata was stripped")
    parser.add_argument("--image", required=True, help="Image or directory")
    parser.add_argument("--calib-path", "--calib", dest="calib_path", required=True)
    parser.add_argument("--extri-path", "--extrinsic", dest="extri_path")
    parser.add_argument("--output", help="Output JPG for one image, directory for a folder")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--save-bev", action="store_true")
    parser.add_argument("--summary", help="Optional timing JSON")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--nms-pre", type=int)
    parser.add_argument("--max-per-image", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    infer(build_parser().parse_args())
