#!/usr/bin/env python3
"""Standalone FCOS3D raw-head ONNX inference, CPU BEV NMS and visualization.

Deployment dependencies: numpy, opencv-python, onnxruntime. A newly exported
ONNX contains the inference contract; older exports also need --config and
their sibling .meta.json. No project_detection package or PyTorch is used.

Example: python infer_onnx.py --model best.onnx --image images/ \
    --calib-path FrontViewCalibParam.json --output rendered/ \
    --provider CPUExecutionProvider
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
BASE_OUTPUT_FIELDS = ("cls", "bbox", "direction", "attribute", "centerness")
CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _require_available_output(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass --overwrite to replace it" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(input_path, recursive=False, max_images=None):
    input_path = Path(input_path)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        candidates = input_path.rglob("*") if recursive else input_path.glob("*")
        images = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(input_path)
    if not images:
        raise FileNotFoundError("No supported images found under %s" % input_path)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        images = images[:max_images]
    return images


def resolve_extrinsic_path(extri_path, image_path, input_path):
    if extri_path is None:
        return None
    extri_path = Path(extri_path)
    if extri_path.is_file():
        return extri_path
    if not extri_path.is_dir():
        raise FileNotFoundError(extri_path)
    image_path, input_path = Path(image_path), Path(input_path)
    relative = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
    candidates = (
        extri_path / relative.with_suffix(".txt"),
        extri_path / relative.parent / "extrinsic.txt",
        extri_path / (image_path.stem + ".txt"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Missing per-frame extrinsic for %s; tried: %s" % (
        image_path, ", ".join(str(path) for path in candidates),
    ))


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def load_front_left_calibration(path, image_hw, extrinsic_path=None):
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
    d = intr.get("distortion_coefficients", {})
    dist = np.asarray([d.get(key, 0) for key in (
        "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    )], dtype=np.float64)
    if extrinsic_path is not None:
        rows = []
        for line_number, line in enumerate(Path(extrinsic_path).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = [float(value) for value in line.split()]
            if len(row) != 3:
                raise ValueError("Expected 3 values in %s:%d" % (extrinsic_path, line_number))
            rows.append(row)
        if len(rows) != 4:
            raise ValueError("Expected 4 numeric rows in %s" % extrinsic_path)
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite extrinsic: %s" % extrinsic_path)
        r_c2v = values[:3].T
        t_c2v = -r_c2v @ values[3]
    else:
        ext = param["extrinsics"]
        r_c2v = (_rotation_z(float(ext.get("yaw", 0))) @
                 _rotation_y(float(ext.get("pitch", 0))) @
                 _rotation_x(float(ext.get("roll", 0))))
        t_c2v = np.asarray([ext.get(key, 0) for key in "xyz"], dtype=np.float64)
    return {"k": k, "dist": dist, "r_c2v": r_c2v, "t_c2v": t_c2v}


def prepare_image(image_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)
    tensor, _ = _preprocess_image(image, config)
    return tensor


def _preprocess_image(image, config):
    original_height, original_width = image.shape[:2]
    target_height, target_width = config["data"]["image_size"]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width, resized_height = int(round(original_width * scale)), int(round(original_height * scale))
    canvas = np.empty((target_height, target_width, 3), dtype=np.float32)
    canvas[...] = np.asarray(config["data"].get("pad_value", [0, 0, 0]), dtype=np.float32)
    canvas[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
    if config["data"].get("color_space", "bgr").lower() == "rgb":
        canvas = canvas[..., ::-1]
    mean = np.asarray(config["data"].get("image_mean", [128, 128, 128]), dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config["data"].get("image_std", [128, 128, 128]), dtype=np.float32).reshape(1, 1, 3)
    if np.any(std == 0):
        raise ValueError("Image normalization std must be non-zero")
    tensor = ((canvas - mean) / std).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), scale


def prepare_sample(image_path, calibration_path, config, extrinsic_path=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)
    tensor, scale = _preprocess_image(image, config)
    calibration = load_front_left_calibration(calibration_path, image.shape[:2], extrinsic_path)
    k = calibration["k"].copy()
    k[0] *= scale
    k[1] *= scale
    target = dict(calibration, camera_matrix=k.astype(np.float32),
                  image_size=np.asarray(config["data"]["image_size"], dtype=np.float32),
                  scale_factor=scale)
    return image, tensor, target


def select_providers(available, requested=None):
    if requested:
        unavailable = [name for name in requested if name not in available]
        if unavailable:
            raise ValueError("Unavailable ONNX Runtime providers: %s; available: %s" % (
                ", ".join(unavailable), ", ".join(available),
            ))
        return list(requested)
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def validate_input(session, tensor):
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Expected one ONNX input, found %d" % len(inputs))
    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise TypeError("Expected float32 ONNX input, found %s" % model_input.type)
    expected = model_input.shape
    if len(expected) != tensor.ndim:
        raise ValueError("Input rank mismatch: model=%s image=%s" % (expected, list(tensor.shape)))
    for axis, (expected_size, actual_size) in enumerate(zip(expected, tensor.shape)):
        if isinstance(expected_size, int) and expected_size != actual_size:
            raise ValueError("Input shape mismatch at axis %d: model=%s image=%s" % (
                axis, expected, list(tensor.shape),
            ))
    return model_input.name


def summarize_outputs(names, values):
    if len(names) != len(values):
        raise ValueError("ONNX Runtime output names and values do not match")
    summary = []
    for name, value in zip(names, values):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise FloatingPointError("Non-finite values in ONNX output %s" % name)
        summary.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
                        "min": float(array.min()), "max": float(array.max()),
                        "mean": float(array.mean())})
    return summary


def output_names(config):
    fields = list(BASE_OUTPUT_FIELDS)
    if config["model"].get("probabilistic_depth", False):
        fields.append("depth_logits")
    if config["model"].get("geometric_depth", False):
        fields.append("geo_weight")
    return ["p%d_%s" % (level, field) for level in range(3, 8) for field in fields]


def reconstruct_outputs(names, values, config, device=None):
    """Validate names and return per-level NumPy maps (device is ignored)."""
    expected = output_names(config)
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError("ONNX output contract mismatch; missing=%s extra=%s" % (
            sorted(set(expected) - set(names)), sorted(set(names) - set(expected)),
        ))
    named = dict(zip(names, values))
    fields = list(BASE_OUTPUT_FIELDS)
    if config["model"].get("probabilistic_depth", False):
        fields.append("depth_logits")
    if config["model"].get("geometric_depth", False):
        fields.append("geo_weight")
    result = []
    for level in range(3, 8):
        maps = {field: np.asarray(named["p%d_%s" % (level, field)]) for field in fields}
        maps["depth_logits"] = maps.get("depth_logits")
        maps["geo_weight"] = maps.get("geo_weight")
        result.append(maps)
    return result


def _deep_merge(base, update):
    result = dict(base)
    for key, value in update.items():
        result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _load_yaml_config(path, overrides=()):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("--config needs PyYAML; embedded ONNX metadata does not") from error
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base = current.pop("_base_", None)
    config = _load_yaml_config(path.parent / base) if base else {}
    config = _deep_merge(config, current)
    for item in overrides:
        if "=" not in item:
            raise ValueError("--set expects dotted.path=value")
        key, value = item.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    return config


def load_inference_contract(session, model_path, config_path=None, overrides=(), metadata_path=None):
    embedded = session.get_modelmeta().custom_metadata_map.get("project_detection_inference")
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    elif embedded:
        metadata = json.loads(embedded)
    else:
        default_path = Path(model_path).with_suffix(".meta.json")
        metadata = json.loads(default_path.read_text(encoding="utf-8")) if default_path.is_file() else {}
    if metadata.get("format") != "project_detection_raw_head_v1":
        raise ValueError("Expected a project_detection_raw_head_v1 ONNX export")
    if config_path:
        config = _load_yaml_config(config_path, overrides)
    else:
        if overrides:
            raise ValueError("--set requires --config; use dedicated threshold options instead")
        config = metadata.get("inference_config")
        if config is None:
            raise ValueError("This older ONNX lacks embedded inference_config; provide --config and its .meta.json")
    if config["model"].get("geometric_depth", False):
        raise NotImplementedError("Standalone inference does not support geometric_depth=True")
    if config["model"].get("depth_mode", "exp") != "exp":
        raise NotImplementedError("Standalone inference currently supports depth_mode=exp only")
    if len(config["model"]["strides"]) != 5:
        raise ValueError("Five FPN strides are required")
    if metadata.get("attribute_prediction_mode") != config["model"].get("attribute_prediction_mode", "parallel"):
        raise ValueError("ONNX and config attribute_prediction_mode differ")
    if bool(metadata.get("probabilistic_depth")) != bool(config["model"].get("probabilistic_depth")):
        raise ValueError("ONNX and config probabilistic_depth differ")
    if config["model"].get("probabilistic_depth") and metadata.get("depth_fuse_logit") is None:
        raise ValueError("ONNX metadata lacks learned depth_fuse_logit")
    expected = metadata.get("output_names")
    actual = [item.name for item in session.get_outputs()]
    if expected != output_names(config) or set(actual) != set(expected):
        raise ValueError("ONNX output names disagree with inference contract")
    return config, metadata


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float32)
    return np.exp(-np.logaddexp(0.0, -value))


def _decode_global_yaw(raw_yaw, direction_class):
    """Resolve the pi-periodic yaw regression with the direction bin."""
    offset = 0.7854
    yaw = np.remainder(raw_yaw - offset, math.pi) + offset
    yaw = yaw + direction_class.astype(raw_yaw.dtype) * math.pi
    return np.remainder(yaw + math.pi, 2.0 * math.pi) - math.pi


def _undistort_points(points, k, coefficients, iterations=6):
    xd = (points[:, 0] - k[0, 2]) / k[0, 0]
    yd = (points[:, 1] - k[1, 2]) / k[1, 1]
    x, y = xd.copy(), yd.copy()
    if coefficients is None:
        return x, y
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    for _ in range(iterations):
        r2 = x*x + y*y
        denominator = np.maximum(1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2, 1e-6)
        radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / denominator
        delta_x = 2*p1*x*y + p2*(r2 + 2*x*x)
        delta_y = p1*(r2 + 2*y*y) + 2*p2*x*y
        x = (xd - delta_x) / np.maximum(radial, 1e-6)
        y = (yd - delta_y) / np.maximum(radial, 1e-6)
    return x, y


def _depth_candidates(raw, logits, config, metadata):
    direct = np.exp(raw[:, 2])
    if logits is None:
        return direct, np.ones(len(raw), dtype=np.float32)
    centers = np.arange(0, config["model"]["depth_bin_max"] + config["model"]["depth_bin_unit"],
                        config["model"]["depth_bin_unit"], dtype=np.float32)
    if logits.shape[1] != len(centers):
        raise ValueError("Depth-bin count does not match ONNX output")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    weight = float(_sigmoid(metadata["depth_fuse_logit"]))
    depth = weight * direct + (1 - weight) * (probability * centers).sum(axis=1)
    confidence = np.partition(probability, -2, axis=1)[:, -2:].mean(axis=1)
    return depth, confidence


def _empty_result():
    return {"boxes3d": np.empty((0, 9), np.float32),
            "scores": np.empty((0,), np.float32),
            "labels": np.empty((0,), np.int64),
            "boxes2d": np.empty((0, 4), np.float32),
            "depth_confidence": np.empty((0,), np.float32),
            "source_levels": np.empty((0,), np.int64),
            "level_candidate_counts": [0] * 5}


def _rotated_iou(box_a, box_b):
    """OpenCV intersection for BEV [x, z, length, width, yaw]."""
    a = ((float(box_a[0]), float(box_a[1])), (float(box_a[2]), float(box_a[3])),
         math.degrees(float(box_a[4])))
    b = ((float(box_b[0]), float(box_b[1])), (float(box_b[2]), float(box_b[3])),
         math.degrees(float(box_b[4])))
    intersection_type, points = cv2.rotatedRectangleIntersection(a, b)
    if intersection_type == cv2.INTERSECT_NONE or points is None:
        return 0.0
    intersection = abs(float(cv2.contourArea(points)))
    union = float(box_a[2] * box_a[3] + box_b[2] * box_b[3] - intersection)
    return intersection / union if union > 1e-7 else 0.0


def rotated_bev_nms(boxes3d, scores, threshold):
    """CPU greedy rotated NMS, preserving descending-score order."""
    if not len(scores):
        return np.empty((0,), np.int64)
    bev = np.asarray(boxes3d[:, [0, 2, 3, 5, 6]], dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not len(remaining):
            break
        # AABB rejection avoids expensive polygon clipping for distant boxes.
        corners_a = cv2.boxPoints(((float(bev[current, 0]), float(bev[current, 1])),
                                   (float(bev[current, 2]), float(bev[current, 3])),
                                   math.degrees(float(bev[current, 4]))))
        left_a, top_a = corners_a.min(axis=0)
        right_a, bottom_a = corners_a.max(axis=0)
        kept_rest = []
        for candidate in remaining:
            box = bev[int(candidate)]
            corners_b = cv2.boxPoints(((float(box[0]), float(box[1])),
                                       (float(box[2]), float(box[3])), math.degrees(float(box[4]))))
            if (right_a < corners_b[:, 0].min() or corners_b[:, 0].max() < left_a or
                    bottom_a < corners_b[:, 1].min() or corners_b[:, 1].max() < top_a or
                    _rotated_iou(bev[current], box) <= threshold):
                kept_rest.append(int(candidate))
        order = np.asarray(kept_rest, dtype=np.int64)
    return np.asarray(keep, dtype=np.int64)


def _nms(boxes, scores, labels, config):
    evaluation = config["evaluation"]
    limit = int(evaluation.get("nms_max_candidates", 1000))
    threshold = float(evaluation["nms_threshold"])
    mode = evaluation.get("nms_mode", "global")
    if mode == "global":
        selected = np.argsort(-scores, kind="stable")[:limit]
        return selected[rotated_bev_nms(boxes[selected], scores[selected], threshold)]
    if mode not in ("per_class", "classwise"):
        raise ValueError("Unsupported nms_mode: %s" % mode)
    kept = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        selected = indices[np.argsort(-scores[indices], kind="stable")[:limit]]
        kept.extend(selected[rotated_bev_nms(boxes[selected], scores[selected], threshold)])
    return np.asarray(kept, dtype=np.int64)


def postprocess(outputs, target, config, metadata):
    """NumPy counterpart of the current non-legacy FCOS3D postprocessor."""
    evaluation = config["evaluation"]
    classes = config["data"]["classes"]
    selected_names = evaluation.get("classes")
    selected_ids = None if selected_names is None else {classes.index(name) for name in selected_names}
    strides = config["model"]["strides"]
    all_raw, all_direction, all_depth, all_conf, all_centers = [], [], [], [], []
    all_scores, all_rank, all_labels, all_levels = [], [], [], []
    level_counts = []
    for level_index, (prediction, stride) in enumerate(zip(outputs, strides)):
        cls = _sigmoid(prediction["cls"][0])
        center = _sigmoid(prediction["centerness"][0])
        height, width = cls.shape[-2:]
        # Report/threshold HAT detection scores; use depth confidence only
        # to rank pre-NMS candidates and final detections.
        combined = cls * center
        all_level_raw = prediction["bbox"][0].transpose(1, 2, 0).reshape(-1, prediction["bbox"].shape[1])
        all_level_direction = prediction["direction"][0].transpose(1, 2, 0).reshape(
            -1, prediction["direction"].shape[1]
        )
        logits = prediction["depth_logits"]
        all_probability = None if logits is None else logits[0].transpose(1, 2, 0).reshape(-1, logits.shape[1])
        level_depth, level_confidence = _depth_candidates(all_level_raw, all_probability, config, metadata)
        ranking = combined * level_confidence.reshape(1, height, width)
        if selected_ids is not None:
            excluded = [index for index in range(len(classes)) if index not in selected_ids]
            combined[excluded] = -1.0
            ranking[excluded] = -1.0
        flattened = combined.reshape(-1)
        flattened_rank = ranking.reshape(-1)
        # Select distinct locations by their best class, retaining all class
        # hypotheses afterwards (same preselection as the native decoder).
        location_rank = ranking.reshape(len(classes), -1).max(axis=0)
        top_count = min(int(evaluation["nms_pre"]), height * width)
        if top_count:
            selected_locations = np.argpartition(-location_rank, top_count - 1)[:top_count]
            top = (np.arange(len(classes))[:, None] * (height * width)
                   + selected_locations[None, :]).reshape(-1)
            top = top[np.argsort(-flattened_rank[top], kind="stable")]
            top = top[flattened[top] >= evaluation["score_threshold"]]
        else:
            top = np.empty((0,), dtype=np.int64)
        level_counts.append(len(top))
        if not len(top):
            continue
        class_ids = top // (height * width)
        locations = top % (height * width)
        raw = all_level_raw[locations]
        direction_logits = all_level_direction[locations]
        local_depth, depth_confidence = level_depth[locations], level_confidence[locations]
        points = np.column_stack((((locations % width) + 0.5) * stride,
                                  ((locations // width) + 0.5) * stride))
        centers = points + raw[:, :2] * stride
        valid = (np.isfinite(raw).all(axis=1) & np.isfinite(direction_logits).all(axis=1) &
                 np.isfinite(local_depth) &
                 np.isfinite(depth_confidence) & np.isfinite(centers).all(axis=1) &
                 np.isfinite(flattened[top]) &
                 np.isfinite(flattened_rank[top]) &
                 (local_depth >= evaluation.get("min_valid_depth", 0.1)) &
                 (local_depth <= evaluation.get("max_valid_depth", 1000.0)) &
                 (raw[:, 3:6] >= math.log(evaluation.get("min_valid_dimension", 1e-3))).all(axis=1) &
                 (raw[:, 3:6] <= math.log(evaluation.get("max_valid_dimension", 100.0))).all(axis=1) &
                 (np.abs(raw[:, 6]) <= evaluation.get("max_abs_yaw", 100.0 * math.pi)))
        if not valid.any():
            continue
        all_raw.append(raw[valid])
        all_direction.append(direction_logits[valid].argmax(axis=1))
        all_depth.append(local_depth[valid])
        all_conf.append(depth_confidence[valid]); all_centers.append(centers[valid])
        all_scores.append(flattened[top][valid]); all_labels.append(class_ids[valid])
        all_rank.append(flattened_rank[top][valid])
        all_levels.append(np.full(valid.sum(), level_index, dtype=np.int64))
    result = _empty_result()
    result["level_candidate_counts"] = level_counts
    if not all_raw:
        return result
    raw = np.concatenate(all_raw)
    direction_class = np.concatenate(all_direction)
    z = np.concatenate(all_depth)
    centers = np.concatenate(all_centers)
    scores = np.concatenate(all_scores)
    rank_scores = np.concatenate(all_rank)
    labels = np.concatenate(all_labels).astype(np.int64)
    confidence = np.concatenate(all_conf)
    source_levels = np.concatenate(all_levels)
    k = target["camera_matrix"]
    norm_x, norm_y = _undistort_points(centers, k, target["dist"])
    dimensions = np.exp(raw[:, 3:6])
    yaw = _decode_global_yaw(raw[:, 6], direction_class)
    boxes = np.column_stack((norm_x * z, norm_y * z, z, dimensions,
                             yaw, np.zeros((len(raw), 2), dtype=np.float32)))
    half_w = k[0, 0] * dimensions[:, 2] / np.maximum(z, 1e-3) / 2
    half_h = k[1, 1] * dimensions[:, 1] / np.maximum(z, 1e-3) / 2
    image_boxes = np.column_stack((centers[:, 0] - half_w, centers[:, 1] - half_h,
                                   centers[:, 0] + half_w, centers[:, 1] + half_h))
    valid = (np.isfinite(boxes).all(axis=1) & np.isfinite(image_boxes).all(axis=1) &
             np.isfinite(scores) & np.isfinite(rank_scores) &
             (z >= evaluation.get("min_valid_depth", 0.1)) &
             (z <= evaluation.get("max_valid_depth", 1000.0)) &
             (np.abs(boxes[:, :3]).max(axis=1) <= evaluation.get("max_abs_position", 1000.0)) &
             (dimensions >= evaluation.get("min_valid_dimension", 1e-3)).all(axis=1) &
             (dimensions <= evaluation.get("max_valid_dimension", 100.0)).all(axis=1) &
             (np.abs(raw[:, 6]) <= evaluation.get("max_abs_yaw", 100.0 * math.pi)))
    if not valid.any():
        return result
    boxes, image_boxes, scores, labels = boxes[valid], image_boxes[valid], scores[valid], labels[valid]
    rank_scores = rank_scores[valid]
    confidence, source_levels = confidence[valid], source_levels[valid]
    kept = _nms(boxes, rank_scores, labels, config)
    kept = kept[np.argsort(-rank_scores[kept], kind="stable")[:int(evaluation["max_per_image"])]]
    result.update(boxes3d=boxes[kept].astype(np.float32),
                  boxes2d=image_boxes[kept].astype(np.float32),
                  scores=scores[kept].astype(np.float32), labels=labels[kept],
                  depth_confidence=confidence[kept].astype(np.float32),
                  source_levels=source_levels[kept])
    return result


def _project_distorted(points, k, dist):
    depth = points[:, 2]
    x = points[:, 0] / np.clip(depth, 1e-6, None)
    y = points[:, 1] / np.clip(depth, 1e-6, None)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist
    r2 = x*x + y*y
    radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / (1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2)
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return np.column_stack((k[0, 0]*xd + k[0, 2], k[1, 1]*yd + k[1, 2])), depth


def _camera_box_corners(box, target):
    x, y, z, length, height, width, yaw = [float(value) for value in box[:7]]
    r_c2v, t_c2v = target["r_c2v"], target["t_c2v"]
    center_v = r_c2v @ np.asarray((x, y, z)) + t_c2v
    heading_projection = np.asarray(((r_c2v[0, 0], r_c2v[1, 0]),
                                     (r_c2v[0, 2], r_c2v[1, 2])))
    heading_camera = np.asarray((math.cos(yaw), math.sin(yaw)))
    try:
        heading_v = np.linalg.solve(heading_projection, heading_camera)
    except np.linalg.LinAlgError:
        heading_v = np.linalg.lstsq(heading_projection, heading_camera, rcond=None)[0]
    yaw_v = math.atan2(float(heading_v[1]), float(heading_v[0]))
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float64)
    corners = signs * np.asarray((length, width, height)) / 2
    corners = corners @ _rotation_z(yaw_v).T + center_v
    return (r_c2v.T @ (corners - t_c2v).T).T


def _class_color(label):
    pixel = np.uint8([[[(int(label) * 47) % 180, 220, 255]]])
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


def draw_camera_view(image, target, result, classes, max_detections):
    label_panels = []
    bbox_regions = []
    for box, score, label, confidence in zip(
        result["boxes3d"][:max_detections], result["scores"][:max_detections],
        result["labels"][:max_detections], result["depth_confidence"][:max_detections],
    ):
        corners = _camera_box_corners(box, target)
        pixels, depth = _project_distorted(corners, target["camera_matrix"], target["dist"])
        pixels /= target["scale_factor"]
        valid = (depth > 0.1) & np.isfinite(pixels).all(axis=1)
        if not valid.any():
            continue
        rounded = np.clip(np.nan_to_num(pixels, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6).round().astype(np.int32)
        color = _class_color(label)
        for start, end in CUBOID_EDGES:
            if valid[start] and valid[end]:
                cv2.line(image, tuple(rounded[start]), tuple(rounded[end]), color, 2, cv2.LINE_AA)
        x, y = rounded[valid].min(axis=0)
        bbox_regions.append((x, y, *(rounded[valid].max(axis=0) + 1)))
        label_panels.append((
            [classes[int(label)], "score=%.2f/%.2f" % (float(score), float(confidence)),
             "z=%.1fm" % float(box[2])],
            (x, y), color, (255, 255, 255),
            bbox_regions[-1],
        ))
    draw_adaptive_labels(image, label_panels, bbox_regions)
    return image


def draw_bev(result, classes, max_detections, size=(700, 700)):
    height, width = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for distance in range(10, 81, 10):
        y = int(height - 30 - distance / 80 * (height - 60))
        cv2.line(canvas, (25, y), (width - 25, y), (210, 210, 210), 1)
        cv2.putText(canvas, "%dm" % distance, (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.line(canvas, (width // 2, 10), (width // 2, height - 15), (160, 160, 160), 1)
    cv2.circle(canvas, (width // 2, height - 30), 5, (0, 0, 0), -1)
    label_panels = []
    bbox_regions = []
    confidences = result.get("depth_confidence")
    for index, (box, score, label) in enumerate(zip(result["boxes3d"][:max_detections],
                                 result["scores"][:max_detections], result["labels"][:max_detections])):
        rect = ((float(box[0]), float(box[2])), (float(box[3]), float(box[5])),
                math.degrees(float(box[6])))
        points = cv2.boxPoints(rect)
        points[:, 0] = width / 2 + points[:, 0] / 40 * (width / 2 - 30)
        points[:, 1] = height - 30 - points[:, 1] / 80 * (height - 60)
        if not np.isfinite(points).all() or np.abs(points).max() > 1e6:
            continue
        pixels = points.round().astype(np.int32)
        color = _class_color(label)
        cv2.polylines(canvas, [pixels], True, color, 2)
        bbox_regions.append((*pixels.min(axis=0), *(pixels.max(axis=0) + 1)))
        score_text = "score=%.2f" % float(score)
        if confidences is not None:
            score_text += "/%.2f" % float(confidences[index])
        label_panels.append((
            [classes[int(label)], score_text],
            tuple(pixels[0]), color, (255, 255, 255),
            bbox_regions[-1],
        ))
    draw_adaptive_labels(canvas, label_panels, bbox_regions)
    return canvas


def visualization_paths(image_path, input_path, output, multiple):
    image_path, input_path = Path(image_path), Path(input_path)
    if not multiple:
        camera = Path(output or "outputs/onnx_inference.jpg")
        return camera, camera.with_name(camera.stem + "_bev" + camera.suffix)
    output_dir = Path(output or "outputs/onnx_inference")
    if output_dir.is_file():
        raise ValueError("--output must be a directory for folder input")
    relative = image_path.relative_to(input_path)
    camera = output_dir / relative.parent / (image_path.stem + "_pred.jpg")
    return camera, camera.with_name(image_path.stem + "_bev.jpg")


def raw_output_path(image_path, input_path, raw_output, multiple):
    if raw_output is None:
        return None
    if not multiple:
        return Path(raw_output)
    relative = Path(image_path).relative_to(input_path)
    return Path(raw_output) / relative.parent / (relative.stem + ".npz")


def infer_model(config, model_path, input_path, calibration_path, *, session,
                metadata, extri_path=None, output=None, visualize=True, save_bev=False,
                recursive=False, max_images=None, raw_output=None, summary_path=None,
                overwrite=False):
    model_path, input_path, calibration_path = map(Path, (model_path, input_path, calibration_path))
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if save_bev and not visualize:
        raise ValueError("--save-bev requires visualization")
    images = collect_images(input_path, recursive, max_images)
    names = [item.name for item in session.get_outputs()]
    multiple = input_path.is_dir()
    records = []
    for index, image_path in enumerate(images, 1):
        started = perf_counter()
        extrinsic = resolve_extrinsic_path(extri_path, image_path, input_path)
        original, tensor, target = prepare_sample(image_path, calibration_path, config, extrinsic)
        input_name = validate_input(session, tensor)
        preprocess_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        values = session.run(None, {input_name: tensor})
        inference_ms = (perf_counter() - started) * 1000
        output_statistics = summarize_outputs(names, values)

        started = perf_counter()
        predictions = reconstruct_outputs(names, values, config)
        result = postprocess(predictions, target, config, metadata)
        postprocess_ms = (perf_counter() - started) * 1000

        raw_path = raw_output_path(image_path, input_path, raw_output, multiple)
        if raw_path is not None:
            raw_path = _require_available_output(raw_path, overwrite)
            np.savez(raw_path, **dict(zip(names, values)))

        camera_path, bev_path, visualization_ms = None, None, None
        if visualize:
            started = perf_counter()
            camera_path, bev_path = visualization_paths(image_path, input_path, output, multiple)
            camera_path = _require_available_output(camera_path, overwrite)
            camera = draw_camera_view(original.copy(), target, result, config["data"]["classes"],
                                      config["evaluation"]["max_per_image"])
            if not cv2.imwrite(str(camera_path), camera):
                raise OSError("Cannot write visualization: %s" % camera_path)
            if save_bev:
                bev_path = _require_available_output(bev_path, overwrite)
                bev = draw_bev(result, config["data"]["classes"], config["evaluation"]["max_per_image"])
                if not cv2.imwrite(str(bev_path), bev):
                    raise OSError("Cannot write BEV visualization: %s" % bev_path)
            else:
                bev_path = None
            visualization_ms = (perf_counter() - started) * 1000
        total_ms = preprocess_ms + inference_ms + postprocess_ms + (visualization_ms or 0)
        record = {
            "image": str(image_path.resolve()), "calibration": str(calibration_path.resolve()),
            "extrinsic": str(extrinsic.resolve()) if extrinsic else None,
            "extrinsic_source": "per_frame_txt" if extrinsic else "calib_json",
            "detections": int(len(result["scores"])),
            "visualization": str(camera_path) if camera_path else None,
            "bev": str(bev_path) if bev_path else None,
            "raw_output": str(raw_path) if raw_path else None,
            "timing": {"preprocess_ms": preprocess_ms, "inference_ms": inference_ms,
                       "postprocess_ms": postprocess_ms, "visualization_ms": visualization_ms,
                       "total_ms": total_ms},
            "outputs": output_statistics,
        }
        records.append(record)
        print("[%d/%d] %s boxes=%d | preprocess=%.2f ms inference=%.2f ms "
              "postprocess=%.2f ms visualization=%s total=%.2f ms%s" % (
                  index, len(images), image_path, record["detections"], preprocess_ms,
                  inference_ms, postprocess_ms,
                  "%.2f ms" % visualization_ms if visualize else "off", total_ms,
                  " -> %s" % camera_path if camera_path else "",
              ), flush=True)
    timing_keys = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
    average = {key: float(np.mean([r["timing"][key] for r in records])) for key in timing_keys}
    if visualize:
        average["visualization_ms"] = float(np.mean([r["timing"]["visualization_ms"] for r in records]))
    summary = {"model": str(model_path.resolve()), "providers": session.get_providers(),
               "postprocess_state": "onnx_metadata", "num_images": len(records),
               "average_timing": average, "images": records}
    if summary_path:
        path = _require_available_output(summary_path, overwrite)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Average over %d images | preprocess=%.2f ms inference=%.2f ms "
          "postprocess=%.2f ms visualization=%s total=%.2f ms" % (
              len(records), average["preprocess_ms"], average["inference_ms"],
              average["postprocess_ms"],
              "%.2f ms" % average["visualization_ms"] if visualize else "off",
              average["total_ms"],
          ))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", help="Only needed for older ONNX without embedded inference config")
    parser.add_argument("--metadata", help="Explicit export .meta.json if quantization removed ONNX metadata")
    parser.add_argument("--image", required=True, help="One image or directory of images")
    parser.add_argument("--calib-path", "--calib", dest="calib_path", required=True)
    parser.add_argument("--extri-path", "--extrinsic", dest="extri_path")
    parser.add_argument("--output", help="Visualization image or output directory")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--save-bev", action="store_true")
    parser.add_argument("--raw-output")
    parser.add_argument("--summary")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--nms-pre", type=int)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--max-per-image", type=int)
    parser.add_argument("--nms-pairwise-chunk-size", type=int,
                        help="Deprecated CUDA option; ignored by CPU NMS")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main():
    args = build_parser().parse_args()
    import onnxruntime as ort
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    providers = select_providers(ort.get_available_providers(), args.providers)
    started = perf_counter()
    session = ort.InferenceSession(str(model_path), providers=providers)
    config, metadata = load_inference_contract(
        session, model_path, args.config, args.overrides, args.metadata,
    )
    for key, value in (("score_threshold", args.score_threshold), ("nms_pre", args.nms_pre),
                       ("nms_threshold", args.nms_threshold), ("max_per_image", args.max_per_image)):
        if value is not None:
            config["evaluation"][key] = value
    evaluation = config["evaluation"]
    if not 0 <= evaluation["score_threshold"] <= 1 or not 0 <= evaluation["nms_threshold"] <= 1:
        raise ValueError("Score and NMS thresholds must be in [0, 1]")
    if evaluation["nms_pre"] <= 0 or evaluation["max_per_image"] <= 0:
        raise ValueError("nms-pre and max-per-image must be positive")
    if evaluation.get("nms_max_candidates", 1000) <= 0:
        raise ValueError("nms_max_candidates must be positive")
    print("Initialization: %.2f ms | ORT providers=%s | CPU postprocess" % (
        (perf_counter() - started) * 1000, session.get_providers(),
    ))
    infer_model(config, model_path, args.image, args.calib_path, session=session,
                metadata=metadata, extri_path=args.extri_path, output=args.output,
                visualize=not args.no_visualize, save_bev=args.save_bev,
                recursive=args.recursive, max_images=args.max_images,
                raw_output=args.raw_output, summary_path=args.summary,
                overwrite=args.overwrite)


if __name__ == "__main__":
    main()
