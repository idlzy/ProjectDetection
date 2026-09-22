from __future__ import annotations

import math

import cv2
import numpy as np

from .data.geometry import project_distorted, vehicle_box_corners


CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
GROUND_TRUTH_COLOR = (0, 0, 0)
GROUND_TRUTH_TEXT_COLOR = (255, 255, 255)


def color_for_class(class_id):
    hue = int((class_id * 47) % 180)
    pixel = np.uint8([[[hue, 220, 255]]])
    return tuple(
        int(value) for value in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    )


def draw_label(image, text, origin, color, text_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.48, 1
    (width, height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = origin
    y = max(y, height + baseline + 2)
    cv2.rectangle(
        image,
        (x, y - height - baseline - 2),
        (x + width + 4, y),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline - 1),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def camera_box_corners(
    box, camera_to_vehicle_rotation=None, camera_to_vehicle_translation=None
):
    """Build corners for a gravity-aligned camera-frame 3D box."""
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
        heading_projection = np.asarray(
            [
                [r_c2v[0, 0], r_c2v[1, 0]],
                [r_c2v[0, 2], r_c2v[1, 2]],
            ],
            dtype=np.float64,
        )
        heading_camera = np.asarray(
            [math.cos(yaw), math.sin(yaw)], dtype=np.float64
        )
        try:
            heading_v = np.linalg.solve(heading_projection, heading_camera)
        except np.linalg.LinAlgError:
            heading_v = np.linalg.lstsq(
                heading_projection, heading_camera, rcond=None
            )[0]
        yaw_v = math.atan2(float(heading_v[1]), float(heading_v[0]))
        corners_v = vehicle_box_corners(
            center_v,
            np.asarray([length, width, height], dtype=np.float64),
            yaw_v,
        )
        return (r_c2v.T @ (corners_v - t_c2v.reshape(1, 3)).T).T

    local = np.asarray(
        [
            [-length / 2, -height / 2, -width / 2],
            [length / 2, -height / 2, -width / 2],
            [length / 2, -height / 2, width / 2],
            [-length / 2, -height / 2, width / 2],
            [-length / 2, height / 2, -width / 2],
            [length / 2, height / 2, -width / 2],
            [length / 2, height / 2, width / 2],
            [-length / 2, height / 2, width / 2],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated_x = cosine * local[:, 0] - sine * local[:, 2]
    rotated_z = sine * local[:, 0] + cosine * local[:, 2]
    return np.stack(
        [rotated_x + x, local[:, 1] + y, rotated_z + z], axis=1
    )


def project_camera_box(box, target):
    corners = camera_box_corners(
        box,
        target["camera_to_vehicle_rotation"].cpu().numpy(),
        target["camera_to_vehicle_translation"].cpu().numpy(),
    )
    camera_matrix = target["camera_matrix"].cpu().numpy().astype(np.float64)
    distortion = target["distortion"].cpu().numpy().astype(np.float64)
    pixels, depth = project_distorted(corners, camera_matrix, distortion)
    return pixels / float(target.get("scale_factor", 1.0)), depth


def draw_projected_cuboid(image, pixels, depth, color, thickness=2):
    valid = (depth > 0.1) & np.isfinite(pixels).all(axis=1)
    safe_pixels = np.nan_to_num(
        pixels, nan=0.0, posinf=1e6, neginf=-1e6
    )
    rounded = np.clip(safe_pixels, -1e6, 1e6).round().astype(np.int32)
    for start, end in CUBOID_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(rounded[start]),
                tuple(rounded[end]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return rounded, valid


def draw_camera_view(
    image, target, result, classes, max_detections, draw_ground_truth=False
):
    if draw_ground_truth:
        for box, label in zip(
            target.get("boxes3d", []), target.get("labels", [])
        ):
            box = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
            pixels, depth = project_camera_box(box, target)
            rounded, valid = draw_projected_cuboid(
                image, pixels, depth, GROUND_TRUTH_COLOR, 3
            )
            if valid.any():
                x, y = rounded[valid].min(axis=0).tolist()
                draw_label(
                    image,
                    "GT " + classes[int(label)],
                    (max(x, 0), max(y, 0)),
                    GROUND_TRUTH_COLOR,
                    GROUND_TRUTH_TEXT_COLOR,
                )

    boxes = result["boxes3d"][:max_detections].cpu()
    scores = result["scores"][:max_detections].cpu()
    labels = result["labels"][:max_detections].cpu()
    depth_confidences = result.get("depth_confidence")
    valid = result.get("geometry_valid")
    weights = result.get("depth_fusion_weight")
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
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
        confidence_text = (
            " depth_conf=%.2f" % float(depth_confidences[index])
            if depth_confidences is not None else ""
        )
        text = "%s score=%.2f%s z=%.1fm%s" % (
            classes[int(label)],
            float(score),
            confidence_text,
            float(box[2]),
            suffix,
        )
        draw_label(image, text, (max(x, 0), max(y, 0)), color)
    return image


def bev_corners(box):
    x, z, length, width, yaw = [
        float(value) for value in box[[0, 2, 3, 5, 6]]
    ]
    local = np.asarray(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return local @ np.asarray(
        [[cosine, -sine], [sine, cosine]]
    ).T + [x, z]


def draw_bev(
    target,
    result,
    classes,
    max_detections,
    size=(700, 700),
    draw_ground_truth=True,
):
    height, width = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x_limit, z_limit = 40.0, 80.0

    def project(points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        pixels = np.empty_like(points)
        pixels[:, 0] = width / 2 + points[:, 0] / x_limit * (width / 2 - 30)
        pixels[:, 1] = height - 30 - points[:, 1] / z_limit * (height - 60)
        if not np.isfinite(pixels).all() or np.abs(pixels).max() > 1e6:
            return None
        return pixels.round().astype(np.int32)

    for distance in range(10, 81, 10):
        y = int(height - 30 - distance / z_limit * (height - 60))
        cv2.line(canvas, (25, y), (width - 25, y), (210, 210, 210), 1)
        cv2.putText(
            canvas,
            "%dm" % distance,
            (5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 80, 80),
            1,
        )
    cv2.line(
        canvas,
        (width // 2, 10),
        (width // 2, height - 15),
        (160, 160, 160),
        1,
    )
    cv2.circle(canvas, (width // 2, height - 30), 5, (0, 0, 0), -1)

    if draw_ground_truth:
        for box in target.get("boxes3d", []):
            box = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
            points = project(bev_corners(box))
            if points is None:
                continue
            cv2.polylines(
                canvas,
                [points],
                True,
                GROUND_TRUTH_COLOR,
                2,
            )
    depth_confidences = result.get("depth_confidence")
    for index, (box, score, label) in enumerate(zip(
        result["boxes3d"][:max_detections].cpu().numpy(),
        result["scores"][:max_detections].cpu().numpy(),
        result["labels"][:max_detections].cpu().numpy(),
    )):
        points = project(bev_corners(box))
        if points is None or not np.isfinite(score):
            continue
        color = color_for_class(int(label))
        cv2.polylines(canvas, [points], True, color, 2)
        cv2.putText(
            canvas,
            "score=%.2f%s" % (
                score,
                " dc=%.2f" % float(depth_confidences[index])
                if depth_confidences is not None else "",
            ),
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    legend = "prediction=class color"
    if draw_ground_truth and len(target.get("boxes3d", [])):
        legend = "GT=black, " + legend
    cv2.putText(
        canvas,
        legend,
        (20, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
    )
    return canvas
