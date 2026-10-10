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
    label_panels = []
    bbox_regions = []
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
                x2, y2 = rounded[valid].max(axis=0).tolist()
                bbox_regions.append((x, y, x2 + 1, y2 + 1))
                label_panels.append((
                    ["GT " + classes[int(label)]],
                    (max(x, 0), max(y, 0)),
                    GROUND_TRUTH_COLOR,
                    GROUND_TRUTH_TEXT_COLOR,
                    bbox_regions[-1],
                ))

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
        x2, y2 = rounded[projected_valid].max(axis=0).tolist()
        bbox_regions.append((x, y, x2 + 1, y2 + 1))
        score_text = "score=%.2f" % float(score)
        if depth_confidences is not None:
            score_text += "/%.2f" % float(depth_confidences[index])
        lines = [
            classes[int(label)],
            score_text,
            "z=%.1fm" % float(box[2]),
        ]
        if valid is not None and bool(valid[index]):
            lines.append("G w=%.2f" % float(weights[index]))
        label_panels.append((lines, (x, y), color, (255, 255, 255), bbox_regions[-1]))
    draw_adaptive_labels(image, label_panels, bbox_regions)
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

    bbox_regions = []
    if draw_ground_truth:
        for box in target.get("boxes3d", []):
            box = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
            points = project(bev_corners(box))
            if points is None:
                continue
            bbox_regions.append((*points.min(axis=0), *(points.max(axis=0) + 1)))
            cv2.polylines(
                canvas,
                [points],
                True,
                GROUND_TRUTH_COLOR,
                2,
            )
    depth_confidences = result.get("depth_confidence")
    label_panels = []
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
        bbox_regions.append((*points.min(axis=0), *(points.max(axis=0) + 1)))
        lines = [classes[int(label)], "score=%.2f" % score]
        if depth_confidences is not None:
            lines[1] += "/%.2f" % float(depth_confidences[index])
        label_panels.append((lines, tuple(points[0]), color, (255, 255, 255), bbox_regions[-1]))
    draw_adaptive_labels(canvas, label_panels, bbox_regions)
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
