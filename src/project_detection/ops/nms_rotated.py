from __future__ import annotations

import math

import numpy as np
import torch


def _validate_inputs(boxes, scores, iou_threshold):
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        raise TypeError("boxes and scores must be torch tensors")
    if not boxes.is_cuda or not scores.is_cuda:
        raise RuntimeError("project nms_rotated requires CUDA tensors")
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("boxes must have shape [N, 5]")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape [N]")
    if boxes.device != scores.device:
        raise ValueError("boxes and scores must be on the same CUDA device")
    if not boxes.is_floating_point() or not scores.is_floating_point():
        raise TypeError("boxes and scores must use a floating-point dtype")
    if not math.isfinite(float(iou_threshold)) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in [0, 1]")


def _corners(boxes, clockwise):
    # Horizon's clockwise=True convention matches the x/z convention used by
    # this project. clockwise=False reverses the supplied angular direction.
    angles = boxes[:, 4] if clockwise else -boxes[:, 4]
    cosine, sine = angles.cos(), angles.sin()
    signs = boxes.new_tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    local = signs[None] * boxes[:, None, 2:4] * 0.5
    x = cosine[:, None] * local[..., 0] - sine[:, None] * local[..., 1]
    y = sine[:, None] * local[..., 0] + cosine[:, None] * local[..., 1]
    return torch.stack((x + boxes[:, None, 0], y + boxes[:, None, 1]), dim=-1)


def _cross_2d(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _points_inside_rectangles(points, rectangles):
    """Test R sets of points against C oriented rectangles."""
    delta = points[:, None] - rectangles[None, :, None, :2]
    cosine = rectangles[None, :, None, 4].cos()
    sine = rectangles[None, :, None, 4].sin()
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    epsilon = 1e-6
    return (
        (local_x.abs() <= rectangles[None, :, None, 2] * 0.5 + epsilon)
        & (local_y.abs() <= rectangles[None, :, None, 3] * 0.5 + epsilon)
    )


def _pairwise_rotated_iou(left, right, clockwise=True):
    """Exact pairwise IoU for two CUDA box groups in [cx, cy, w, h, angle]."""
    if not clockwise:
        left = left.clone()
        right = right.clone()
        left[:, 4].neg_()
        right[:, 4].neg_()
    left_corners = _corners(left, clockwise=True)
    right_corners = _corners(right, clockwise=True)
    rows, columns = left.shape[0], right.shape[0]

    left_inside = _points_inside_rectangles(left_corners, right)
    right_inside = _points_inside_rectangles(right_corners, left).permute(1, 0, 2)

    left_start = left_corners[:, None, :, None, :]
    left_vector = (
        left_corners.roll(-1, dims=1) - left_corners
    )[:, None, :, None, :]
    right_start = right_corners[None, :, None, :, :]
    right_vector = (
        right_corners.roll(-1, dims=1) - right_corners
    )[None, :, None, :, :]
    offset = right_start - left_start
    denominator = _cross_2d(left_vector, right_vector)
    nonparallel = denominator.abs() > 1e-8
    safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
    left_parameter = _cross_2d(offset, right_vector) / safe_denominator
    right_parameter = _cross_2d(offset, left_vector) / safe_denominator
    intersections_valid = (
        nonparallel
        & (left_parameter >= -1e-6)
        & (left_parameter <= 1.0 + 1e-6)
        & (right_parameter >= -1e-6)
        & (right_parameter <= 1.0 + 1e-6)
    )
    intersections = left_start + left_parameter[..., None] * left_vector
    intersections = intersections.reshape(rows, columns, 16, 2)
    intersections_valid = intersections_valid.reshape(rows, columns, 16)

    left_candidates = left_corners[:, None].expand(-1, columns, -1, -1)
    right_candidates = right_corners[None].expand(rows, -1, -1, -1)
    candidates = torch.cat(
        (left_candidates, right_candidates, intersections), dim=2
    )
    candidate_valid = torch.cat(
        (left_inside, right_inside, intersections_valid), dim=2
    )

    valid_count = candidate_valid.sum(dim=2, keepdim=True)
    centroid = (
        candidates * candidate_valid[..., None]
    ).sum(dim=2) / valid_count.clamp_min(1)
    relative = candidates - centroid[:, :, None]
    angles = torch.atan2(relative[..., 1], relative[..., 0])
    angles = angles.masked_fill(~candidate_valid, 4.0 * math.pi)
    order = angles.argsort(dim=2)
    ordered_points = candidates.gather(
        2, order[..., None].expand(-1, -1, -1, 2)
    )
    ordered_valid = candidate_valid.gather(2, order)
    first_point = ordered_points[:, :, :1]
    ordered_points = torch.where(
        ordered_valid[..., None], ordered_points, first_point
    )
    next_points = ordered_points.roll(-1, dims=2)
    intersection_area = 0.5 * _cross_2d(
        ordered_points, next_points
    ).sum(dim=2).abs()
    intersection_area = torch.where(
        valid_count.squeeze(2) >= 3,
        intersection_area,
        torch.zeros_like(intersection_area),
    )
    left_area = (left[:, 2] * left[:, 3]).abs()[:, None]
    right_area = (right[:, 2] * right[:, 3]).abs()[None]
    union = (left_area + right_area - intersection_area).clamp_min(1e-7)
    return (intersection_area / union).clamp(0.0, 1.0)


@torch.no_grad()
def nms_rotated(
    boxes,
    scores,
    iou_threshold,
    clockwise=True,
    pairwise_chunk_size=32,
):
    """CUDA rotated NMS with Horizon-compatible arguments.

    Args:
        boxes: CUDA tensor shaped [N, 5] in [cx, cy, width, height, angle].
        scores: CUDA tensor shaped [N].
        iou_threshold: suppress boxes whose rotated IoU exceeds this value.
        clockwise: angular convention flag, matching Horizon's default.
        pairwise_chunk_size: rows processed together to cap temporary memory.

    Returns:
        A ``(dets, keep)`` tuple. ``dets`` contains kept boxes plus scores in
        its last column, and ``keep`` indexes the original input.
    """
    _validate_inputs(boxes, scores, iou_threshold)
    count = boxes.shape[0]
    if count == 0:
        return torch.empty((0, 6), dtype=boxes.dtype, device=boxes.device), torch.empty(
            0, dtype=torch.long, device=boxes.device
        )
    chunk_size = max(int(pairwise_chunk_size), 1)
    order = scores.argsort(descending=True)
    # Float32 is considerably faster and more stable than fp16 for polygon
    # intersections. Keep the public return dtype unchanged.
    sorted_boxes = boxes[order].float().contiguous()
    suppression = np.zeros((count, count), dtype=np.bool_)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        iou = _pairwise_rotated_iou(
            sorted_boxes[start:end], sorted_boxes[start:], clockwise=clockwise
        )
        suppression[start:end, start:] = (
            iou > float(iou_threshold)
        ).cpu().numpy()

    removed = np.zeros(count, dtype=np.bool_)
    kept_sorted = []
    for index in range(count):
        if removed[index]:
            continue
        kept_sorted.append(index)
        removed[index + 1:] |= suppression[index, index + 1:]
    kept_sorted = torch.as_tensor(kept_sorted, dtype=torch.long, device=boxes.device)
    keep = order[kept_sorted]
    dets = torch.cat((boxes[keep], scores[keep, None].to(boxes.dtype)), dim=1)
    return dets, keep
