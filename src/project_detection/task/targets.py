from __future__ import annotations

import math

import torch
from torch.nn.utils.rnn import pad_sequence


def feature_points(height, width, stride, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
    )
    return torch.stack([(x.reshape(-1) + 0.5) * stride, (y.reshape(-1) + 0.5) * stride], dim=1)


def _target_assignment_geometry(points, targets, regress_range, stride, center_radius):
    """Build the point-to-GT validity matrix shared by training and diagnostics."""
    device = points.device
    boxes = targets["boxes2d"].to(device)
    centers = targets["centers2d"].to(device)
    if boxes.numel() == 0:
        distances = points.new_zeros((points.shape[0], 0, 4))
        valid = torch.zeros(
            (points.shape[0], 0), dtype=torch.bool, device=device
        )
        return boxes, centers, distances, valid

    px, py = points[:, 0:1], points[:, 1:2]
    left = px - boxes[:, 0]
    top = py - boxes[:, 1]
    right = boxes[:, 2] - px
    bottom = boxes[:, 3] - py
    distances = torch.stack([left, top, right, bottom], dim=2)
    max_distance = distances.max(dim=2).values
    in_range = (max_distance >= regress_range[0]) & (
        max_distance <= regress_range[1]
    )
    radius = stride * center_radius
    center_ok = (px - centers[:, 0]).abs() < radius
    center_ok &= (py - centers[:, 1]).abs() < radius
    valid = in_range & center_ok
    return boxes, centers, distances, valid


def _match_target_candidates(points, centers, valid):
    """Resolve candidate conflicts by distance to the projected 3D center."""
    count = valid.shape[0]
    matched_indices = torch.full(
        (count,), -1, dtype=torch.long, device=valid.device
    )
    if centers.numel() == 0:
        return matched_indices

    distances = torch.sqrt(((points[:, None, :] - centers[None]) ** 2).sum(dim=2))
    candidate_distances = distances.masked_fill(~valid, float("inf"))
    min_distance, matched = candidate_distances.min(dim=1)
    positive = torch.isfinite(min_distance)
    matched_indices[positive] = matched[positive]
    return matched_indices


def _center_distance_centerness(points, centers, stride, center_radius):
    """HAT FCOS3D centerness: exponential decay inside the center region."""
    distance = torch.sqrt(((points - centers) ** 2).sum(dim=-1))
    relative_distance = distance / (1.414 * center_radius * float(stride))
    return torch.exp(-2.5 * relative_distance)


def target_assignment_diagnostics(
    points, targets, regress_range, stride, center_radius=1.5
):
    """Return candidates before conflict resolution and final GT matches.

    The candidate matrix has shape ``[num_points, num_gt]``. A true entry
    means that the point satisfies the regression-range and center
    sampling constraints for that GT. ``matched_indices`` applies the same
    nearest-center conflict resolution used by :func:`assign_targets`.
    """
    _, centers, _, valid = _target_assignment_geometry(
        points, targets, regress_range, stride, center_radius
    )
    matched_indices = _match_target_candidates(points, centers, valid)
    return valid, matched_indices


def assign_targets(points, targets, regress_range, stride, center_radius=1.5):
    count = points.shape[0]; device = points.device
    labels = torch.full((count,), -1, dtype=torch.long, device=device)
    regression = torch.zeros((count, 9), device=device)
    centerness = torch.zeros((count,), device=device)
    direction = torch.zeros((count,), dtype=torch.long, device=device)
    boxes, centers, _, valid = _target_assignment_geometry(
        points, targets, regress_range, stride, center_radius
    )
    matched_indices = _match_target_candidates(points, centers, valid)
    if boxes.numel() == 0:
        return labels, regression, centerness, direction, matched_indices
    boxes3d = targets["boxes3d"].to(device)
    gt_labels = targets["labels"].to(device)
    positive = matched_indices >= 0
    if not positive.any():
        return labels, regression, centerness, direction, matched_indices
    match = matched_indices[positive]
    labels[positive] = gt_labels[match]
    matched_indices[positive] = match
    regression[positive, :2] = (centers[match] - points[positive]) / float(stride)
    regression[positive, 2:] = boxes3d[match, 2:]
    centerness[positive] = _center_distance_centerness(
        points[positive], centers[match], stride, center_radius
    )
    yaw = regression[positive, 6]
    direction[positive] = ((yaw - 0.7854) % (2 * math.pi) >= math.pi).long()
    return labels, regression, centerness, direction, matched_indices


def prepare_target_batch(targets, device):
    """Copy and pad variable-length image targets once for all FPN levels."""
    lengths = torch.as_tensor(
        [target["labels"].shape[0] for target in targets],
        dtype=torch.long,
        device=device,
    )
    boxes2d = pad_sequence(
        [target["boxes2d"].to(device) for target in targets],
        batch_first=True,
    )
    centers2d = pad_sequence(
        [target["centers2d"].to(device) for target in targets],
        batch_first=True,
    )
    boxes3d = pad_sequence(
        [target["boxes3d"].to(device) for target in targets],
        batch_first=True,
    )
    labels = pad_sequence(
        [target["labels"].to(device) for target in targets],
        batch_first=True,
        padding_value=-1,
    )
    return {
        "lengths": lengths,
        "boxes2d": boxes2d,
        "centers2d": centers2d,
        "boxes3d": boxes3d,
        "labels": labels,
    }


def assign_targets_batch(
    points,
    targets,
    regress_range,
    stride,
    center_radius=1.5,
    image_chunk_size=8,
):
    """Assign one FPN level for a padded batch in bounded image chunks."""
    batch = targets["lengths"].shape[0]
    count = points.shape[0]
    device = points.device
    all_labels = []
    all_regression = []
    all_centerness = []
    all_direction = []
    all_matched_indices = []
    chunk_size = max(int(image_chunk_size), 1)

    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        chunk_batch = end - start
        labels = torch.full(
            (chunk_batch, count), -1, dtype=torch.long, device=device
        )
        regression = torch.zeros((chunk_batch, count, 9), device=device)
        centerness = torch.zeros((chunk_batch, count), device=device)
        direction = torch.zeros(
            (chunk_batch, count), dtype=torch.long, device=device
        )
        matched_indices = torch.full(
            (chunk_batch, count), -1, dtype=torch.long, device=device
        )

        boxes = targets["boxes2d"][start:end]
        object_count = boxes.shape[1]
        if object_count:
            centers = targets["centers2d"][start:end]
            boxes3d = targets["boxes3d"][start:end]
            gt_labels = targets["labels"][start:end]
            valid_objects = (
                torch.arange(object_count, device=device)[None]
                < targets["lengths"][start:end, None]
            )
            px = points[None, :, None, 0]
            py = points[None, :, None, 1]
            left = px - boxes[:, None, :, 0]
            top = py - boxes[:, None, :, 1]
            right = boxes[:, None, :, 2] - px
            bottom = boxes[:, None, :, 3] - py
            distances = torch.stack([left, top, right, bottom], dim=3)
            max_distance = distances.max(dim=3).values
            in_range = (max_distance >= regress_range[0]) & (
                max_distance <= regress_range[1]
            )
            radius = stride * center_radius
            center_ok = (px - centers[:, None, :, 0]).abs() < radius
            center_ok &= (py - centers[:, None, :, 1]).abs() < radius
            valid = in_range & center_ok
            valid &= valid_objects[:, None]
            center_distances = torch.sqrt(
                (px - centers[:, None, :, 0]) ** 2
                + (py - centers[:, None, :, 1]) ** 2
            )
            candidate_distances = center_distances.masked_fill(~valid, float("inf"))
            min_distance, matched = candidate_distances.min(dim=2)
            positive_batch, positive_point = torch.where(torch.isfinite(min_distance))
            if positive_batch.numel():
                matched_object = matched[positive_batch, positive_point]
                labels[positive_batch, positive_point] = gt_labels[
                    positive_batch, matched_object
                ]
                matched_indices[positive_batch, positive_point] = matched_object
                regression[positive_batch, positive_point, :2] = (
                    centers[positive_batch, matched_object]
                    - points[positive_point]
                ) / float(stride)
                regression[positive_batch, positive_point, 2:] = boxes3d[
                    positive_batch, matched_object, 2:
                ]
                centerness[positive_batch, positive_point] = (
                    _center_distance_centerness(
                        points[positive_point],
                        centers[positive_batch, matched_object],
                        stride,
                        center_radius,
                    )
                )
                yaw = regression[positive_batch, positive_point, 6]
                direction[positive_batch, positive_point] = (
                    (yaw - 0.7854) % (2 * math.pi) >= math.pi
                ).long()

        all_labels.append(labels)
        all_regression.append(regression)
        all_centerness.append(centerness)
        all_direction.append(direction)
        all_matched_indices.append(matched_indices)

    return (
        torch.cat(all_labels),
        torch.cat(all_regression),
        torch.cat(all_centerness),
        torch.cat(all_direction),
        torch.cat(all_matched_indices),
    )
