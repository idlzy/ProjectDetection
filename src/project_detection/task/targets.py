from __future__ import annotations

import math

import torch


def feature_points(height, width, stride, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
    )
    return torch.stack([(x.reshape(-1) + 0.5) * stride, (y.reshape(-1) + 0.5) * stride], dim=1)


def assign_targets(points, targets, regress_range, stride, center_radius=1.5):
    count = points.shape[0]; device = points.device
    labels = torch.full((count,), -1, dtype=torch.long, device=device)
    regression = torch.zeros((count, 9), device=device)
    centerness = torch.zeros((count,), device=device)
    direction = torch.zeros((count,), dtype=torch.long, device=device)
    matched_indices = torch.full((count,), -1, dtype=torch.long, device=device)
    boxes = targets["boxes2d"].to(device)
    if boxes.numel() == 0:
        return labels, regression, centerness, direction, matched_indices
    centers = targets["centers2d"].to(device); boxes3d = targets["boxes3d"].to(device)
    gt_labels = targets["labels"].to(device)
    px, py = points[:, 0:1], points[:, 1:2]
    left = px - boxes[:, 0]; top = py - boxes[:, 1]
    right = boxes[:, 2] - px; bottom = boxes[:, 3] - py
    distances = torch.stack([left, top, right, bottom], dim=2)
    inside_box = distances.min(dim=2).values > 0
    max_distance = distances.max(dim=2).values
    in_range = (max_distance >= regress_range[0]) & (max_distance <= regress_range[1])
    radius = stride * center_radius
    center_ok = (px - centers[:, 0]).abs() <= radius
    center_ok &= (py - centers[:, 1]).abs() <= radius
    valid = inside_box & in_range & center_ok
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).unsqueeze(0).repeat(count, 1)
    areas[~valid] = float("inf")
    min_area, matched = areas.min(dim=1)
    positive = torch.isfinite(min_area)
    if not positive.any():
        return labels, regression, centerness, direction, matched_indices
    match = matched[positive]
    labels[positive] = gt_labels[match]
    matched_indices[positive] = match
    regression[positive, :2] = (centers[match] - points[positive]) / float(stride)
    regression[positive, 2:] = boxes3d[match, 2:]
    selected = distances[positive, match]
    lr = selected[:, [0, 2]]; tb = selected[:, [1, 3]]
    center = torch.sqrt((lr.min(1).values / lr.max(1).values.clamp_min(1e-6)) *
                        (tb.min(1).values / tb.max(1).values.clamp_min(1e-6)))
    centerness[positive] = center.pow(2.5)
    yaw = regression[positive, 6]
    direction[positive] = ((yaw - 0.7854) % (2 * math.pi) >= math.pi).long()
    return labels, regression, centerness, direction, matched_indices
