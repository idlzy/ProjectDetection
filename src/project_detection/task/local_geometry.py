"""Differentiable image projection for PGD-style local box consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .depth_propagation import undistort_points


_CORNER_SIGNS = (
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
)

# Beyond this normalized ray, a corner is far outside the calibrated image
# while the cubic distortion polynomial can overflow even in float32.
_MAX_NORMALIZED_COORD = 10.0
_MAX_PROJECTED_PIXEL = 1e6


def project_predicted_boxes_with_gt_depth(
    centers2d, predicted_depth, gt_depth, dimensions, yaw,
    camera_matrix, distortion=None, camera_to_vehicle_rotation=None,
):
    """Project predicted 3D boxes while replacing only their Z with GT depth.

    Like PGD, X/Y are first back-projected using predicted depth.  Replacing
    only Z therefore does not entirely remove depth gradients from this loss.
    Boxes use the project's camera-coordinate [length, height, width, yaw]
    convention.  The camera ground-normal recovers the vehicle-frame vertical
    axis when camera pitch or roll is present.
    """
    count = centers2d.shape[0]
    if count == 0:
        return centers2d.new_empty((0, 4)), torch.zeros(
            0, dtype=torch.bool, device=centers2d.device)
    if centers2d.shape != (count, 2) or dimensions.shape != (count, 3):
        raise ValueError("centers2d and dimensions must have shapes [N,2] and [N,3]")
    if any(value.shape != (count,) for value in (predicted_depth, gt_depth, yaw)):
        raise ValueError("depths and yaw must have shape [N]")

    camera_matrix = camera_matrix.to(centers2d)
    if distortion is not None:
        distortion = distortion.to(centers2d)
    principal = torch.stack((camera_matrix[0, 2], camera_matrix[1, 2]))
    focal = torch.stack((camera_matrix[0, 0], camera_matrix[1, 1]))
    center_rays = (centers2d - principal) / focal
    valid_centers = torch.isfinite(center_rays).all(dim=1)
    valid_centers &= (center_rays.abs() <= _MAX_NORMALIZED_COORD).all(dim=1)
    safe_centers = torch.where(valid_centers[:, None], centers2d, principal)
    ray_x, ray_y = undistort_points(safe_centers, camera_matrix, distortion)
    centers3d = torch.stack(
        (ray_x * predicted_depth, ray_y * predicted_depth, gt_depth), dim=1)

    if camera_to_vehicle_rotation is None:
        up = centers2d.new_tensor((0.0, -1.0, 0.0))
    else:
        rotation = camera_to_vehicle_rotation.to(centers2d)
        up = rotation.transpose(0, 1).matmul(
            centers2d.new_tensor((0.0, 0.0, 1.0)))
        up = functional.normalize(up, dim=0)
    if bool((up[1].abs() < 1e-4).item()):
        raise ValueError("Camera vertical axis must have a nonzero Y component")
    heading_x, heading_z = yaw.cos(), yaw.sin()
    heading_y = -(up[0] * heading_x + up[2] * heading_z) / up[1]
    heading = functional.normalize(
        torch.stack((heading_x, heading_y, heading_z), dim=1), dim=1)
    sideways = functional.normalize(
        torch.cross(up.expand_as(heading), heading, dim=1), dim=1)
    signs = centers2d.new_tensor(_CORNER_SIGNS)
    corners = (
        centers3d[:, None, :]
        + signs[None, :, 0:1] * dimensions[:, None, 0:1] / 2 * heading[:, None, :]
        + signs[None, :, 1:2] * dimensions[:, None, 2:3] / 2 * sideways[:, None, :]
        + signs[None, :, 2:3] * dimensions[:, None, 1:2] / 2 * up[None, None, :]
    )
    depth = corners[:, :, 2]
    valid_corners = valid_centers[:, None] & (depth > 0.1)
    valid_corners &= torch.isfinite(corners).all(dim=2)
    valid_corners &= (
        corners[:, :, :2].abs()
        <= depth[:, :, None] * _MAX_NORMALIZED_COORD
    ).all(dim=2)
    # Mask before division and distortion, not just before box aggregation:
    # otherwise a corner behind the camera can create 0 * inf in backward.
    safe_corners = torch.where(valid_corners[:, :, None], corners, 0.0)
    safe_depth = torch.where(valid_corners, depth, 1.0)
    x = safe_corners[:, :, 0] / safe_depth
    y = safe_corners[:, :, 1] / safe_depth
    if distortion is not None:
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion
        r2 = x.square() + y.square()
        radial = (1 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)) / (
            1 + k4 * r2 + k5 * r2.square() + k6 * r2.pow(3)
        ).clamp_min(1e-6)
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
        distorted_y = y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
        x, y = distorted_x, distorted_y
    pixel_x = camera_matrix[0, 0] * x + camera_matrix[0, 2]
    pixel_y = camera_matrix[1, 1] * y + camera_matrix[1, 2]
    projected_valid = valid_corners & torch.isfinite(pixel_x) & torch.isfinite(pixel_y)
    projected_valid &= (pixel_x.abs() <= _MAX_PROJECTED_PIXEL)
    projected_valid &= (pixel_y.abs() <= _MAX_PROJECTED_PIXEL)
    left = pixel_x.masked_fill(~projected_valid, float("inf")).min(dim=1).values
    top = pixel_y.masked_fill(~projected_valid, float("inf")).min(dim=1).values
    right = pixel_x.masked_fill(~projected_valid, float("-inf")).max(dim=1).values
    bottom = pixel_y.masked_fill(~projected_valid, float("-inf")).max(dim=1).values
    boxes = torch.stack((left, top, right, bottom), dim=1)
    valid = (projected_valid.sum(dim=1) >= 2) & torch.isfinite(boxes).all(dim=1)
    valid &= (right > left) & (bottom > top)
    return boxes, valid


def generalized_iou_loss(first, second):
    """Elementwise 1-GIoU for aligned xyxy boxes."""
    left_top = torch.maximum(first[:, :2], second[:, :2])
    right_bottom = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp_min(1e-6)
    enclosing_left_top = torch.minimum(first[:, :2], second[:, :2])
    enclosing_right_bottom = torch.maximum(first[:, 2:], second[:, 2:])
    enclosing_area = (
        enclosing_right_bottom - enclosing_left_top).clamp_min(0).prod(dim=1)
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-6)
    return 1 - giou
