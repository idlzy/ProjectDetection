from __future__ import annotations

import torch
import torch.nn.functional as functional


def undistort_points(points, camera_matrix, coefficients, iterations=6):
    """Convert distorted pixels to normalized pinhole-camera rays."""
    xd = (points[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
    yd = (points[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1]
    x, y = xd.clone(), yd.clone()
    if coefficients is None:
        return x, y
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) / (
            1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2
        ).clamp_min(1e-6)
        delta_x = 2*p1*x*y + p2*(r2 + 2*x*x)
        delta_y = p1*(r2 + 2*y*y) + 2*p2*x*y
        x = (xd - delta_x) / radial.clamp_min(1e-6)
        y = (yd - delta_y) / radial.clamp_min(1e-6)
    return x, y


def horizon_relative_y(normalized_x, normalized_y, camera_to_vehicle_rotation=None):
    """Return normalized vertical distance from the ground-plane horizon."""
    if camera_to_vehicle_rotation is None:
        return normalized_y
    rotation = camera_to_vehicle_rotation.to(normalized_x)
    ground_normal_vehicle = rotation.new_tensor([0.0, 0.0, 1.0])
    ground_normal_camera = rotation.transpose(0, 1).matmul(ground_normal_vehicle)
    denominator = ground_normal_camera[1]
    safe_denominator = torch.where(
        denominator.abs() < 1e-6,
        denominator.new_tensor(1.0),
        denominator,
    )
    horizon_y = -(
        ground_normal_camera[0] * normalized_x + ground_normal_camera[2]
    ) / safe_denominator
    return normalized_y - horizon_y


def propagate_geometric_depth(
    centers2d,
    local_depth,
    dimensions_height,
    class_probabilities,
    depth_confidence,
    camera_matrix,
    distortion=None,
    camera_to_vehicle_rotation=None,
    image_size=None,
    enabled_mask=None,
    node_scores=None,
    instance_ids=None,
    topk_edges=8,
    max_nodes=128,
    min_horizon_distance=0.01,
    return_validity=False,
):
    """Build a parameter-free PGD graph and return geometric instance depth.

    Rows are target nodes and columns are source nodes. Graph inputs are
    detached by design, matching PGD's cut-off-gradient propagation stage.
    Nodes without a reliable incoming edge fall back to local depth.
    """
    if local_depth.ndim != 1:
        raise ValueError(
            "local_depth must have shape [N], got %s"
            % (tuple(local_depth.shape),)
        )
    if local_depth.numel() < 2:
        result = local_depth.detach()
        validity = torch.zeros_like(local_depth, dtype=torch.bool)
        return (result, validity) if return_validity else result

    centers = centers2d.detach()
    depth = local_depth.detach()
    heights = dimensions_height.detach()
    classes = class_probabilities.detach()
    confidence = depth_confidence.detach()
    device = depth.device
    count = depth.numel()
    expected_vectors = {
        "dimensions_height": heights,
        "depth_confidence": confidence,
    }
    for name, value in expected_vectors.items():
        if value.ndim != 1 or value.numel() != count:
            raise ValueError(
                "%s must have shape [%d], got %s"
                % (name, count, tuple(value.shape))
            )
    if centers.ndim != 2 or tuple(centers.shape) != (count, 2):
        raise ValueError(
            "centers2d must have shape [%d, 2], got %s"
            % (count, tuple(centers.shape))
        )
    if classes.ndim != 2 or classes.shape[0] != count:
        raise ValueError(
            "class_probabilities must have shape [%d, C], got %s"
            % (count, tuple(classes.shape))
        )
    if enabled_mask is None:
        enabled_mask = torch.ones(count, dtype=torch.bool, device=device)
    else:
        enabled_mask = enabled_mask.detach().bool()
        if enabled_mask.ndim != 1 or enabled_mask.numel() != count:
            raise ValueError(
                "enabled_mask must have shape [%d], got %s"
                % (count, tuple(enabled_mask.shape))
            )
    if node_scores is None:
        node_scores = confidence
    else:
        node_scores = node_scores.detach()
        if node_scores.ndim != 1 or node_scores.numel() != count:
            raise ValueError(
                "node_scores must have shape [%d], got %s"
                % (count, tuple(node_scores.shape))
            )

    # Select a bounded set directly.  Calling the one-argument torch.where on
    # all dense FCOS positives first can enter an overflowing CUDA nonzero
    # path on older PyTorch builds, even though the graph only needs 128 nodes.
    selection_count = min(count, max(int(max_nodes), 0) or count)
    selectable = enabled_mask & torch.isfinite(node_scores)
    selection_scores = node_scores.masked_fill(~selectable, -torch.inf)
    selected_scores, selected = selection_scores.topk(selection_count)
    selected = selected[selected_scores.isfinite()]
    if selected.numel() < 2:
        validity = torch.zeros_like(depth, dtype=torch.bool)
        return (depth, validity) if return_validity else depth

    selected_centers = centers[selected]
    selected_depth = depth[selected]
    selected_heights = heights[selected]
    selected_classes = classes[selected]
    selected_confidence = confidence[selected].clamp(0, 1)
    k = camera_matrix.detach().to(depth)
    coefficients = distortion.detach().to(depth) if distortion is not None else None
    normalized_x, normalized_y = undistort_points(selected_centers, k, coefficients)
    relative_y = horizon_relative_y(
        normalized_x,
        normalized_y,
        camera_to_vehicle_rotation.detach() if camera_to_vehicle_rotation is not None else None,
    )

    target_y = relative_y[:, None]
    source_y = relative_y[None, :]
    safe_target_y = torch.where(
        target_y.abs() < min_horizon_distance,
        target_y.sign().masked_fill(target_y == 0, 1.0) * min_horizon_distance,
        target_y,
    )
    propagated = (
        source_y / safe_target_y * selected_depth[None, :]
        + (selected_heights[None, :] - selected_heights[:, None])
        / (2.0 * safe_target_y)
    )

    if image_size is None:
        image_dimensions = torch.stack([2.0 * k[1, 2], 2.0 * k[0, 2]])
    else:
        image_dimensions = image_size.detach().to(depth)
    image_diagonal = torch.linalg.vector_norm(image_dimensions).clamp_min(1.0)
    distances = torch.cdist(selected_centers, selected_centers)
    distance_score = (1.0 - distances / image_diagonal).clamp(0, 1)
    normalized_classes = functional.normalize(selected_classes, dim=1, eps=1e-6)
    class_similarity = normalized_classes.matmul(normalized_classes.transpose(0, 1)).clamp(0, 1)
    edge_scores = selected_confidence[None, :] * distance_score * class_similarity

    valid = torch.isfinite(propagated) & (propagated > 0.1)
    valid &= torch.isfinite(edge_scores)
    valid &= relative_y.abs()[:, None] >= min_horizon_distance
    valid.fill_diagonal_(False)
    if instance_ids is not None:
        instance_ids = instance_ids.detach()
        if instance_ids.ndim != 1 or instance_ids.numel() != count:
            raise ValueError(
                "instance_ids must have shape [%d], got %s"
                % (count, tuple(instance_ids.shape))
            )
        ids = instance_ids[selected]
        valid &= ids[:, None] != ids[None, :]
    edge_scores = edge_scores.masked_fill(~valid, 0.0)

    edge_count = min(max(int(topk_edges), 1), selected.numel() - 1)
    values, indices = edge_scores.topk(edge_count, dim=1)
    propagated_topk = propagated.gather(1, indices)
    denominator = values.sum(dim=1)
    geometric_selected = (values * propagated_topk).sum(dim=1) / denominator.clamp_min(1e-6)
    geometric_selected = torch.where(
        denominator > 0,
        geometric_selected,
        selected_depth,
    )
    geometric = depth.clone()
    geometric[selected] = geometric_selected
    reliability = torch.zeros_like(depth, dtype=torch.bool)
    reliability[selected] = denominator > 0
    return (geometric, reliability) if return_validity else geometric
