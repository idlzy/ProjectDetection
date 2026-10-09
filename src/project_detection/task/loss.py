from __future__ import annotations

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import nn

from .depth_propagation import propagate_geometric_depth
from .local_geometry import generalized_iou_loss, project_predicted_boxes_with_gt_depth
from .targets import assign_targets_batch, feature_points, prepare_target_batch


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    probability = logits.sigmoid()
    ce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma) * alpha_t
    return loss.sum() if reduction == "sum" else loss.mean()


def depth_bin_soft_targets(depth, depth_centers):
    """Interpolate GT depth between adjacent, strictly increasing bin centers.

    Out-of-range depths are clamped to the endpoint bins. Use FP32 to keep
    interpolation accurate when the training predictions use mixed precision.
    """
    centers = depth_centers.to(device=depth.device, dtype=torch.float32)
    depth = depth.detach().float().reshape(-1).contiguous()
    targets = depth.new_zeros((depth.numel(), centers.numel()))
    if centers.numel() == 1:
        return targets.fill_(1.0)
    depth = depth.clamp(min=centers[0], max=centers[-1])
    upper = torch.searchsorted(centers, depth).clamp(1, centers.numel() - 1)
    lower = upper - 1
    upper_weight = (depth - centers[lower]) / (centers[upper] - centers[lower])
    targets.scatter_add_(1, lower[:, None], (1 - upper_weight)[:, None])
    targets.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return targets


class FCOS3DLoss(nn.Module):
    def __init__(self, num_classes, strides, regress_ranges,
                 geometry_config=None, class_names=None,
                 target_assignment_chunk_size=8, uncertainty_loss_weight=0.1,
                 local_geometric_constraints=False):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.target_assignment_chunk_size = int(target_assignment_chunk_size)
        self.uncertainty_loss_weight = float(uncertainty_loss_weight)
        self.local_geometric_constraints = bool(local_geometric_constraints)
        self.geometry = geometry_config or {}
        enabled_names = set(self.geometry.get("classes", class_names or []))
        self.geometry_class_ids = {
            index for index, name in enumerate(class_names or [])
            if name in enabled_names
        }

    def forward(self, outputs, targets, head):
        if getattr(head, "attribute_prediction_mode", "parallel") == "adaptive":
            branch_losses = [
                self._forward_single(
                    [level[name] for level in outputs], targets, head
                )
                for name in ("parallel", "chain")
            ]
            return {
                key: (branch_losses[0][key] + branch_losses[1][key]) / 2
                for key in branch_losses[0]
            }
        return self._forward_single(outputs, targets, head)

    def _forward_single(self, outputs, targets, head):
        loss_cls = outputs[0]["cls"].new_tensor(0.0)
        loss_offset = loss_cls.clone()
        loss_depth = loss_cls.clone()
        loss_size = loss_cls.clone()
        loss_yaw = loss_cls.clone()
        loss_depth_cls = loss_cls.clone()
        loss_dir = loss_cls.clone()
        loss_center = loss_cls.clone()
        loss_uncertainty = loss_cls.clone()
        loss_bbox2d = loss_cls.clone()
        loss_consistency = loss_cls.clone()
        positives = loss_cls.new_zeros(())
        nodes_by_image = [[] for _ in targets]
        prepared_targets = prepare_target_batch(
            targets, outputs[0]["cls"].device
        )
        for prediction, stride, regress_range in zip(outputs, self.strides, self.regress_ranges):
            batch, _, height, width = prediction["cls"].shape
            points = feature_points(height, width, stride, prediction["cls"].device)
            cls_logits = prediction["cls"].permute(0, 2, 3, 1).reshape(
                batch, -1, self.num_classes
            )
            batched_assignment = assign_targets_batch(
                points,
                prepared_targets,
                regress_range,
                stride,
                image_chunk_size=self.target_assignment_chunk_size,
            )
            assignments = list(zip(*batched_assignment))

            labels_by_image = torch.stack(
                [assignment[0] for assignment in assignments]
            )
            one_hot = torch.zeros_like(cls_logits)
            positive_locations = torch.where(labels_by_image >= 0)
            if positive_locations[0].numel():
                one_hot[
                    positive_locations[0],
                    positive_locations[1],
                    labels_by_image[positive_locations],
                ] = 1
            loss_cls = loss_cls + sigmoid_focal_loss(cls_logits, one_hot)

            for image_index, assignment in enumerate(assignments):
                (
                    labels,
                    regression_target,
                    center_target,
                    direction_target,
                    matched_indices,
                ) = assignment
                positive = labels >= 0
                positives = positives + positive.sum()
                image_cls_logits = cls_logits[image_index]
                bbox_channels = prediction["bbox"].shape[1]
                raw = (
                    prediction["bbox"][image_index]
                    .permute(1, 2, 0)
                    .reshape(-1, bbox_channels)[positive]
                )
                depth_logits = None
                if prediction["depth_logits"] is not None:
                    depth_logits = prediction["depth_logits"][image_index].permute(1, 2, 0)
                    depth_logits = depth_logits.reshape(-1, depth_logits.shape[-1])[positive]
                geo_weight = None
                if prediction.get("geo_weight") is not None:
                    geo_weight = prediction["geo_weight"][image_index].reshape(-1)[positive]
                depth_log_scale = None
                if prediction.get("depth_log_scale") is not None:
                    depth_log_scale = prediction["depth_log_scale"][image_index].reshape(-1)[positive]
                bbox2d = None
                if self.local_geometric_constraints:
                    bbox2d = prediction["bbox2d"][image_index].permute(1, 2, 0)
                    bbox2d = bbox2d.reshape(-1, 4)[positive]
                nodes_by_image[image_index].append({
                    "raw": raw,
                    "target": regression_target[positive],
                    "centerness_target": center_target[positive],
                    "direction_target": direction_target[positive],
                    "direction_logits": prediction["direction"][image_index].permute(1, 2, 0).reshape(-1, 2)[positive],
                    "center_logits": prediction["centerness"][image_index].reshape(-1)[positive],
                    "depth_logits": depth_logits,
                    "geo_weight": geo_weight,
                    "depth_log_scale": depth_log_scale,
                    "bbox2d": bbox2d,
                    "points": points[positive],
                    "strides": raw.new_full((raw.shape[0],), float(stride)),
                    "centers2d": points[positive] + raw[:, :2] * float(stride),
                    "class_probabilities": image_cls_logits[positive].sigmoid(),
                    "labels": labels[positive],
                    "instance_ids": matched_indices[positive],
                })

        for image_index, chunks in enumerate(nodes_by_image):
            if not chunks or sum(chunk["raw"].shape[0] for chunk in chunks) == 0:
                continue
            raw = torch.cat([chunk["raw"] for chunk in chunks])
            regression_target = torch.cat([chunk["target"] for chunk in chunks])
            center_target = torch.cat([chunk["centerness_target"] for chunk in chunks])
            depth_logits = None
            if chunks[0]["depth_logits"] is not None:
                depth_logits = torch.cat([chunk["depth_logits"] for chunk in chunks])
            local_depth, depth_confidence = head.decode_depth_candidates(
                raw[:, 2], depth_logits
            )
            geometric_depth = local_depth.detach()
            geometry_valid = torch.zeros_like(local_depth, dtype=torch.bool)
            geo_weight = None
            if chunks[0]["geo_weight"] is not None:
                geo_weight = torch.cat([chunk["geo_weight"] for chunk in chunks])
            if self.geometry.get("enabled", False) and geo_weight is not None:
                labels = torch.cat([chunk["labels"] for chunk in chunks])
                enabled_mask = torch.zeros_like(labels, dtype=torch.bool)
                for class_id in self.geometry_class_ids:
                    enabled_mask |= labels == class_id
                target_meta = targets[image_index]
                geometric_depth, geometry_valid = propagate_geometric_depth(
                    centers2d=torch.cat([chunk["centers2d"] for chunk in chunks]),
                    local_depth=local_depth,
                    dimensions_height=raw[:, 4].exp(),
                    class_probabilities=torch.cat([chunk["class_probabilities"] for chunk in chunks]),
                    depth_confidence=depth_confidence,
                    camera_matrix=target_meta["camera_matrix"],
                    distortion=target_meta.get("distortion"),
                    camera_to_vehicle_rotation=target_meta.get("camera_to_vehicle_rotation"),
                    image_size=target_meta.get("image_size"),
                    enabled_mask=enabled_mask,
                    node_scores=center_target * depth_confidence,
                    instance_ids=torch.cat([chunk["instance_ids"] for chunk in chunks]),
                    topk_edges=self.geometry.get("topk_edges", 8),
                    max_nodes=self.geometry.get("max_nodes", 128),
                    min_horizon_distance=self.geometry.get("min_horizon_distance", 0.01),
                    return_validity=True,
                )
            depth = head.fuse_geometric_depth(
                local_depth, geometric_depth, geo_weight, geometry_valid
            )
            decoded = raw.clone()
            decoded[:, 2] = depth
            decoded[:, 3:6] = raw[:, 3:6].exp()
            regression_target = regression_target[:, : decoded.shape[1]]
            box_loss = functional.smooth_l1_loss(
                decoded[:, :6], regression_target[:, :6],
                beta=1.0/9.0, reduction="none"
            )
            box_loss = box_loss * center_target[:, None]
            loss_offset = loss_offset + box_loss[:, 0:2].sum()
            loss_depth = loss_depth + box_loss[:, 2].sum()
            if chunks[0]["depth_log_scale"] is not None:
                log_scale = torch.cat(
                    [chunk["depth_log_scale"] for chunk in chunks]
                ).float().clamp(-5.0, 5.0)
                residual = (depth.float() - regression_target[:, 2].float()).abs()
                loss_uncertainty = loss_uncertainty + (
                    (2.0 ** 0.5) * residual * (-log_scale).exp() + log_scale
                ).mul(center_target.float()).sum()
            loss_size = loss_size + box_loss[:, 3:6].sum()
            yaw_pred = torch.sin(decoded[:, 6]) * torch.cos(regression_target[:, 6])
            yaw_target = torch.cos(decoded[:, 6]) * torch.sin(regression_target[:, 6])
            yaw_loss = functional.smooth_l1_loss(
                yaw_pred, yaw_target, beta=1.0/9.0, reduction="none"
            )
            loss_yaw = loss_yaw + (yaw_loss * center_target).sum()
            if self.local_geometric_constraints:
                points = torch.cat([chunk["points"] for chunk in chunks])
                strides = torch.cat([chunk["strides"] for chunk in chunks])
                matched = torch.cat([chunk["instance_ids"] for chunk in chunks])
                predicted_ltrb = torch.cat([chunk["bbox2d"] for chunk in chunks])
                target_meta = targets[image_index]
                gt_boxes2d = target_meta["boxes2d"].to(points)[matched]
                target_ltrb = torch.stack((
                    points[:, 0] - gt_boxes2d[:, 0],
                    points[:, 1] - gt_boxes2d[:, 1],
                    gt_boxes2d[:, 2] - points[:, 0],
                    gt_boxes2d[:, 3] - points[:, 1],
                ), dim=1)
                # HAT permits positives outside the 2D GT box, but the
                # nonnegative bbox2d head cannot regress their signed ltrb.
                inside_gt_box = (target_ltrb > 0).all(dim=1)
                if inside_gt_box.any():
                    target_ltrb = target_ltrb[inside_gt_box] / strides[inside_gt_box, None]
                    predicted_ltrb = predicted_ltrb[inside_gt_box]
                    loss_bbox2d = loss_bbox2d + functional.smooth_l1_loss(
                        predicted_ltrb, target_ltrb, beta=1.0 / 9.0,
                        reduction="sum",
                    )
                    inside_points = points[inside_gt_box]
                    pixel_ltrb = predicted_ltrb * strides[inside_gt_box, None]
                    predicted_boxes2d = torch.stack((
                        inside_points[:, 0] - pixel_ltrb[:, 0],
                        inside_points[:, 1] - pixel_ltrb[:, 1],
                        inside_points[:, 0] + pixel_ltrb[:, 2],
                        inside_points[:, 1] + pixel_ltrb[:, 3],
                    ), dim=1)
                    gt_depth = target_meta["boxes3d"].to(points)[matched[inside_gt_box], 2]
                    projected_boxes2d, projection_valid = (
                        project_predicted_boxes_with_gt_depth(
                            centers2d=torch.cat(
                                [chunk["centers2d"] for chunk in chunks]
                            )[inside_gt_box].float(),
                            predicted_depth=depth[inside_gt_box].float(),
                            gt_depth=gt_depth.float(),
                            dimensions=decoded[inside_gt_box, 3:6].float(),
                            yaw=decoded[inside_gt_box, 6].float(),
                            camera_matrix=target_meta["camera_matrix"],
                            distortion=target_meta.get("distortion"),
                            camera_to_vehicle_rotation=target_meta.get(
                                "camera_to_vehicle_rotation"
                            ),
                        )
                    )
                    if projection_valid.any():
                        loss_consistency = loss_consistency + generalized_iou_loss(
                            projected_boxes2d[projection_valid],
                            predicted_boxes2d[projection_valid].float(),
                        ).sum()
            if depth_logits is not None:
                depth_bin_target = depth_bin_soft_targets(
                    regression_target[:, 2], head.depth_centers
                )
                depth_classification = -(
                    depth_bin_target
                    * functional.log_softmax(depth_logits.float(), dim=1)
                ).sum(dim=1)
                loss_depth_cls = loss_depth_cls + (
                    depth_classification * center_target
                ).sum()
            direction_logits = torch.cat([chunk["direction_logits"] for chunk in chunks])
            direction_target = torch.cat([chunk["direction_target"] for chunk in chunks])
            loss_dir = loss_dir + functional.cross_entropy(
                direction_logits, direction_target, reduction="sum"
            )
            center_logits = torch.cat([chunk["center_logits"] for chunk in chunks])
            loss_center = loss_center + functional.binary_cross_entropy_with_logits(
                center_logits, center_target, reduction="sum"
            )
        normalizer = positives.detach().clone()
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(normalizer, op=distributed.ReduceOp.SUM)
            # DDP averages gradients across ranks. Dividing each local loss by
            # global_positives / world_size yields a global sum/global count
            # after that gradient average.
            normalizer = normalizer / distributed.get_world_size()
        normalizer = normalizer.clamp_min(1)
        losses = {
            "loss_cls": loss_cls / normalizer,
            "loss_offset": loss_offset / normalizer,
            "loss_depth": loss_depth / normalizer,
            "loss_size": loss_size / normalizer,
            "loss_yaw": loss_yaw / normalizer,
            "loss_depth_cls": loss_depth_cls / normalizer,
            "loss_direction": loss_dir / normalizer,
            "loss_centerness": loss_center / normalizer,
        }
        if outputs[0].get("depth_log_scale") is not None:
            losses["loss_uncertainty"] = (
                self.uncertainty_loss_weight * loss_uncertainty / normalizer
            )
        if self.local_geometric_constraints:
            losses["loss_bbox2d"] = loss_bbox2d / normalizer
            losses["loss_consistency"] = loss_consistency / normalizer
        losses["loss_total"] = sum(losses.values())
        return losses
