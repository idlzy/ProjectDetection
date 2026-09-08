from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .depth_propagation import propagate_geometric_depth
from .targets import assign_targets, feature_points


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    probability = logits.sigmoid()
    ce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma) * alpha_t
    return loss.sum() if reduction == "sum" else loss.mean()


class FCOS3DLoss(nn.Module):
    def __init__(self, model, num_classes, strides, regress_ranges,
                 geometry_config=None, class_names=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.geometry = geometry_config or {}
        enabled_names = set(self.geometry.get("classes", class_names or []))
        self.geometry_class_ids = {
            index for index, name in enumerate(class_names or [])
            if name in enabled_names
        }

    def forward(self, outputs, targets):
        loss_cls = outputs[0]["cls"].new_tensor(0.0)
        loss_bbox = loss_cls.clone(); loss_dir = loss_cls.clone(); loss_center = loss_cls.clone()
        positives = 0
        nodes_by_image = [[] for _ in targets]
        for prediction, stride, regress_range in zip(outputs, self.strides, self.regress_ranges):
            batch, _, height, width = prediction["cls"].shape
            points = feature_points(height, width, stride, prediction["cls"].device)
            for image_index in range(batch):
                labels, regression_target, center_target, direction_target, matched_indices = assign_targets(
                    points, targets[image_index], regress_range, stride)
                cls_logits = prediction["cls"][image_index].permute(1, 2, 0).reshape(-1, self.num_classes)
                one_hot = torch.zeros_like(cls_logits)
                positive = labels >= 0
                if positive.any(): one_hot[positive, labels[positive]] = 1
                loss_cls = loss_cls + sigmoid_focal_loss(cls_logits, one_hot)
                if not positive.any():
                    continue
                positives += int(positive.sum())
                raw = prediction["bbox"][image_index].permute(1, 2, 0).reshape(-1, 9)[positive]
                depth_logits = None
                if prediction["depth_logits"] is not None:
                    depth_logits = prediction["depth_logits"][image_index].permute(1, 2, 0)
                    depth_logits = depth_logits.reshape(-1, depth_logits.shape[-1])[positive]
                geo_weight = None
                if prediction.get("geo_weight") is not None:
                    geo_weight = prediction["geo_weight"][image_index].reshape(-1)[positive]
                nodes_by_image[image_index].append({
                    "raw": raw,
                    "target": regression_target[positive],
                    "centerness_target": center_target[positive],
                    "direction_target": direction_target[positive],
                    "direction_logits": prediction["direction"][image_index].permute(1, 2, 0).reshape(-1, 2)[positive],
                    "center_logits": prediction["centerness"][image_index].reshape(-1)[positive],
                    "depth_logits": depth_logits,
                    "geo_weight": geo_weight,
                    "centers2d": points[positive] + raw[:, :2] * float(stride),
                    "class_probabilities": cls_logits[positive].sigmoid(),
                    "labels": labels[positive],
                    "instance_ids": matched_indices[positive],
                })

        for image_index, chunks in enumerate(nodes_by_image):
            if not chunks:
                continue
            raw = torch.cat([chunk["raw"] for chunk in chunks])
            regression_target = torch.cat([chunk["target"] for chunk in chunks])
            center_target = torch.cat([chunk["centerness_target"] for chunk in chunks])
            depth_logits = None
            if chunks[0]["depth_logits"] is not None:
                depth_logits = torch.cat([chunk["depth_logits"] for chunk in chunks])
            local_depth, depth_confidence = self.model.head.decode_depth_candidates(
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
            depth = self.model.head.fuse_geometric_depth(
                local_depth, geometric_depth, geo_weight, geometry_valid
            )
            decoded = raw.clone()
            decoded[:, 2] = depth
            decoded[:, 3:6] = raw[:, 3:6].exp()
            code_weight = decoded.new_tensor([1, 1, 1, 1, 1, 1, 1, 0, 0])
            box_loss = functional.smooth_l1_loss(
                decoded, regression_target, beta=1.0/9.0, reduction="none"
            )
            loss_bbox = loss_bbox + (box_loss * code_weight * center_target[:, None]).sum()
            direction_logits = torch.cat([chunk["direction_logits"] for chunk in chunks])
            direction_target = torch.cat([chunk["direction_target"] for chunk in chunks])
            loss_dir = loss_dir + functional.cross_entropy(
                direction_logits, direction_target, reduction="sum"
            )
            center_logits = torch.cat([chunk["center_logits"] for chunk in chunks])
            loss_center = loss_center + functional.binary_cross_entropy_with_logits(
                center_logits, center_target, reduction="sum"
            )
        normalizer = max(positives, 1)
        losses = {
            "loss_cls": loss_cls / normalizer,
            "loss_bbox": loss_bbox / normalizer,
            "loss_direction": loss_dir / normalizer,
            "loss_centerness": loss_center / normalizer,
        }
        losses["loss_total"] = sum(losses.values())
        return losses
