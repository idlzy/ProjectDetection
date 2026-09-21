from __future__ import annotations

import torch

from ..ops import nms_rotated as _project_cuda_nms_rotated
from .depth_propagation import propagate_geometric_depth, undistort_points
from .targets import feature_points
import math

def axis_aligned_nms(boxes, scores, threshold):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order, keep = scores.argsort(descending=True), []
    while order.numel() > 0:
        current = order[0]; keep.append(current)
        if order.numel() == 1: break
        rest = order[1:]
        xx1 = torch.maximum(x1[current], x1[rest]); yy1 = torch.maximum(y1[current], y1[rest])
        xx2 = torch.minimum(x2[current], x2[rest]); yy2 = torch.minimum(y2[current], y2[rest])
        intersection = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        iou = intersection / (areas[current] + areas[rest] - intersection).clamp_min(1e-6)
        order = rest[iou <= threshold]
    return torch.stack(keep)


def rotated_bev_nms(
    boxes3d,
    scores,
    threshold,
    pairwise_chunk_size=32,
):
    """Run rotated BEV NMS using the project's CUDA implementation.

    Args:
        boxes3d: Camera boxes in ``[x,y,z,l,h,w,yaw,vx,vz]`` order.
        scores: Ranking score for each box.
        threshold: Rotated IoU suppression threshold.
    """
    if not boxes3d.is_cuda or not scores.is_cuda:
        raise RuntimeError("rotated BEV NMS requires CUDA tensors")
    boxes_xywhr = boxes3d[:, [0, 2, 3, 5, 6]].contiguous()
    _, keep = _project_cuda_nms_rotated(
        boxes_xywhr,
        scores.contiguous(),
        threshold,
        clockwise=True,
        pairwise_chunk_size=pairwise_chunk_size,
    )
    return keep.long()


class FCOS3DPostProcessor:
    """Decode network predictions. Axis-aligned image NMS is an explicit MVP fallback."""

    def __init__(self, model, config):
        self.model = model
        self.strides = config["model"]["strides"]
        evaluation = config["evaluation"]
        self.score_threshold = evaluation["score_threshold"]
        self.nms_pre = evaluation["nms_pre"]
        self.nms_threshold = evaluation["nms_threshold"]
        self.nms_pairwise_chunk_size = evaluation.get(
            "nms_pairwise_chunk_size", 32
        )
        self.max_per_image = evaluation["max_per_image"]
        evaluated_names = evaluation.get("classes")
        if evaluated_names is None:
            evaluated_names = config["data"]["classes"]
        evaluated_names = set(evaluated_names)
        self.evaluation_class_ids = {
            index for index, name in enumerate(config["data"]["classes"])
            if name in evaluated_names
        }
        self.geometry = config["model"].get("geometry", {})
        enabled_names = set(self.geometry.get("classes", config["data"]["classes"]))
        self.geometry_class_ids = {
            index for index, name in enumerate(config["data"]["classes"])
            if name in enabled_names
        }

    @torch.no_grad()
    def __call__(self, outputs, targets):
        if getattr(self.model.head, "legacy_hat_postprocess", False):
            return self._legacy_hat(outputs, targets)
        results = []
        for image_index in range(outputs[0]["cls"].shape[0]):
            raws, local_depths, depth_confidences = [], [], []
            centers, scores, labels, class_probabilities, geo_weights = [], [], [], [], []
            for prediction, stride in zip(outputs, self.strides):
                cls = prediction["cls"][image_index].sigmoid()
                center = prediction["centerness"][image_index].sigmoid()
                combined = torch.sqrt(cls * center)
                if len(self.evaluation_class_ids) != cls.shape[0]:
                    enabled = torch.zeros(cls.shape[0], dtype=torch.bool, device=cls.device)
                    enabled[list(self.evaluation_class_ids)] = True
                    combined = combined.masked_fill(~enabled[:, None, None], -1.0)
                flat_scores, flat_indices = combined.flatten().topk(min(self.nms_pre, combined.numel()))
                keep = flat_scores >= self.score_threshold
                flat_scores, flat_indices = flat_scores[keep], flat_indices[keep]
                if not keep.any(): continue
                height, width = cls.shape[-2:]
                class_ids = flat_indices // (height * width)
                locations = flat_indices % (height * width)
                bbox_channels = prediction["bbox"].shape[1]
                raw = (
                    prediction["bbox"][image_index]
                    .permute(1, 2, 0)
                    .reshape(-1, bbox_channels)[locations]
                )
                probability = None
                if prediction["depth_logits"] is not None:
                    probability = prediction["depth_logits"][image_index].permute(1, 2, 0).reshape(-1, prediction["depth_logits"].shape[1])[locations]
                local_depth, depth_confidence = self.model.head.decode_depth_candidates(
                    raw[:, 2], probability
                )
                points = feature_points(height, width, stride, raw.device)[locations]
                centers2d = points + raw[:, :2] * stride
                cls_vectors = cls.permute(1, 2, 0).reshape(-1, cls.shape[0])[locations]
                geo_weight = None
                if prediction.get("geo_weight") is not None:
                    geo_weight = prediction["geo_weight"][image_index].reshape(-1)[locations]
                raws.append(raw)
                local_depths.append(local_depth)
                depth_confidences.append(depth_confidence)
                centers.append(centers2d)
                class_probabilities.append(cls_vectors)
                if geo_weight is not None: geo_weights.append(geo_weight)
                scores.append(flat_scores); labels.append(class_ids)
            if not scores:
                device = outputs[0]["cls"].device
                results.append({"boxes3d": torch.empty((0,9),device=device), "scores": torch.empty(0,device=device), "depth_confidence": torch.empty(0,device=device), "labels": torch.empty(0,dtype=torch.long,device=device)})
                continue
            raw = torch.cat(raws)
            local_depth = torch.cat(local_depths)
            depth_confidence = torch.cat(depth_confidences)
            centers2d = torch.cat(centers)
            class_vectors = torch.cat(class_probabilities)
            scores = torch.cat(scores); labels = torch.cat(labels)
            geo_weight = torch.cat(geo_weights) if geo_weights else None
            geometric_depth = local_depth
            geometry_valid = torch.zeros_like(local_depth, dtype=torch.bool)
            target_meta = targets[image_index]
            k = target_meta["camera_matrix"].to(raw.device)
            coefficients = target_meta.get("distortion")
            if coefficients is not None: coefficients = coefficients.to(raw.device)
            if self.geometry.get("enabled", False) and geo_weight is not None:
                enabled_mask = torch.zeros_like(labels, dtype=torch.bool)
                for class_id in self.geometry_class_ids:
                    enabled_mask |= labels == class_id
                geometric_depth, geometry_valid = propagate_geometric_depth(
                    centers2d=centers2d,
                    local_depth=local_depth,
                    dimensions_height=raw[:, 4].exp(),
                    class_probabilities=class_vectors,
                    depth_confidence=depth_confidence,
                    camera_matrix=k,
                    distortion=coefficients,
                    camera_to_vehicle_rotation=target_meta.get("camera_to_vehicle_rotation"),
                    image_size=target_meta.get("image_size"),
                    enabled_mask=enabled_mask,
                    node_scores=scores,
                    topk_edges=self.geometry.get("topk_edges", 8),
                    max_nodes=self.geometry.get("max_nodes", 128),
                    min_horizon_distance=self.geometry.get("min_horizon_distance", 0.01),
                    return_validity=True,
                )
            z = self.model.head.fuse_geometric_depth(
                local_depth, geometric_depth, geo_weight, geometry_valid
            )
            normalized_x, normalized_y = undistort_points(centers2d, k, coefficients)
            x = normalized_x * z; y = normalized_y * z
            dims = raw[:, 3:6].exp(); yaw = raw[:, 6]
            velocity = raw.new_zeros((raw.shape[0], 2))
            boxes3d = torch.cat(
                [
                    x[:, None], y[:, None], z[:, None], dims,
                    yaw[:, None], velocity,
                ],
                1,
            )
            half_w = (k[0, 0] * dims[:, 2] / z.clamp_min(1e-3)) / 2
            half_h = (k[1, 1] * dims[:, 1] / z.clamp_min(1e-3)) / 2
            image_boxes = torch.stack([centers2d[:,0]-half_w, centers2d[:,1]-half_h,
                                       centers2d[:,0]+half_w, centers2d[:,1]+half_h], 1)
            kept = []
            for label in labels.unique():
                indices = torch.where(labels == label)[0]
                kept.append(indices[rotated_bev_nms(
                    boxes3d[indices], scores[indices], self.nms_threshold,
                    pairwise_chunk_size=self.nms_pairwise_chunk_size,
                )])
            kept = torch.cat(kept); kept = kept[scores[kept].argsort(descending=True)[:self.max_per_image]]
            result = {"boxes3d": boxes3d[kept], "scores": scores[kept], "depth_confidence": depth_confidence[kept], "labels": labels[kept], "boxes2d": image_boxes[kept]}
            if geo_weight is not None:
                result.update({
                    "depth_local": local_depth[kept],
                    "depth_geometric": geometric_depth[kept],
                    "depth_fusion_weight": geo_weight[kept].sigmoid(),
                    "geometry_valid": geometry_valid[kept],
                })
            results.append(result)
        return results

    @torch.no_grad()
    def _legacy_hat(self, outputs, targets):
        """Decode legacy HAT predictions using their original conventions.

        HAT regressed ``point - center`` (the current trainer uses the opposite
        sign), thresholded ``class * centerness`` and used probabilistic depth
        confidence only for candidate ranking.  Direction bins also have to be
        folded into local yaw before adding the camera-ray angle.
        """
        results = []
        for image_index in range(outputs[0]["cls"].shape[0]):
            boxes_by_level = []
            image_boxes_by_level = []
            class_scores_by_level = []
            rank_scores_by_level = []
            for prediction, stride in zip(outputs, self.strides):
                cls = prediction["cls"][image_index].sigmoid()
                centerness = prediction["centerness"][image_index].sigmoid()
                height, width = cls.shape[-2:]
                raw = prediction["bbox"][image_index].permute(1, 2, 0).reshape(-1, 9)
                cls_scores = cls.permute(1, 2, 0).reshape(-1, cls.shape[0])
                center_scores = centerness.reshape(-1)
                direction = (
                    prediction["direction"][image_index]
                    .permute(1, 2, 0)
                    .reshape(-1, 2)
                    .argmax(dim=1)
                )
                depth_logits = prediction.get("depth_logits")
                if depth_logits is not None:
                    depth_logits = (
                        depth_logits[image_index]
                        .permute(1, 2, 0)
                        .reshape(-1, depth_logits.shape[1])
                    )
                depth, depth_confidence = self.model.head.decode_depth_candidates(
                    raw[:, 2], depth_logits
                )
                evaluated_ids = sorted(self.evaluation_class_ids)
                location_rank = (
                    cls_scores[:, evaluated_ids]
                    * center_scores[:, None]
                    * depth_confidence[:, None]
                ).max(dim=1).values
                if self.nms_pre > 0 and location_rank.numel() > self.nms_pre:
                    selected = location_rank.topk(self.nms_pre).indices
                    raw = raw[selected]
                    cls_scores = cls_scores[selected]
                    center_scores = center_scores[selected]
                    direction = direction[selected]
                    depth = depth[selected]
                    depth_confidence = depth_confidence[selected]
                else:
                    selected = None

                points = feature_points(height, width, stride, raw.device)
                if selected is not None:
                    points = points[selected]
                centers2d = points - raw[:, :2] * float(stride)
                target_meta = targets[image_index]
                camera = target_meta["camera_matrix"].to(raw.device)
                distortion = target_meta.get("distortion")
                if distortion is not None:
                    distortion = distortion.to(raw.device)
                normalized_x, normalized_y = undistort_points(
                    centers2d, camera, distortion
                )
                x = normalized_x * depth
                y = normalized_y * depth
                dimensions = raw[:, 3:6].exp()
                local_yaw = (
                    torch.remainder(raw[:, 6] - 0.7854, math.pi)
                    + 0.7854
                    + math.pi * direction.to(raw.dtype)
                )
                yaw = local_yaw + torch.atan2(normalized_x, torch.ones_like(normalized_x))
                boxes = torch.cat(
                    [
                        x[:, None], y[:, None], depth[:, None], dimensions,
                        yaw[:, None], raw[:, 7:9],
                    ],
                    dim=1,
                )
                half_width = (
                    camera[0, 0] * dimensions[:, 2] / depth.clamp_min(1e-3)
                ) / 2
                half_height = (
                    camera[1, 1] * dimensions[:, 1] / depth.clamp_min(1e-3)
                ) / 2
                image_boxes = torch.stack(
                    [
                        centers2d[:, 0] - half_width,
                        centers2d[:, 1] - half_height,
                        centers2d[:, 0] + half_width,
                        centers2d[:, 1] + half_height,
                    ],
                    dim=1,
                )
                detection_scores = cls_scores * center_scores[:, None]
                ranking_scores = detection_scores * depth_confidence[:, None]
                boxes_by_level.append(boxes)
                image_boxes_by_level.append(image_boxes)
                class_scores_by_level.append(detection_scores)
                rank_scores_by_level.append(ranking_scores)

            all_boxes = torch.cat(boxes_by_level)
            all_image_boxes = torch.cat(image_boxes_by_level)
            all_scores = torch.cat(class_scores_by_level)
            all_rank_scores = torch.cat(rank_scores_by_level)
            kept_boxes, kept_image_boxes = [], []
            kept_scores, kept_labels, kept_rank = [], [], []
            for label in sorted(self.evaluation_class_ids):
                valid = all_scores[:, label] >= self.score_threshold
                if not valid.any():
                    continue
                label_boxes = all_boxes[valid]
                label_image_boxes = all_image_boxes[valid]
                label_scores = all_scores[valid, label]
                label_rank = all_rank_scores[valid, label]
                keep = rotated_bev_nms(
                    label_boxes,
                    label_rank,
                    self.nms_threshold,
                    pairwise_chunk_size=self.nms_pairwise_chunk_size,
                )
                kept_boxes.append(label_boxes[keep])
                kept_image_boxes.append(label_image_boxes[keep])
                kept_scores.append(label_scores[keep])
                kept_rank.append(label_rank[keep])
                kept_labels.append(
                    torch.full(
                        (keep.numel(),), label, dtype=torch.long,
                        device=all_boxes.device,
                    )
                )
            if not kept_boxes:
                results.append(
                    {
                        "boxes3d": all_boxes.new_empty((0, 9)),
                        "scores": all_boxes.new_empty((0,)),
                        "labels": torch.empty(
                            0, dtype=torch.long, device=all_boxes.device
                        ),
                        "boxes2d": all_boxes.new_empty((0, 4)),
                    }
                )
                continue
            boxes = torch.cat(kept_boxes)
            image_boxes = torch.cat(kept_image_boxes)
            scores = torch.cat(kept_scores)
            labels = torch.cat(kept_labels)
            ranking = torch.cat(kept_rank)
            order = ranking.argsort(descending=True)[: self.max_per_image]
            results.append(
                {
                    "boxes3d": boxes[order],
                    "scores": scores[order],
                    "labels": labels[order],
                    "boxes2d": image_boxes[order],
                }
            )
        return results
