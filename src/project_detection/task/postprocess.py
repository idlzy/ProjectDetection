from __future__ import annotations

import math

import numpy as np
import torch

from ..ops import nms_rotated as _project_cuda_nms_rotated
from .depth_propagation import propagate_geometric_depth, undistort_points
from .targets import feature_points

try:
    from horizon_plugin_pytorch.functional import nms_rotated as _horizon_nms_rotated
except ImportError:
    _horizon_nms_rotated = None


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


def _rectangle_corners(box):
    x, z, length, width, yaw = box
    local = np.asarray([[-length/2, -width/2], [length/2, -width/2],
                        [length/2, width/2], [-length/2, width/2]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.asarray([[c, -s], [s, c]]).T + np.asarray([x, z])


def _polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    polygon = np.asarray(polygon)
    return abs(float(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                     np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))) / 2


def _polygon_clip(subject, clip):
    output = [point for point in subject]
    orientation = np.sign(np.cross(clip[1]-clip[0], clip[2]-clip[1])) or 1.0
    for index in range(len(clip)):
        edge_a, edge_b = clip[index], clip[(index+1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break
        def inside(point):
            return orientation * np.cross(edge_b-edge_a, point-edge_a) >= -1e-9
        def intersection(start, end):
            segment, edge = end-start, edge_b-edge_a
            denominator = np.cross(segment, edge)
            if abs(denominator) < 1e-12:
                return end
            return start + segment * (np.cross(edge_a-start, edge) / denominator)
        previous = input_points[-1]
        for current in input_points:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
    return output


def _reference_rotated_bev_nms(boxes3d, scores, threshold):
    """Portable exact-polygon reference implementation."""
    device = boxes3d.device
    compact = boxes3d[:, [0, 2, 3, 5, 6]].detach().cpu().double().numpy()
    order = scores.detach().cpu().numpy().argsort()[::-1]
    polygons = [_rectangle_corners(box) for box in compact]
    areas = np.asarray([max(_polygon_area(poly), 1e-9) for poly in polygons])
    keep = []
    while len(order):
        current = int(order[0]); keep.append(current); remaining = []
        for other in order[1:]:
            other = int(other)
            intersection = _polygon_area(_polygon_clip(polygons[current], polygons[other]))
            iou = intersection / max(areas[current] + areas[other] - intersection, 1e-9)
            if iou <= threshold:
                remaining.append(other)
        order = np.asarray(remaining, dtype=np.int64)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def rotated_bev_nms(boxes3d, scores, threshold, backend="auto"):
    """Run rotated BEV NMS using Horizon CUDA when available.

    Args:
        boxes3d: Camera boxes in ``[x,y,z,l,h,w,yaw,vx,vz]`` order.
        scores: Ranking score for each box.
        threshold: Rotated IoU suppression threshold.
        backend: ``auto``, ``horizon``, ``cuda`` or ``reference``.
    """
    use_horizon = backend == "horizon" or (
        backend == "auto" and boxes3d.is_cuda and _horizon_nms_rotated is not None
    )
    if use_horizon:
        if _horizon_nms_rotated is None:
            raise RuntimeError("Horizon NMS requested but horizon_plugin_pytorch is unavailable")
        if not boxes3d.is_cuda:
            raise RuntimeError("Horizon rotated NMS requires CUDA tensors")
        # Horizon expects [x_center, y_center, width, height, angle] and returns
        # indices into the original, unsorted input. Camera BEV uses x/z and l/w.
        boxes_xywhr = boxes3d[:, [0, 2, 3, 5, 6]].contiguous()
        _, keep = _horizon_nms_rotated(
            boxes_xywhr, scores.contiguous(), threshold, clockwise=True
        )
        return keep.long()
    use_project_cuda = backend == "cuda" or (
        backend == "auto" and boxes3d.is_cuda
    )
    if use_project_cuda:
        if not boxes3d.is_cuda:
            raise RuntimeError("Project rotated CUDA NMS requires CUDA tensors")
        boxes_xywhr = boxes3d[:, [0, 2, 3, 5, 6]].contiguous()
        _, keep = _project_cuda_nms_rotated(
            boxes_xywhr, scores.contiguous(), threshold, clockwise=True
        )
        return keep.long()
    if backend not in ("auto", "reference", "cuda"):
        raise ValueError("Unknown NMS backend: %s" % backend)
    return _reference_rotated_bev_nms(boxes3d, scores, threshold)


class FCOS3DPostProcessor:
    """Decode network predictions. Axis-aligned image NMS is an explicit MVP fallback."""

    def __init__(self, model, config):
        self.model = model
        self.strides = config["model"]["strides"]
        evaluation = config["evaluation"]
        self.score_threshold = evaluation["score_threshold"]
        self.nms_pre = evaluation["nms_pre"]
        self.nms_threshold = evaluation["nms_threshold"]
        self.nms_backend = evaluation.get("nms_backend", "auto")
        self.max_per_image = evaluation["max_per_image"]
        self.geometry = config["model"].get("geometry", {})
        enabled_names = set(self.geometry.get("classes", config["data"]["classes"]))
        self.geometry_class_ids = {
            index for index, name in enumerate(config["data"]["classes"])
            if name in enabled_names
        }

    @torch.no_grad()
    def __call__(self, outputs, targets):
        results = []
        for image_index in range(outputs[0]["cls"].shape[0]):
            raws, local_depths, depth_confidences = [], [], []
            centers, scores, labels, class_probabilities, geo_weights = [], [], [], [], []
            for prediction, stride in zip(outputs, self.strides):
                cls = prediction["cls"][image_index].sigmoid()
                center = prediction["centerness"][image_index].sigmoid()
                combined = torch.sqrt(cls * center)
                flat_scores, flat_indices = combined.flatten().topk(min(self.nms_pre, combined.numel()))
                keep = flat_scores >= self.score_threshold
                flat_scores, flat_indices = flat_scores[keep], flat_indices[keep]
                if not keep.any(): continue
                height, width = cls.shape[-2:]
                class_ids = flat_indices // (height * width)
                locations = flat_indices % (height * width)
                raw = prediction["bbox"][image_index].permute(1, 2, 0).reshape(-1, 9)[locations]
                probability = None
                if prediction["depth_logits"] is not None:
                    probability = prediction["depth_logits"][image_index].permute(1, 2, 0).reshape(-1, prediction["depth_logits"].shape[1])[locations]
                local_depth, depth_confidence = self.model.head.decode_depth_candidates(
                    raw[:, 2], probability
                )
                flat_scores = flat_scores * depth_confidence
                # The externally reported PGDA score includes depth confidence.
                # Apply the threshold again after fusion so visualization and
                # evaluation never contain scores below score_threshold.
                final_keep = flat_scores >= self.score_threshold
                if not final_keep.any():
                    continue
                flat_scores = flat_scores[final_keep]
                class_ids = class_ids[final_keep]
                locations = locations[final_keep]
                raw = raw[final_keep]
                local_depth = local_depth[final_keep]
                depth_confidence = depth_confidence[final_keep]
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
                results.append({"boxes3d": torch.empty((0,9),device=device), "scores": torch.empty(0,device=device), "labels": torch.empty(0,dtype=torch.long,device=device)})
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
            boxes3d = torch.cat([x[:,None], y[:,None], z[:,None], dims, yaw[:,None], raw[:,7:9]], 1)
            half_w = (k[0, 0] * dims[:, 2] / z.clamp_min(1e-3)) / 2
            half_h = (k[1, 1] * dims[:, 1] / z.clamp_min(1e-3)) / 2
            image_boxes = torch.stack([centers2d[:,0]-half_w, centers2d[:,1]-half_h,
                                       centers2d[:,0]+half_w, centers2d[:,1]+half_h], 1)
            kept = []
            for label in labels.unique():
                indices = torch.where(labels == label)[0]
                kept.append(indices[rotated_bev_nms(
                    boxes3d[indices], scores[indices], self.nms_threshold,
                    backend=self.nms_backend,
                )])
            kept = torch.cat(kept); kept = kept[scores[kept].argsort(descending=True)[:self.max_per_image]]
            result = {"boxes3d": boxes3d[kept], "scores": scores[kept], "labels": labels[kept], "boxes2d": image_boxes[kept]}
            if geo_weight is not None:
                result.update({
                    "depth_local": local_depth[kept],
                    "depth_geometric": geometric_depth[kept],
                    "depth_fusion_weight": geo_weight[kept].sigmoid(),
                    "geometry_valid": geometry_valid[kept],
                })
            results.append(result)
        return results
