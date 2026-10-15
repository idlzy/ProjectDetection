"""GT-assigned foreground regions shared by all distillation components."""

from typing import NamedTuple

import torch

from ..targets import assign_targets_batch, feature_points, prepare_target_batch


class DistillationRegion(NamedTuple):
    mask: torch.Tensor
    gt_depth: torch.Tensor
    centerness: torch.Tensor
    positive_count: torch.Tensor
    bbox2d_valid: torch.Tensor


def distillation_regions(outputs, targets, config):
    regions = []
    device = outputs[0]["cls"].device
    prepared = prepare_target_batch(targets, device)
    for output, stride, regress_range in zip(
        outputs, config["model"]["strides"], config["model"]["regress_ranges"]
    ):
        height, width = output["cls"].shape[-2:]
        points = feature_points(height, width, stride, output["cls"].device)
        labels_batch, regression_batch, center_batch, _, matched_batch = assign_targets_batch(
            points, prepared, regress_range, stride,
            image_chunk_size=config["train"].get("target_assignment_chunk_size", 8),
        )
        level = []
        for image, target in enumerate(targets):
            labels, regression, center = labels_batch[image], regression_batch[image], center_batch[image]
            positive = labels >= 0
            mask = positive.clone()
            region = config["train"]["distillation"].get("region", {})
            if region.get("exclude_padding", True):
                if "valid_image_size" not in target:
                    raise ValueError("Distillation requires valid_image_size")
                valid_size = target["valid_image_size"].to(points.device)
                mask &= (points[:, 0] < valid_size[1]) & (points[:, 1] < valid_size[0])
            inside = torch.zeros_like(mask)
            if positive.any():
                boxes = target["boxes2d"].to(points)[matched_batch[image][positive]]
                locations = points[positive]
                inside[positive] = ((locations > boxes[:, :2]).all(1)
                                    & (locations < boxes[:, 2:]).all(1))
            level.append(DistillationRegion(mask, regression[:, 2], center,
                                            positive.sum(), inside))
        regions.append(level)
    return regions
