from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import torch


# The suppression matrix is one byte per pair. Pairwise polygon intersection
# needs substantially more temporary storage, so it has an independent budget.
_GPU_SUPPRESSION_BUDGET_BYTES = 128 * 1024 * 1024
_PAIRWISE_TEMPORARY_BUDGET_BYTES = 256 * 1024 * 1024
_CUDA_FREE_MEMORY_RESERVE_BYTES = 256 * 1024 * 1024
_ESTIMATED_PAIRWISE_BYTES_PER_PAIR = 2048
_MAX_NMS_CANDIDATES = 1000
_MAX_BOUNDED_FALLBACK_CANDIDATES = 512


class _BoxGeometry(NamedTuple):
    boxes: torch.Tensor
    cosine: torch.Tensor
    sine: torch.Tensor
    corners: torch.Tensor
    area: torch.Tensor

    def sliced(self, index) -> "_BoxGeometry":
        return _BoxGeometry(*(value[index] for value in self))


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


def _corners_from_trigonometry(boxes, cosine, sine):
    signs = boxes.new_tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    local = signs[None] * boxes[:, None, 2:4] * 0.5
    x = cosine[:, None] * local[..., 0] - sine[:, None] * local[..., 1]
    y = sine[:, None] * local[..., 0] + cosine[:, None] * local[..., 1]
    return torch.stack((x + boxes[:, None, 0], y + boxes[:, None, 1]), dim=-1)


def _corners(boxes, clockwise):
    angles = boxes[:, 4] if clockwise else -boxes[:, 4]
    cosine, sine = angles.cos(), angles.sin()
    return _corners_from_trigonometry(boxes, cosine, sine)


def _precompute_box_geometry(boxes, clockwise=True):
    # Normalize the convention once. All following calculations use clockwise
    # angles, avoiding repeated clones and trigonometric kernels per chunk.
    if not clockwise:
        boxes = boxes.clone()
        boxes[:, 4].neg_()
    cosine, sine = boxes[:, 4].cos(), boxes[:, 4].sin()
    return _BoxGeometry(
        boxes=boxes,
        cosine=cosine,
        sine=sine,
        corners=_corners_from_trigonometry(boxes, cosine, sine),
        area=(boxes[:, 2] * boxes[:, 3]).abs(),
    )


def _cross_2d(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _points_inside_cached_rectangles(points, rectangles):
    """Test R sets of points against C cached oriented rectangles."""
    delta = points[:, None] - rectangles.boxes[None, :, None, :2]
    cosine = rectangles.cosine[None, :, None]
    sine = rectangles.sine[None, :, None]
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    epsilon = 1e-6
    return (
        (local_x.abs() <= rectangles.boxes[None, :, None, 2] * 0.5 + epsilon)
        & (local_y.abs() <= rectangles.boxes[None, :, None, 3] * 0.5 + epsilon)
    )


def _pairwise_rotated_iou_from_geometry(left, right):
    left_corners, right_corners = left.corners, right.corners
    rows, columns = left.boxes.shape[0], right.boxes.shape[0]

    left_inside = _points_inside_cached_rectangles(left_corners, right)
    right_inside = _points_inside_cached_rectangles(right_corners, left).permute(1, 0, 2)

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
    candidates = torch.cat((left_candidates, right_candidates, intersections), dim=2)
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
    ordered_points = torch.where(ordered_valid[..., None], ordered_points, first_point)
    next_points = ordered_points.roll(-1, dims=2)
    intersection_area = 0.5 * _cross_2d(ordered_points, next_points).sum(dim=2).abs()
    intersection_area = torch.where(
        valid_count.squeeze(2) >= 3,
        intersection_area,
        torch.zeros_like(intersection_area),
    )
    union = (
        left.area[:, None] + right.area[None] - intersection_area
    ).clamp_min(1e-7)
    return (intersection_area / union).clamp(0.0, 1.0)


def _pairwise_rotated_iou(left, right, clockwise=True):
    """Exact pairwise IoU for boxes in ``[cx, cy, w, h, angle]``."""
    return _pairwise_rotated_iou_from_geometry(
        _precompute_box_geometry(left, clockwise),
        _precompute_box_geometry(right, clockwise),
    )


def _automatic_chunk_size(count):
    if count <= 256:
        return max(count, 1)
    if count <= 1000:
        return 128
    return 64


def _resolve_pairwise_chunk_size(count, requested, temporary_budget_bytes):
    preferred = (
        _automatic_chunk_size(count) if requested is None else max(int(requested), 1)
    )
    bytes_per_row = max(count, 1) * _ESTIMATED_PAIRWISE_BYTES_PER_PAIR
    budget_rows = max(int(temporary_budget_bytes) // bytes_per_row, 1)
    return max(min(preferred, max(count, 1), budget_rows), 1)


def _cuda_free_bytes(device):
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except TypeError:  # Older PyTorch accepts no device argument.
        with torch.cuda.device(device):
            free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes)


def _effective_suppression_budget(device):
    available = max(_cuda_free_bytes(device) - _CUDA_FREE_MEMORY_RESERVE_BYTES, 0)
    return min(_GPU_SUPPRESSION_BUDGET_BYTES, available)


def _is_cuda_out_of_memory(error):
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        isinstance(error, out_of_memory_type)
        or "out of memory" in str(error).lower()
    )


def _greedy_keep_from_suppression(suppression):
    count = suppression.shape[0]
    removed = np.zeros(count, dtype=np.bool_)
    kept_sorted = []
    for index in range(count):
        if removed[index]:
            continue
        kept_sorted.append(index)
        removed[index + 1:] |= suppression[index, index + 1:]
    return kept_sorted


def _format_nms_result(
    boxes, scores, order, kept_sorted, candidate_indices=None
):
    kept_sorted = torch.as_tensor(
        kept_sorted, dtype=torch.long, device=boxes.device
    )
    keep = order[kept_sorted]
    if candidate_indices is not None:
        keep = candidate_indices[keep]
    dets = torch.cat((boxes[keep], scores[keep, None].to(boxes.dtype)), dim=1)
    return dets, keep


def _limit_sorted_candidates(scores, limit):
    """Return score-sorted candidate indices and the number discarded."""
    order = scores.argsort(descending=True)
    if order.numel() <= limit:
        return order, 0
    return order[:limit], order.numel() - limit


def _bounded_greedy_nms(geometry, threshold):
    """Exact O(N)-memory fallback for unusually large candidate sets."""
    count = geometry.boxes.shape[0]
    removed = np.zeros(count, dtype=np.bool_)
    kept_sorted = []
    for index in range(count):
        if removed[index]:
            continue
        kept_sorted.append(index)
        if index + 1 >= count:
            continue
        row_iou = _pairwise_rotated_iou_from_geometry(
            geometry.sliced(slice(index, index + 1)),
            geometry.sliced(slice(index + 1, None)),
        )
        suppressed = (row_iou[0] > threshold).cpu().numpy()
        removed[index + 1:] |= suppressed
    return kept_sorted


@torch.no_grad()
def nms_rotated(
    boxes,
    scores,
    iou_threshold,
    clockwise=True,
    pairwise_chunk_size=32,
):
    """Run exact rotated NMS using the project's CUDA tensor implementation.

    The public interface follows the former Horizon-style call signature, but
    this function has one implementation only and requires CUDA tensors.
    """
    _validate_inputs(boxes, scores, iou_threshold)
    count = boxes.shape[0]
    if count == 0:
        return (
            torch.empty((0, 6), dtype=boxes.dtype, device=boxes.device),
            torch.empty(0, dtype=torch.long, device=boxes.device),
        )

    candidate_indices, discarded = _limit_sorted_candidates(
        scores, _MAX_NMS_CANDIDATES
    )
    if discarded:
        warnings.warn(
            "rotated NMS received %d candidates; keeping the top %d to "
            "bound exact-IoU work" % (count, _MAX_NMS_CANDIDATES),
            RuntimeWarning,
            stacklevel=2,
        )
    candidate_boxes = boxes[candidate_indices]
    candidate_scores = scores[candidate_indices]
    threshold = float(iou_threshold)
    order = candidate_scores.argsort(descending=True)
    sorted_boxes = candidate_boxes[order].float().contiguous()
    count = sorted_boxes.shape[0]
    geometry = _precompute_box_geometry(sorted_boxes, clockwise)
    suppression_bytes = count * count
    suppression_budget = _effective_suppression_budget(boxes.device)

    if suppression_bytes > suppression_budget:
        if count > _MAX_BOUNDED_FALLBACK_CANDIDATES:
            warnings.warn(
                "rotated NMS CUDA workspace is below the matrix budget; "
                "keeping the top %d of %d candidates before the exact "
                "bounded-memory fallback"
                % (_MAX_BOUNDED_FALLBACK_CANDIDATES, count),
                RuntimeWarning,
                stacklevel=2,
            )
            order = order[:_MAX_BOUNDED_FALLBACK_CANDIDATES]
            geometry = _precompute_box_geometry(
                candidate_boxes[order].float().contiguous(), clockwise
            )
        else:
            warnings.warn(
                "rotated NMS CUDA workspace is below the matrix budget; "
                "using the exact bounded-memory fallback for %d candidates"
                % count,
                RuntimeWarning,
                stacklevel=2,
            )
        kept_sorted = _bounded_greedy_nms(geometry, threshold)
        return _format_nms_result(
            boxes, scores, order, kept_sorted, candidate_indices
        )

    try:
        suppression = torch.zeros(
            (count, count), dtype=torch.bool, device=boxes.device
        )
        free_after_matrix = max(
            _cuda_free_bytes(boxes.device) - _CUDA_FREE_MEMORY_RESERVE_BYTES,
            1,
        )
        temporary_budget = min(
            _PAIRWISE_TEMPORARY_BUDGET_BYTES, free_after_matrix
        )
        chunk_size = _resolve_pairwise_chunk_size(
            count, pairwise_chunk_size, temporary_budget
        )
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            iou = _pairwise_rotated_iou_from_geometry(
                geometry.sliced(slice(start, end)),
                geometry.sliced(slice(start, None)),
            )
            suppression[start:end, start:] = iou > threshold

        # The normal path synchronizes and transfers exactly once.
        suppression_cpu = suppression.cpu().numpy()
    except RuntimeError as error:
        if not _is_cuda_out_of_memory(error):
            raise
        warnings.warn(
            "rotated NMS CUDA workspace exhausted; using exact bounded-memory fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        if "suppression" in locals():
            del suppression
        if "iou" in locals():
            del iou
        torch.cuda.empty_cache()
        if count > _MAX_BOUNDED_FALLBACK_CANDIDATES:
            warnings.warn(
                "rotated NMS fallback is limiting candidates from %d to %d"
                % (count, _MAX_BOUNDED_FALLBACK_CANDIDATES),
                RuntimeWarning,
                stacklevel=2,
            )
            order = order[:_MAX_BOUNDED_FALLBACK_CANDIDATES]
            geometry = _precompute_box_geometry(
                candidate_boxes[order].float().contiguous(), clockwise
            )
        kept_sorted = _bounded_greedy_nms(geometry, threshold)
        return _format_nms_result(
            boxes, scores, order, kept_sorted, candidate_indices
        )

    kept_sorted = _greedy_keep_from_suppression(suppression_cpu)
    return _format_nms_result(
        boxes, scores, order, kept_sorted, candidate_indices
    )
