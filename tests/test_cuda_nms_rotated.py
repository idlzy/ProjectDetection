import math

import numpy as np
import pytest
import torch

from project_detection.ops.nms_rotated import _pairwise_rotated_iou, nms_rotated
from project_detection.task.postprocess import (
    _polygon_area,
    _polygon_clip,
    _rectangle_corners,
    _reference_rotated_bev_nms,
    rotated_bev_nms,
)


def _reference_iou(left, right):
    left_polygon = _rectangle_corners(left)
    right_polygon = _rectangle_corners(right)
    intersection = _polygon_area(_polygon_clip(left_polygon, right_polygon))
    union = abs(left[2] * left[3]) + abs(right[2] * right[3]) - intersection
    return intersection / max(union, 1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_pairwise_rotated_iou_matches_reference():
    generator = np.random.RandomState(23)
    left = np.column_stack(
        [
            generator.normal(0, 4, 13),
            generator.normal(10, 4, 13),
            generator.uniform(0.5, 6, 13),
            generator.uniform(0.5, 3, 13),
            generator.uniform(-math.pi, math.pi, 13),
        ]
    ).astype(np.float32)
    right = np.column_stack(
        [
            generator.normal(0, 4, 11),
            generator.normal(10, 4, 11),
            generator.uniform(0.5, 6, 11),
            generator.uniform(0.5, 3, 11),
            generator.uniform(-math.pi, math.pi, 11),
        ]
    ).astype(np.float32)

    actual = _pairwise_rotated_iou(
        torch.from_numpy(left).cuda(), torch.from_numpy(right).cuda()
    ).cpu().numpy()
    expected = np.asarray(
        [[_reference_iou(a, b) for b in right] for a in left]
    )

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_nms_matches_reference_and_horizon_style_return():
    generator = np.random.RandomState(29)
    count = 80
    compact = np.column_stack(
        [
            generator.normal(0, 5, count),
            generator.normal(15, 5, count),
            generator.uniform(0.5, 6, count),
            generator.uniform(0.5, 3, count),
            generator.uniform(-1.5, 1.5, count),
        ]
    ).astype(np.float32)
    scores = generator.uniform(size=count).astype(np.float32)
    compact_cuda = torch.from_numpy(compact).cuda()
    scores_cuda = torch.from_numpy(scores).cuda()
    dets, keep = nms_rotated(compact_cuda, scores_cuda, 0.3, clockwise=True)
    boxes3d = torch.zeros(count, 9, device="cuda")
    boxes3d[:, [0, 2, 3, 5, 6]] = compact_cuda
    expected = _reference_rotated_bev_nms(boxes3d, scores_cuda, 0.3)

    assert keep.cpu().tolist() == expected.cpu().tolist()
    assert dets.shape == (keep.numel(), 6)
    torch.testing.assert_close(dets[:, :5], compact_cuda[keep])
    torch.testing.assert_close(dets[:, 5], scores_cuda[keep])
    assert rotated_bev_nms(
        boxes3d, scores_cuda, 0.3, backend="cuda"
    ).cpu().tolist() == expected.cpu().tolist()
    # With no Horizon plugin, auto selects the project CUDA implementation.
    assert rotated_bev_nms(
        boxes3d, scores_cuda, 0.3, backend="auto"
    ).cpu().tolist() == expected.cpu().tolist()


def test_cuda_nms_rejects_cpu_inputs():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        nms_rotated(torch.zeros(1, 5), torch.ones(1), 0.3)
