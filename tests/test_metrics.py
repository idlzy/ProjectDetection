import math

import pytest
import torch

from project_detection.metrics import Mw3dMetric


def _sample(predicted_boxes, ground_truth_boxes):
    predictions = [{
        "boxes3d": torch.tensor(predicted_boxes, dtype=torch.float32).reshape(-1, 9),
        "scores": torch.ones(len(predicted_boxes)),
        "labels": torch.zeros(len(predicted_boxes), dtype=torch.long),
    }]
    targets = [{
        "boxes3d": torch.tensor(ground_truth_boxes, dtype=torch.float32).reshape(-1, 9),
        "labels": torch.zeros(len(ground_truth_boxes), dtype=torch.long),
    }]
    return predictions, targets


def test_made_is_mean_absolute_camera_depth_error_for_matched_boxes():
    metric = Mw3dMetric(["car"], distance_thresholds=(2.0,), tp_threshold=2.0)
    predictions, targets = _sample(
        [
            [0, 0, 11, 4, 2, 2, 0, 0, 0],
            [0, 0, 19, 4, 2, 2, 0, 0, 0],
        ],
        [
            [0, 0, 10, 4, 2, 2, 0, 0, 0],
            [0, 0, 20, 4, 2, 2, 0, 0, 0],
        ],
    )

    metric.update(predictions, targets)

    assert metric.compute()["mADE"] == pytest.approx(1.0)


def test_made_is_nan_when_there_are_no_matched_boxes():
    metric = Mw3dMetric(["car"], distance_thresholds=(2.0,), tp_threshold=2.0)
    predictions, targets = _sample(
        [[0, 0, 20, 4, 2, 2, 0, 0, 0]],
        [[0, 0, 10, 4, 2, 2, 0, 0, 0]],
    )

    metric.update(predictions, targets)

    assert math.isnan(metric.compute()["mADE"])
