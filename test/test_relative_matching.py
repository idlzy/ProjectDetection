import json
import math

import numpy as np
import pytest
import torch

from project_detection.config import load_config, validate_config
from project_detection.engine import _build_metric
from project_detection.metrics import Mw3dMetric
from project_detection.reporting import _match_threshold_label, write_test_report


def box(z=100., x=0.):
    return [x, 0., z, 4., 2., 1., 0., 0., 0.]


def batch(predicted, gt, pred_labels=None, gt_labels=None):
    predictions = [{"boxes3d": torch.tensor(predicted).reshape(-1, 9),
                    "scores": torch.ones(len(predicted)),
                    "labels": torch.tensor(pred_labels or [0]*len(predicted), dtype=torch.long)}]
    targets = [{"boxes3d": torch.tensor(gt).reshape(-1, 9),
                "labels": torch.tensor(gt_labels or [0]*len(gt), dtype=torch.long),
                "camera_to_vehicle_rotation": torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]),
                "camera_to_vehicle_translation": torch.zeros(3)}]
    return predictions, targets


def metric(classes=("car",), mode="relative", **kwargs):
    return Mw3dMetric(classes, distance_thresholds=[.05, .1, .15, .2],
                      tp_distance_threshold=.1, match_mode=mode, **kwargs)


@pytest.mark.parametrize("distance,error,expected", [
    (100., 9.9, 1), (100., 10., 0), (10., .9, 1), (10., 1., 0),
])
def test_relative_tp_boundary_and_confusion(distance, error, expected):
    m = metric()
    p, t = batch([box(distance+error)], [box(distance)])
    m.update(p, t)
    result = m.compute()
    assert result["num_tp"] == expected
    assert result["confusion_matrix"]["counts"][0][0] == expected
    assert result["match_mode"] == "relative"
    assert result["threshold_unit"] == "ratio"
    if expected:
        assert result["mATE"] == pytest.approx(error, abs=1e-4)
        assert result["mADE"] == pytest.approx(error, abs=1e-4)


def test_gt_distance_not_prediction_distance():
    m = metric()
    p, t = batch([box(111.)], [box(100.)])
    m.update(p, t)
    assert m.compute()["num_tp"] == 0  # Not 11.1m tolerance from the prediction.


def test_vehicle_horizontal_reference_includes_translation():
    m = metric()
    p, t = batch([box(111.)], [box(100.)])
    t[0]["camera_to_vehicle_translation"][0] = 20.
    m.update(p, t)
    assert m.compute()["num_tp"] == 1  # 120m GT -> 12m tolerance.


def test_vehicle_horizontal_reference_includes_lateral_distance():
    m = metric()
    p, t = batch([box(60., 89.)], [box(60., 80.)])
    m.update(p, t)
    assert m.compute()["num_tp"] == 1  # sqrt(60^2+80^2)=100, not camera z=60.


def test_relative_selects_eligible_gt_using_normalized_error():
    m = metric()
    index, distance = m._select_match([1.1, 1.2], [0, 1], .1, np.array([10., 100.]))
    assert index == 1
    assert distance == 1.2
    p, t = batch([box(10.96)], [box(10.), box(12.)])
    m.update(p, t)
    accumulated = m._accumulate_class(0, .1)
    assert accumulated["matched_gt_depths"] == [12.]
    assert accumulated["num_tp"] == 1


def test_relative_one_to_one_matching_and_cross_class_confusion():
    m = metric(("car", "truck"))
    p, t = batch([box(105.), box(106.)], [box(100.)], pred_labels=[1, 1])
    m.update(p, t)
    r = m.compute()
    assert r["num_tp"] == 0  # AP is class-aware.
    assert r["confusion_matrix"]["counts"] == [[0, 1, 0], [0, 0, 0], [0, 1, 0]]


def test_fixed_default_unchanged_and_no_extrinsic_requirement():
    p, t = batch([box(105.)], [box(100.)])
    t[0].pop("camera_to_vehicle_rotation")
    t[0].pop("camera_to_vehicle_translation")
    default, explicit = Mw3dMetric(["car"]), Mw3dMetric(["car"], match_mode="fix")
    for m in (default, explicit):
        m.update(p, t)
    assert default.state_dict()[0][3].shape == (1, 9)
    assert len(default.state_dict()[0]) == 5
    assert default.compute()["num_tp"] == explicit.compute()["num_tp"] == 0
    np.testing.assert_equal(default.compute(), explicit.compute())


def test_relative_requires_extrinsics_and_state_preserves_ranges():
    p, t = batch([box(105.)], [box(100.)])
    bad = dict(t[0])
    bad.pop("camera_to_vehicle_rotation")
    with pytest.raises(ValueError, match="relative matching requires"):
        metric().update(p, [bad])
    m = metric()
    m.update(p, t)
    replica = metric()
    replica.load_state_dict(m.state_dict())
    assert replica.compute()["num_tp"] == 1
    with pytest.raises(ValueError, match="stored GT vehicle distances"):
        replica.load_state_dict([m.state_dict()[0][:5]])


def test_relative_zero_range_and_empty_samples_are_safe():
    m = metric()
    p, t = batch([box(0.)], [box(0.)])
    m.update(p, t)
    assert m.compute()["num_tp"] == 0  # Strict comparison: 0 < 0 is false.
    m = metric()
    p, t = batch([], [])
    m.update(p, t)
    assert m.compute()["num_gt"] == m.compute()["num_predictions"] == 0


def test_relative_with_class_range_keeps_all_diagnostics_in_scope():
    m = metric(class_range={"car": 110.})
    p, t = batch([box(105.), box(200.)], [box(100.), box(200.)])
    m.update(p, t)
    r = m.compute()
    assert r["num_tp"] == r["num_gt"] == r["num_predictions"] == 1
    assert r["confidence_distribution"]["total"]["num_predictions"] == 1
    assert m.samples[0][5].tolist() == [100.]


@pytest.mark.parametrize("mode,thresholds,tp", [
    ("fixed", [.1], .1), ("relative", [2.], 2.), ("fix", [0.], 0.),
    ("relative", [float("nan")], .1), ("relative", [.1], .2),
    ("relative", [.1, .1], .1), ("relative", [True], True),
])
def test_invalid_match_configuration(mode, thresholds, tp):
    c = load_config("configs/experiments/fcos3d_exp_pgda_cop_nr_lgc.yaml")
    c["evaluation"].update(match_mode=mode, distance_thresholds=thresholds, tp_distance_threshold=tp)
    with pytest.raises(ValueError):
        validate_config(c)
    with pytest.raises(ValueError):
        Mw3dMetric(["car"], distance_thresholds=thresholds, tp_distance_threshold=tp, match_mode=mode)


def test_engine_and_report_propagate_mode(tmp_path):
    c = load_config("configs/experiments/fcos3d_exp_pgda_cop_nr_lgc.yaml")
    c["evaluation"].update(match_mode="relative", distance_thresholds=[.05, .1, .15, .2],
                           tp_distance_threshold=.1, classes=["car"], class_range=None)
    m = _build_metric(c)
    p, t = batch([box(105.)], [box(100.)])
    p[0]["labels"][:] = c["data"]["classes"].index("car")
    t[0]["labels"][:] = c["data"]["classes"].index("car")
    m.update(p, t)
    result = m.compute()
    assert _match_threshold_label(.1, result) == "10%"
    assert _match_threshold_label(2., {}) == "2 m"
    output = tmp_path / "metrics.json"
    write_test_report(result, output, tmp_path / "plots", {"split": "test"})
    saved = json.loads(output.read_text())
    assert saved["protocol"] == "mw3d_relative_center_distance"
    assert saved["metadata"]["match_mode"] == "relative"
    assert saved["metadata"]["threshold_unit"] == "ratio"
    assert saved["metadata"]["relative_distance_reference"] == "gt_vehicle_xy"
    assert saved["metrics"]["confusion_matrix"]["match_mode"] == "relative"
