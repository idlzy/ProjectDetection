import json

import pytest

pytest.importorskip("cv2")

from project_detection.reporting import write_test_report


def test_write_test_report_creates_json_and_png_outputs(tmp_path):
    metrics = {
        "NDS": 0.4,
        "mAP": 0.3,
        "mATE": 0.5,
        "mASE": 0.2,
        "mAOE": 0.3,
        "mADE": 0.45,
        "mRecall": 0.6,
        "mPrecision": 0.25,
        "F1": 0.35,
        "mAP_by_distance": {"0.5": 0.1, "1.0": 0.2, "2.0": 0.3, "4.0": 0.4},
        "per_class_AP": {"car": 0.5, "truck": 0.25},
        "per_class_AP_by_distance": {
            "car": {"0.5": 0.2, "1.0": 0.4, "2.0": 0.6, "4.0": 0.8},
            "truck": {"0.5": None, "1.0": 0.1, "2.0": 0.3, "4.0": 0.5},
        },
    }
    metric_path = tmp_path / "metrics.json"
    plot_dir = tmp_path / "plots"

    report = write_test_report(metrics, metric_path, plot_dir, {"checkpoint": "best.pth"})

    assert report["metrics"]["NDS"] == 0.4
    assert json.loads(metric_path.read_text())["protocol"] == "mw3d_camera_bev_center_distance"
    for filename in (
        "summary_bars.json", "summary_bars.png", "tp_errors.json", "tp_errors.png", "per_class_ap.json",
        "per_class_ap.png", "map_matrix.json", "map_matrix.png", "plots_index.json",
    ):
        assert (plot_dir / filename).stat().st_size > 0
