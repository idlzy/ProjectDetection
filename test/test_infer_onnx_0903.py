import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.infer_onnx_hat import postprocess, rotated_bev_nms


def test_0903_cli_needs_no_project_package(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools" / "infer_onnx_hat.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=str(tmp_path),
        env={"PYTHONPATH": ""}, capture_output=True, text=True, check=True,
    )
    assert "--calib-path" in completed.stdout


def test_0903_rotated_nms_uses_orientation():
    boxes = np.asarray([
        [0, 0, 10, 4, 2, 1, 0, 0, 0],
        [0, 0, 10, 4, 2, 1, 0, 0, 0],
        [0, 0, 10, 4, 2, 1, np.pi / 2, 0, 0],
    ], np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], np.float32)
    assert rotated_bev_nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_0903_decode_uses_negative_offset_and_direction_bin():
    config = {
        "data": {"classes": ["car"]},
        "model": {"strides": [8, 16, 32, 64, 128],
                  "depth_bin_unit": 10.0, "depth_bin_max": 20.0},
        "evaluation": {"score_threshold": 0.5, "nms_pre": 10,
                       "nms_threshold": 0.3, "max_per_image": 10},
    }
    outputs = {}
    for level in range(3, 8):
        raw = np.zeros((1, 9, 1, 1), np.float32)
        raw[0, 0, 0, 0] = 1.0
        raw[0, 2, 0, 0] = np.log(10.0)
        raw[0, 3:6, 0, 0] = np.log([4, 2, 1])
        direction = np.asarray([[[[0.0]], [[1.0]]]], np.float32)
        outputs.update({
            "p%d_cls" % level: np.asarray([[[[5.0 if level == 3 else -10.0]]]], np.float32),
            "p%d_bbox" % level: raw,
            "p%d_direction" % level: direction,
            "p%d_centerness" % level: np.asarray([[[[5.0]]]], np.float32),
            "p%d_depth_logits" % level: np.asarray([[[[0.0]], [[10.0]], [[0.0]]]], np.float32),
        })
    target = {"camera_matrix": np.asarray([[100, 0, 0], [0, 100, 0], [0, 0, 1]], np.float32),
              "dist": np.zeros(8, np.float32)}
    result = postprocess(outputs, target, config, {"depth_fuse_logit": 20.0})
    assert len(result["scores"]) == 1
    assert result["depth_confidence"].shape == result["scores"].shape
    assert result["depth_confidence"][0] == pytest.approx(
        (np.exp(10.0) + 1.0) / (np.exp(10.0) + 2.0) / 2.0, abs=1e-6,
    )
    np.testing.assert_allclose(result["boxes3d"][0, :6],
                               [-0.4, 0.4, 10, 4, 2, 1], rtol=1e-4, atol=1e-4)
    assert result["boxes3d"][0, 6] == pytest.approx(2 * np.pi + np.arctan(-0.04), abs=1e-4)
