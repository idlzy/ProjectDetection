from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from project_detection.task.postprocess import FCOS3DPostProcessor


class _Head:
    def decode_depth_candidates(self, raw_depth, probability):
        return torch.full_like(raw_depth, 10.0), torch.full_like(raw_depth, 0.1)

    def fuse_geometric_depth(self, local, geometric, weight, valid):
        return local


def test_score_threshold_is_applied_after_depth_confidence():
    model = SimpleNamespace(head=_Head())
    config = {
        "model": {"strides": [8], "geometry": {"enabled": False}},
        "data": {"classes": ["car"]},
        "evaluation": {
            "score_threshold": 0.35,
            "nms_pre": 10,
            "nms_threshold": 0.3,
            "nms_backend": "reference",
            "max_per_image": 10,
        },
    }
    processor = FCOS3DPostProcessor(model, config)
    prediction = {
        "cls": torch.tensor([[[[10.0]]]]),
        "centerness": torch.tensor([[[[10.0]]]]),
        "bbox": torch.zeros(1, 9, 1, 1),
        "depth_logits": torch.zeros(1, 1, 1, 1),
        "geo_weight": None,
    }
    target = {
        "camera_matrix": torch.eye(3),
        "distortion": torch.zeros(4),
    }

    result = processor([prediction], [target])[0]

    assert result["scores"].numel() == 0
