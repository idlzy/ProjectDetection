import json
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from project_detection.config import load_config
from project_detection.engine import (
    NonFiniteTrainingError,
    _named_tensors_are_finite,
    _nonfinite_tensor_details,
    _trip_nonfinite_guard,
)


def test_full_pgd_config_enables_nonfinite_guard():
    config = load_config(
        str(
            Path(__file__).parents[1]
            / "configs/experiments/fcos3d_r101_fpn_dcn_full_pgd.yaml"
        )
    )

    assert config["train"]["nonfinite_guard"] is True
    assert config["train"]["nonfinite_parameter_check_every"] == 100
    assert config["train"]["nonfinite_max_consecutive_amp_overflows"] == 8
    assert config["train"]["amp_initial_scale"] == 2048.0


def test_nonfinite_tensor_detection_reports_tensor_and_counts():
    tensors = [
        ("healthy", torch.tensor([1.0, 2.0])),
        ("broken", torch.tensor([float("nan"), float("inf"), -float("inf")])),
    ]

    assert not _named_tensors_are_finite(tensors)
    assert _nonfinite_tensor_details(tensors) == [
        {
            "name": "broken",
            "shape": [3],
            "dtype": "torch.float32",
            "nonfinite": 3,
            "nan": 1,
            "positive_inf": 1,
            "negative_inf": 1,
        }
    ]


def test_nonfinite_guard_writes_diagnostic_without_checkpoint(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    logger = logging.getLogger("test.nonfinite_guard")

    with pytest.raises(NonFiniteTrainingError, match="no checkpoint was written"):
        _trip_nonfinite_guard(
            logger=logger,
            output_dir=tmp_path,
            rank=0,
            kind="loss",
            epoch=2,
            step=4,
            steps_per_epoch=10,
            optimizer=optimizer,
            scaler=scaler,
            losses={"loss_total": torch.tensor(float("nan"))},
            details=[{"name": "loss_total", "nonfinite": 1}],
        )

    reports = list((tmp_path / "diagnostics").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["failure"] == "nonfinite_loss"
    assert report["epoch"] == 3
    assert report["step"] == 5
    assert report["global_step"] == 25
    assert report["losses"]["loss_total"] == "nan"
    assert report["checkpoint_written"] is False
    assert not (tmp_path / "checkpoints" / "last.pth").exists()
