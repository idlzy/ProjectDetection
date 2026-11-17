import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from project_detection.config import load_config
from project_detection.engine import evaluate
from project_detection.onnx_evaluation import (
    OnnxEvaluationModel, build_inference_backend, normalize_hat_boxes,
    validate_dataset_contract,
)


def config():
    result = load_config("configs/experiments/fcos3d_exp_pgda_cop_nr_lgc.yaml")
    result["data"]["image_size"] = [128, 128]
    result["evaluation"].update(classes=["car"], score_threshold=.1,
                                 head_level_stats=True, cop_stats=True, cop_ablation=True)
    result["runtime"].update(device="cpu", val_log_every=0)
    return result


def constant_model(tmp_path, backend):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tools.onnx_contract import output_names
    from tools.infer_onnx_hat import NATIVE_CHANNELS
    c = config()
    arrays = {}
    if backend == "hat-onnx":
        for branch, channels in enumerate(NATIVE_CHANNELS + (9,)):
            for level, stride in enumerate([8, 16, 32, 64, 128]):
                value = np.zeros((1, channels, 128//stride, 128//stride), np.float32)
                if branch == 0:
                    value.fill(-30)
                    if level == 0:
                        value[0, 9, 0, 0] = 8
                elif branch == 2:
                    value.fill(np.log(10))
                elif branch == 3:
                    value[:] = np.log([4, 2, 1])[None, :, None, None]
                elif branch == 9:
                    value[:, 1] = 20
                arrays[str(branch*5+level)] = value
    else:
        channels = {"cls": 12, "bbox": 7, "direction": 2, "attribute": 9,
                    "centerness": 1, "depth_logits": 9}
        for name in output_names(c):
            level, field = name.split("_", 1)
            size = 128 // (2**int(level[1:]))
            value = np.zeros((1, channels[field], size, size), np.float32)
            if field == "cls":
                value.fill(-30)
                if level == "p3":
                    value[0, 9, 0, 0] = 8
            elif field == "bbox":
                value[:, 2] = np.log(10)
                value[:, 3:6] = np.log([4, 2, 1])[None, :, None, None]
            elif field == "depth_logits":
                value[:, 1] = 20
            arrays[name] = value
    nodes = [onnx.helper.make_node("Constant", [], [name],
                                  value=onnx.numpy_helper.from_array(value))
             for name, value in arrays.items()]
    graph = onnx.helper.make_graph(nodes, "constant_detector", [
        onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 128, 128])],
        [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(value.shape))
         for name, value in arrays.items()])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / (backend + ".onnx")
    if backend == "onnx":
        meta = {"format": "project_detection_raw_head_v1", "inference_config": copy.deepcopy(c),
                "probabilistic_depth": True, "attribute_prediction_mode": "adaptive",
                "depth_fuse_logit": 0., "output_names": list(arrays)}
        onnx.helper.set_model_props(model, {"project_detection_inference": json.dumps(meta)})
    else:
        path.with_suffix(".pgd.json").write_text(json.dumps({
            "unit": 10, "max": 80, "bins": 9, "depth_fuse_logit": 0., "fuse_w": .5}))
    onnx.save(model, str(path))
    return path, c


def target(token):
    return {"camera_matrix": torch.tensor([[100., 0, 0], [0, 100., 0], [0, 0, 1.]]),
            "distortion": torch.zeros(8), "sample_token": token,
            "camera_to_vehicle_rotation": torch.tensor([[0., 0, 1], [-1., 0, 0], [0, -1., 0]]),
            "camera_to_vehicle_translation": torch.zeros(3)}


@pytest.mark.parametrize("backend", ["onnx", "hat-onnx"])
@pytest.mark.parametrize("match_mode", ["fix", "relative"])
def test_onnx_uses_shared_metric_loop_with_batches_and_level_stats(tmp_path, backend, monkeypatch, match_mode):
    from project_detection import models
    monkeypatch.setattr(models, "build_model", lambda *a, **kw: pytest.fail("ONNX must not build a torch detector"))
    path, c = constant_model(tmp_path, backend)
    if match_mode == "relative":
        c["evaluation"].update(match_mode="relative", distance_thresholds=[.05, .1, .15, .2],
                               tp_distance_threshold=.1)
    else:
        c["evaluation"].update(match_mode="fix", distance_thresholds=[.5, 1., 2., 4.],
                               tp_distance_threshold=2.)
    # Exercise class-name mapping: model order must not overwrite GT order.
    c["data"]["classes"] = list(reversed(c["data"]["classes"]))
    model, processor, device = build_inference_backend(c, path, backend)
    assert device.type == "cpu"
    images = torch.zeros(2, 3, 128, 128)
    targets = [target("sample1"), target("sample2")]
    results = processor(model(images), targets)
    for t, prediction in zip(targets, results):
        assert len(prediction["scores"]) == 1
        assert prediction["labels"][0] == c["data"]["classes"].index("car")
        t.update({key: prediction[key].clone() for key in ("boxes3d", "boxes2d", "labels")})
    metrics = evaluate(model, [(images, targets)], c, device)
    assert metrics["num_samples"] == 2
    assert metrics["match_mode"] == match_mode
    assert metrics["num_tp"] == 2
    assert metrics["mAOE"] == pytest.approx(0, abs=1e-6)
    assert len(metrics["head_level_metrics"]) == 5
    assert metrics["head_level_metrics"]["P3_stride8"]["num_tp"] == 2
    assert not metrics["cop_diagnostics"]["enabled"]
    assert "ONNX" in metrics["cop_diagnostics"]["reason"]
    assert metrics["inference_backend"]["backend"] == backend
    if backend == "hat-onnx":
        assert model.contract["evaluation"]["nms_mode"] == "classwise"
        assert metrics["inference_backend"]["pgd_json_sha256"]
    # Distance filters must apply to GT and predictions, including level metrics.
    c["evaluation"]["class_range"] = {"car": 1}
    metrics = evaluate(model, [(images, targets)], c, device)
    assert metrics["num_gt"] == 0
    assert metrics["num_predictions"] == 0


@pytest.mark.parametrize("field,value", [
    ("image_size", [512, 896]), ("image_mean", [0, 0, 0]),
    ("image_std", [1, 1, 1]), ("pad_value", [128]*3),
    ("color_space", "rgb"), ("classes", ["car"]),
])
def test_reject_dataset_contract_mismatch(field, value):
    c = config()
    contract = copy.deepcopy(c)
    c["data"][field] = value
    with pytest.raises(ValueError):
        validate_dataset_contract(c, contract)


def test_hat_raw_box_convention_is_not_renderer_ret():
    boxes = np.array([[1, 2, 10, 4, 2, 1, 2*np.pi+.7, 0, 0]], np.float32)
    actual = normalize_hat_boxes(boxes)
    np.testing.assert_array_equal(actual[:, :6], boxes[:, :6])
    assert actual[0, 6] == pytest.approx(.7, abs=1e-6)
    assert boxes[0, 6] > 2*np.pi  # no input mutation


def test_native_hat_matches_existing_legacy_torch_decoder(tmp_path, monkeypatch):
    from project_detection.models.heads import LegacyHatFCOS3DHead
    from project_detection.task import FCOS3DPostProcessor
    from project_detection.task import postprocess as torch_decoder
    # This is a decode-convention regression, independent of CUDA-only NMS.
    monkeypatch.setattr(torch_decoder, "rotated_bev_nms",
                        lambda boxes, scores, *a, **kw: torch.arange(len(scores)))
    path, c = constant_model(tmp_path, "hat-onnx")
    c["evaluation"].update(nms_mode="classwise")
    model = OnnxEvaluationModel(c, path, "hat-onnx")
    values = model(torch.zeros(1, 3, 128, 128))
    t = target("sample")
    actual = model.evaluation_processor(values, [t])[0]
    maps = model.decoder.pack_native_outputs(values[0], model.contract)
    outputs = [{key.split("_", 1)[1]: torch.from_numpy(value)
                for key, value in maps.items() if key.startswith("p%d_" % level)
                and value is not None} for level in range(3, 8)]
    head = LegacyHatFCOS3DHead()
    legacy = FCOS3DPostProcessor(SimpleNamespace(head=head), c)(outputs, [t])[0]
    np.testing.assert_allclose(actual["boxes3d"],
                               normalize_hat_boxes(legacy["boxes3d"].numpy()), atol=1e-5)
    np.testing.assert_allclose(actual["scores"], legacy["scores"], atol=1e-6)


def test_pgd_argument_rejected_for_pytorch_before_model_loading():
    with pytest.raises(ValueError, match="hat-onnx"):
        build_inference_backend(config(), "missing.pth", "pytorch", "extra.json")
