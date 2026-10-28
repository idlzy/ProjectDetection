"""Native HAT contract and CPU decoder regressions (no dataset required)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import infer_onnx_hat as native


def make_session(count=50):
    channels = native.NATIVE_CHANNELS + ((9,) if count == 50 else ())
    outputs = []
    for channel in channels:
        for stride in native.STRIDES:
            outputs.append(SimpleNamespace(name=str(len(outputs)),
                                           shape=[1, channel, 128 // stride, 128 // stride]))
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input.1", shape=[1, 3, 128, 128],
                                            type="tensor(float)")],
        get_outputs=lambda: outputs,
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={}),
    )


def contract(tmp_path, count=50, **changes):
    model = tmp_path / "float-mw.onnx"
    meta = {"unit": 10., "max": 80., "bins": 9, "prob_depth": True,
            "depth_fuse_logit": .1, "fuse_w": float(native._sigmoid(.1))}
    meta.update(changes)
    model.with_suffix(".pgd.json").write_text(json.dumps(meta))
    return native.load_contract(make_session(count), model)


def arrays(count=50):
    return [np.zeros(x.shape, np.float32) for x in make_session(count).get_outputs()]


def test_native_output_mapping_and_defaults(tmp_path):
    config, meta = contract(tmp_path)
    values = arrays()
    for i, value in enumerate(values):
        value.fill(i)
    output = native.pack_native_outputs(values, config)
    assert config["data"]["classes"][9] == "car"
    assert meta["fuse_w"] == pytest.approx(float(native._sigmoid(.1)))
    for level in range(5):
        prefix = "p%d_" % (level + 3)
        assert (output[prefix + "direction"] == 30 + level).all()
        assert (output[prefix + "attribute"] == 35 + level).all()
        assert (output[prefix + "centerness"] == 40 + level).all()
        assert (output[prefix + "depth_logits"] == 45 + level).all()
        expected = [5+level]*2 + [10+level] + [15+level]*3 + [20+level] + [25+level]*2
        assert output[prefix + "bbox"][0, :, 0, 0].tolist() == expected


def test_non_pgd_native_45_outputs(tmp_path):
    config, meta = contract(tmp_path, count=45)
    assert not config["model"]["probabilistic_depth"]
    assert meta["pgd_json"] is None
    output = native.pack_native_outputs(arrays(45), config)
    assert output["p3_depth_logits"] is None
    raw = np.zeros((1, 9), np.float32)
    raw[:, 2] = np.log(12.)
    depth, confidence = native._depth(raw, None, config, meta)
    np.testing.assert_allclose(depth, [12.])
    np.testing.assert_equal(confidence, [1.])


def test_pgd_requires_trained_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="--pgd-json"):
        native.load_contract(make_session(), tmp_path / "missing.onnx")


@pytest.mark.parametrize("changes", [
    {"bins": 8}, {"unit": 0}, {"max": float("nan")},
    {"depth_fuse_logit": None, "fuse_w": None},
    {"fuse_w": .9}, {"depth_fuse_logit": float("inf")},
])
def test_reject_invalid_pgd_json(tmp_path, changes):
    with pytest.raises(ValueError):
        contract(tmp_path, **changes)


def test_no_old_project_export_support(tmp_path):
    session = make_session()
    session.get_modelmeta = lambda: SimpleNamespace(custom_metadata_map={
        "project_detection_inference": "{}"})
    with pytest.raises(ValueError, match="re-exports are unsupported"):
        native.load_contract(session, tmp_path / "old.onnx")
    session = make_session()
    session.get_outputs = lambda: [SimpleNamespace(name="p3_cls", shape=[1, 12, 16, 16])]*30
    with pytest.raises(ValueError, match="45/50"):
        native.load_contract(session, tmp_path / "old.onnx")


def test_reject_wrong_branch_layout(tmp_path):
    config, _ = contract(tmp_path)
    values = arrays()
    values[0], values[5] = values[5], values[0]
    with pytest.raises(ValueError, match="Native output 0"):
        native.pack_native_outputs(values, config)


@pytest.mark.parametrize("mode", ["distort", "pinhole"])
@pytest.mark.parametrize("count", [45, 50])
def test_decoder_parity_with_original_reference(tmp_path, mode, count):
    pytest.importorskip("onnxruntime")
    path = Path(__file__).parent / "infer_onnx_deploy_viz.py"
    if not path.is_file():
        pytest.skip("Optional original HAT deploy reference is not included")
    spec = importlib.util.spec_from_file_location("hat_deploy_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # The reference has a legacy environment override; isolate this test from it.
    reference._depth_prior_from_env = lambda: None
    config, meta = contract(tmp_path, count=count)
    config["evaluation"].update(score_threshold=.1, nms_pre=30, max_per_image=8,
                                decode_mode=mode)
    rng = np.random.RandomState(17)
    values = arrays(count)
    for i, value in enumerate(values):
        branch = i // 5
        value[:] = rng.normal(0, .2, value.shape)
        if branch == 0:
            value.fill(-20)
        elif branch == 2:
            value += np.log(20.)
        elif branch == 3:
            value += np.log([4., 2., 1.])[None, :, None, None]
    for level in range(5):
        values[level][0, 9, 0, 0] = 2. + .1*level
        values[level][0, 10, 0, 0] = 1. + .1*level  # cross-class overlap must survive NMS
    k = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]], np.float32)
    dist = np.array([.1, -.01, .001, .002, 0, 0, 0, 0], np.float64)
    got = native.postprocess(native.pack_native_outputs(values, config),
                             {"camera_matrix": k, "dist": dist}, config, meta)
    boxes, scores, labels = reference.fcos3d_postprocess(
        values, k, score_thr=.1, nms_thr=.3, nms_pre=30, max_per_img=8,
        decode_mode=mode, dist=dist, fuse_w=meta.get("fuse_w", .5),
        prob_unit=10., prob_max=80., use_pgd=count == 50)
    # Reference renderer uses W,L,H,-yaw; native renderer retains L,H,W,yaw.
    boxes[:, 3:6] = boxes[:, [4, 5, 3]]
    boxes[:, 6] *= -1
    order = np.argsort(-scores)
    np.testing.assert_allclose(got["boxes3d"], boxes[order], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got["scores"], scores[order], rtol=1e-6)
    np.testing.assert_array_equal(got["labels"], labels[order])


def test_preprocessing_matches_reference(tmp_path):
    import cv2
    config, _ = contract(tmp_path)
    # Compare at native default resolution, including nontrivial padding and RGB.
    config["data"]["image_size"] = [512, 896]
    image = np.zeros((60, 100, 3), np.uint8)
    image[:] = [20, 100, 220]
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), image)
    calib = {"front": {"camera_location": "front_left",
                       "intrinsics": {"focal_length_px": {"fx": 100, "fy": 100},
                                      "principal_point_px": {"cx": 50, "cy": 30}},
                       "extrinsics": {}}}
    calibration = tmp_path / "calib.json"
    calibration.write_text(json.dumps(calib))
    original, tensor, target = native.prepare_sample(path, calibration, config)
    assert original.shape == image.shape
    np.testing.assert_allclose(tensor[0, :, 0, 0], (np.array([20, 100, 220])-128)/128)
    np.testing.assert_allclose(tensor[0, :, 0, -1], [-1, -1, -1])
    assert target["camera_matrix"][0, 0] == pytest.approx(100 * 512/60)


def test_camera_corners_match_native_hat_convention():
    pytest.importorskip("onnxruntime")
    path = Path(__file__).parent / "infer_onnx_deploy_viz.py"
    if not path.is_file():
        pytest.skip("Optional original HAT deploy reference is not included")
    spec = importlib.util.spec_from_file_location("hat_corner_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    rotation = native._rot_y(.03) @ rotation
    translation = np.array([1., .2, 1.5])
    box = np.array([2., .5, 20., 4., 2., 1., .7, 0., 0.])
    target = {"r_c2v": rotation, "t_c2v": translation}
    got = native._camera_corners(box, target)
    ret = box.copy()
    ret[3:6] = box[[5, 3, 4]]
    ret[6] *= -1
    expected = reference.box_to_camera_corners_swap_lh(ret, rotation, translation)
    # Different corner orders, same physical set of eight points.
    distances = np.linalg.norm(got[:, None] - expected[None], axis=2)
    assert distances.min(axis=1).max() < 5e-6
