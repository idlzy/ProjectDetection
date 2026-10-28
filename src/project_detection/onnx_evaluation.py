"""Dataset evaluation adapters for the two standalone CPU ONNX decoders."""
import copy
import hashlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def _decoder_module(backend):
    name = "infer_onnx_hat" if backend == "hat-onnx" else "infer_onnx"
    path = Path(__file__).resolve().parents[2] / "tools" / (name + ".py")
    if not path.is_file():
        raise FileNotFoundError("ONNX evaluation requires the repository tools: %s" % path)
    spec = importlib.util.spec_from_file_location("_eval_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def validate_dataset_contract(config, contract):
    """The loader and decoder must see exactly the same input preprocessing."""
    data, expected = config["data"], contract["data"]
    for key, default in (("image_size", None), ("image_mean", [128]*3),
                         ("image_std", [128]*3), ("pad_value", [0]*3)):
        if list(data.get(key, default)) != list(expected.get(key, default)):
            raise ValueError("Dataset %s disagrees with ONNX preprocessing" % key)
    if expected.get("color_space", "bgr").lower() != "bgr" or data.get("color_space", "bgr").lower() != "bgr":
        raise ValueError("Dataset evaluation currently requires BGR preprocessing")
    classes, model_classes = data["classes"], expected["classes"]
    if (len(set(classes)) != len(classes) or len(set(model_classes)) != len(model_classes)
            or set(classes) != set(model_classes)):
        raise ValueError("Dataset and ONNX class names must match (order may differ)")
    if list(config["model"]["strides"]) != list(contract["model"]["strides"]):
        raise ValueError("Dataset FPN strides disagree with ONNX contract")


def normalize_hat_boxes(boxes):
    """Native decoder already returns the evaluator's L,H,W,camera-yaw.

    Match FCOS3DPostProcessor._legacy_hat. Do NOT apply the reference
    renderer's final W,L,H,-yaw ('ret') conversion: that is a separate
    visualization convention, not the raw decoded camera-box interface.
    """
    boxes = boxes.copy()
    if len(boxes):
        boxes[:, 6] = (boxes[:, 6] + np.pi) % (2*np.pi) - np.pi
    return boxes


class OnnxEvaluationModel(nn.Module):
    """Use the normal evaluation loop, running static batch-1 ONNX per sample."""

    def __init__(self, config, checkpoint, backend, pgd_json=None):
        super().__init__()
        import onnxruntime as ort
        self.head = SimpleNamespace(attribute_prediction_mode="parallel")
        self.backend = backend
        self.decoder = _decoder_module(backend)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(checkpoint), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        if backend == "hat-onnx":
            contract, self.metadata = self.decoder.load_contract(
                self.session, checkpoint, pgd_json,
                class_order="bgr", color_space="bgr", decode_mode="distort")
        else:
            if pgd_json:
                raise ValueError("--pgd-json is only valid with --backend hat-onnx")
            contract, self.metadata = self.decoder.load_inference_contract(self.session, checkpoint)
        validate_dataset_contract(config, contract)
        self.contract = copy.deepcopy(contract)
        self.contract["evaluation"].update(config["evaluation"])
        if backend == "hat-onnx":
            # These are model conventions, never experiment-config overrides.
            self.contract["evaluation"].update(decode_mode="distort", nms_mode="classwise")
        self.label_map = np.array([config["data"]["classes"].index(name)
                                   for name in contract["data"]["classes"]], dtype=np.int64)
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.backend_metadata = {
            "backend": backend, "providers": self.session.get_providers(),
            "postprocess_device": "cpu", "model_format": self.metadata.get("format"),
            "input_shape": self.session.get_inputs()[0].shape,
            "onnxruntime_version": ort.__version__,
            "depth_fuse_logit": self.metadata.get("depth_fuse_logit"),
            "preprocessing": copy.deepcopy(contract["data"]),
            "decoding": copy.deepcopy(contract["model"]),
            "postprocessing": copy.deepcopy(self.contract["evaluation"]),
            "model_to_dataset_labels": self.label_map.tolist(),
            "cop_supported": False,
            "cop_reason": ("Native HAT ONNX has no CoP branches" if backend == "hat-onnx"
                           else "ONNX exports do not expose CoP branch diagnostics"),
        }
        if backend == "hat-onnx":
            self.backend_metadata.update(decode_mode="distort", class_order="bgr", color_space="bgr")
        sidecar = self.metadata.get("pgd_json")
        if sidecar:
            self.backend_metadata.update(pgd_json=sidecar,
                pgd_json_sha256=hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
                fuse_w=self.metadata["fuse_w"])
        if config["evaluation"].get("cop_stats") or config["evaluation"].get("cop_ablation"):
            logging.getLogger(__name__).warning("%s: CoP statistics/ablation unavailable; skipping", backend)

    def forward(self, images):
        outputs = []
        for image in images:
            tensor = np.ascontiguousarray(_numpy(image)[None], dtype=np.float32)
            validator = (self.decoder._validate_input if self.backend == "hat-onnx"
                         else self.decoder.validate_input)
            input_name = validator(self.session, tensor)
            values = self.session.run(None, {input_name: tensor})
            if any(not np.isfinite(value).all() for value in values):
                raise FloatingPointError("Non-finite ONNX output during evaluation")
            outputs.append(values)
        return outputs

    def evaluation_processor(self, outputs, targets):
        if len(outputs) != len(targets):
            raise ValueError("One ONNX result is required per dataset sample")
        results = []
        for values, target in zip(outputs, targets):
            calibration = {"camera_matrix": _numpy(target["camera_matrix"]),
                           "dist": _numpy(target["distortion"])}
            if self.backend == "hat-onnx":
                predictions = self.decoder.pack_native_outputs(values, self.contract)
            else:
                predictions = self.decoder.reconstruct_outputs(self.output_names, values, self.contract)
            result = self.decoder.postprocess(predictions, calibration, self.contract, self.metadata)
            if self.backend == "hat-onnx":
                result["boxes3d"] = normalize_hat_boxes(result["boxes3d"])
            result["labels"] = self.label_map[result["labels"]]
            results.append({key: torch.from_numpy(np.ascontiguousarray(value))
                            if isinstance(value, np.ndarray) else value
                            for key, value in result.items()})
        return results


def build_inference_backend(config, checkpoint, backend="pytorch", pgd_json=None):
    if backend == "pytorch":
        if pgd_json:
            raise ValueError("--pgd-json is only valid with --backend hat-onnx")
        from .engine import load_checkpoint
        from .models import build_model
        from .task import FCOS3DPostProcessor
        device = torch.device("cuda" if config["runtime"]["device"] == "cuda"
                              and torch.cuda.is_available() else "cpu")
        model = build_model(config).to(device)
        load_checkpoint(checkpoint, model, strict=True)
        model.eval()
        return model, FCOS3DPostProcessor(model, config), device
    if backend not in ("onnx", "hat-onnx"):
        raise ValueError("Unknown inference backend: %s" % backend)
    model = OnnxEvaluationModel(config, checkpoint, backend, pgd_json)
    model.eval()
    return model, model.evaluation_processor, torch.device("cpu")
