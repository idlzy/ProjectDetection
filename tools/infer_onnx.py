#!/usr/bin/env python3
"""Run calibrated FCOS3D ONNX inference and render its predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from project_detection.config import load_config
from project_detection.data.geometry import load_front_left_calibration
from project_detection.engine import load_checkpoint
from project_detection.models import build_model
from project_detection.task import FCOS3DPostProcessor
from project_detection.visualization import draw_bev, draw_camera_view

if __package__:
    from tools.onnx_contract import output_fields, output_names
else:
    from onnx_contract import output_fields, output_names

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _require_available_output(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "%s already exists; pass --overwrite to replace it" % path
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(input_path, recursive=False, max_images=None):
    input_path = Path(input_path)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        candidates = input_path.rglob("*") if recursive else input_path.glob("*")
        images = sorted(
            path for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(input_path)
    if not images:
        raise FileNotFoundError(
            "No supported images found under %s" % input_path
        )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        images = images[:max_images]
    return images


def resolve_extrinsic_path(extri_path, image_path, input_path):
    """Resolve a shared TXT or a required per-frame TXT from a directory."""
    if extri_path is None:
        return None
    extri_path = Path(extri_path)
    if extri_path.is_file():
        return extri_path
    if not extri_path.is_dir():
        raise FileNotFoundError(extri_path)
    image_path = Path(image_path)
    input_path = Path(input_path)
    relative = (
        image_path.relative_to(input_path)
        if input_path.is_dir()
        else Path(image_path.name)
    )
    candidates = [
        extri_path / relative.with_suffix(".txt"),
        extri_path / relative.parent / "extrinsic.txt",
        extri_path / (image_path.stem + ".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Missing per-frame extrinsic for %s; tried: %s"
        % (image_path, ", ".join(str(path) for path in candidates))
    )


def prepare_sample(image_path, calibration_path, config, extrinsic_path=None):
    """Load image/calibration using the training and evaluation conventions."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)

    original_height, original_width = image.shape[:2]
    target_height, target_width = config["data"]["image_size"]
    scale = min(
        target_width / original_width,
        target_height / original_height,
    )
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height))

    canvas = np.empty((target_height, target_width, 3), dtype=np.float32)
    canvas[...] = np.asarray(
        config["data"].get("pad_value", [0, 0, 0]), dtype=np.float32
    )
    canvas[:resized_height, :resized_width] = resized

    calibration = load_front_left_calibration(
        calibration_path,
        (original_height, original_width),
        extrinsic_path=extrinsic_path,
    )
    camera_matrix = calibration["k"].copy()
    camera_matrix[0] *= scale
    camera_matrix[1] *= scale
    target = {
        "camera_matrix": torch.from_numpy(camera_matrix.astype(np.float32)),
        "distortion": torch.from_numpy(
            calibration["dist"].astype(np.float32)
        ),
        "camera_to_vehicle_rotation": torch.from_numpy(
            calibration["r_c2v"].astype(np.float32)
        ),
        "camera_to_vehicle_translation": torch.from_numpy(
            calibration["t_c2v"].astype(np.float32)
        ),
        "image_size": torch.tensor(
            [target_height, target_width], dtype=torch.int64
        ),
        "scale_factor": float(scale),
    }
    mean = np.asarray(
        config["data"].get("image_mean", [128, 128, 128]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    std = np.asarray(
        config["data"].get("image_std", [128, 128, 128]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    if np.any(std == 0):
        raise ValueError("Image normalization std must be non-zero")
    tensor = ((canvas - mean) / std).transpose(2, 0, 1)[None]
    return (
        image,
        np.ascontiguousarray(tensor, dtype=np.float32),
        target,
    )


def prepare_image(image_path, config):
    """Backward-compatible raw preprocessing helper without calibration."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: %s" % image_path)
    height, width = image.shape[:2]
    temporary_calibration = {
        "data": config["data"],
    }
    target_height, target_width = config["data"]["image_size"]
    scale = min(target_width / width, target_height / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    canvas = np.empty((target_height, target_width, 3), dtype=np.float32)
    canvas[...] = np.asarray(
        temporary_calibration["data"].get("pad_value", [0, 0, 0]),
        dtype=np.float32,
    )
    canvas[:resized_height, :resized_width] = cv2.resize(
        image, (resized_width, resized_height)
    )
    mean = np.asarray(
        config["data"].get("image_mean", [128, 128, 128]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    std = np.asarray(
        config["data"].get("image_std", [128, 128, 128]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    if np.any(std == 0):
        raise ValueError("Image normalization std must be non-zero")
    tensor = ((canvas - mean) / std).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def select_providers(available, requested=None):
    if requested:
        unavailable = [name for name in requested if name not in available]
        if unavailable:
            raise ValueError(
                "Unavailable ONNX Runtime providers: %s; available: %s"
                % (", ".join(unavailable), ", ".join(available))
            )
        return list(requested)
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def validate_input(session, tensor):
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Expected one ONNX input, found %d" % len(inputs))
    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise TypeError(
            "Expected float32 ONNX input, found %s" % model_input.type
        )
    expected = model_input.shape
    if len(expected) != tensor.ndim:
        raise ValueError(
            "Input rank mismatch: model=%s image=%s"
            % (expected, list(tensor.shape))
        )
    for axis, (expected_size, actual_size) in enumerate(
        zip(expected, tensor.shape)
    ):
        if isinstance(expected_size, int) and expected_size != actual_size:
            raise ValueError(
                "Input shape mismatch at axis %d: model=%s image=%s"
                % (axis, expected, list(tensor.shape))
            )
    return model_input.name


def summarize_outputs(names, values):
    if len(names) != len(values):
        raise ValueError("ONNX Runtime output names and values do not match")
    summary = []
    for name, value in zip(names, values):
        array = np.asarray(value)
        finite = bool(np.isfinite(array).all())
        if not finite:
            raise FloatingPointError("Non-finite values in ONNX output %s" % name)
        summary.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
            }
        )
    return summary


def reconstruct_outputs(names, values, config, device):
    """Rebuild the five dictionaries consumed by FCOS3DPostProcessor."""
    expected = output_names(config)
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            "ONNX output contract mismatch; missing=%s extra=%s"
            % (missing, extra)
        )
    named = dict(zip(names, values))
    predictions = []
    fields = output_fields(config)
    for level in range(3, 8):
        prediction = {
            field: torch.from_numpy(
                np.ascontiguousarray(named["p%d_%s" % (level, field)])
            ).to(device)
            for field in fields
        }
        prediction.setdefault("depth_logits", None)
        prediction.setdefault("geo_weight", None)
        predictions.append(prediction)
    return predictions


def load_postprocess_model(config, model_path, checkpoint, device):
    """Load learned host-postprocess state from checkpoint or export metadata."""
    model_path = Path(model_path)
    model = build_model(config)
    checkpoint_path = Path(checkpoint) if checkpoint else model_path.with_suffix(
        ".pth"
    )
    if checkpoint_path.is_file():
        load_checkpoint(checkpoint_path, model, strict=True)
        state_source = str(checkpoint_path)
    else:
        metadata_path = model_path.with_suffix(".meta.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Post-processing needs learned depth fusion state; provide "
                "--checkpoint or place %s next to the ONNX model"
                % metadata_path.name
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        exported_mode = metadata.get("attribute_prediction_mode")
        configured_mode = config["model"].get("attribute_prediction_mode", "parallel")
        if exported_mode is not None and exported_mode != configured_mode:
            raise ValueError(
                "ONNX attribute prediction mode mismatch: exported=%s configured=%s"
                % (exported_mode, configured_mode)
            )
        exported_threshold = metadata.get("chain_reliability_threshold")
        if exported_mode == "adaptive" and exported_threshold is not None:
            configured_threshold = config["model"].get(
                "chain_reliability_threshold", 0.2
            )
            if float(exported_threshold) != float(configured_threshold):
                raise ValueError("ONNX chain reliability threshold mismatch")
        fuse_logit = metadata.get("depth_fuse_logit")
        if model.head.depth_fuse_logit is not None:
            if fuse_logit is None:
                raise ValueError(
                    "depth_fuse_logit is missing in %s" % metadata_path
                )
            with torch.no_grad():
                model.head.depth_fuse_logit.fill_(float(fuse_logit))
        state_source = str(metadata_path)
    model.eval()
    # Only the head participates in host post-processing. Keeping the
    # backbone and neck on CPU avoids wasting deployment GPU memory.
    model.head.to(device)
    return model, state_source


def visualization_paths(image_path, input_path, output, multiple):
    image_path = Path(image_path)
    input_path = Path(input_path)
    if not multiple:
        camera_path = Path(output or "outputs/onnx_inference.jpg")
        return camera_path, camera_path.with_name(
            camera_path.stem + "_bev" + camera_path.suffix
        )
    output_dir = Path(output or "outputs/onnx_inference")
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            "--output must be a directory for directory inference: %s"
            % output_dir
        )
    relative = image_path.relative_to(input_path)
    camera_path = output_dir / relative.parent / (
        image_path.stem + "_pred.jpg"
    )
    return camera_path, camera_path.with_name(image_path.stem + "_bev.jpg")


def raw_output_path(image_path, input_path, raw_output, multiple):
    if raw_output is None:
        return None
    raw_output = Path(raw_output)
    if not multiple:
        return raw_output
    relative = Path(image_path).relative_to(input_path)
    return raw_output / relative.parent / (relative.stem + ".npz")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_model(
    config,
    model_path,
    input_path,
    calibration_path,
    checkpoint=None,
    extri_path=None,
    output=None,
    visualize=True,
    save_bev=False,
    recursive=False,
    max_images=None,
    raw_output=None,
    summary_path=None,
    providers=None,
    postprocess_device="cuda",
    overwrite=False,
):
    """Run calibrated ONNX inference, official post-processing and visualization."""
    import onnxruntime as ort

    model_path = Path(model_path)
    input_path = Path(input_path)
    calibration_path = Path(calibration_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if save_bev and not visualize:
        raise ValueError("--save-bev requires visualization")
    device = torch.device(postprocess_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Official rotated-NMS post-processing requires an available CUDA device"
        )

    images = collect_images(input_path, recursive, max_images)
    multiple = input_path.is_dir()
    selected_providers = select_providers(
        ort.get_available_providers(), providers
    )
    initialization_start = perf_counter()
    session = ort.InferenceSession(
        str(model_path), providers=selected_providers
    )
    names = [item.name for item in session.get_outputs()]
    postprocess_model, state_source = load_postprocess_model(
        config, model_path, checkpoint, device
    )
    processor = FCOS3DPostProcessor(postprocess_model, config)
    initialization_ms = (perf_counter() - initialization_start) * 1000.0
    print(
        "Initialization: %.2f ms | ORT providers=%s | postprocess_state=%s"
        % (initialization_ms, session.get_providers(), state_source)
    )

    records = []
    for index, image_path in enumerate(images, 1):
        stage_start = perf_counter()
        resolved_extrinsic = resolve_extrinsic_path(
            extri_path, image_path, input_path
        )
        original, tensor, target = prepare_sample(
            image_path,
            calibration_path,
            config,
            extrinsic_path=resolved_extrinsic,
        )
        input_name = validate_input(session, tensor)
        preprocess_ms = (perf_counter() - stage_start) * 1000.0

        stage_start = perf_counter()
        runtime_outputs = session.run(None, {input_name: tensor})
        inference_ms = (perf_counter() - stage_start) * 1000.0
        output_statistics = summarize_outputs(names, runtime_outputs)

        stage_start = perf_counter()
        predictions = reconstruct_outputs(
            names, runtime_outputs, config, device
        )
        result = processor(predictions, [target])[0]
        _synchronize(device)
        postprocess_ms = (perf_counter() - stage_start) * 1000.0

        raw_path = raw_output_path(
            image_path, input_path, raw_output, multiple
        )
        if raw_path is not None:
            raw_path = _require_available_output(raw_path, overwrite)
            np.savez(raw_path, **dict(zip(names, runtime_outputs)))

        camera_path = None
        bev_path = None
        visualization_ms = 0.0
        if visualize:
            stage_start = perf_counter()
            camera_path, bev_path = visualization_paths(
                image_path, input_path, output, multiple
            )
            camera_path = _require_available_output(camera_path, overwrite)
            camera = draw_camera_view(
                original.copy(),
                target,
                result,
                config["data"]["classes"],
                config["evaluation"]["max_per_image"],
                draw_ground_truth=False,
            )
            if not cv2.imwrite(str(camera_path), camera):
                raise OSError("Cannot write visualization: %s" % camera_path)
            if save_bev:
                bev_path = _require_available_output(bev_path, overwrite)
                bev = draw_bev(
                    target,
                    result,
                    config["data"]["classes"],
                    config["evaluation"]["max_per_image"],
                    draw_ground_truth=False,
                )
                if not cv2.imwrite(str(bev_path), bev):
                    raise OSError("Cannot write BEV visualization: %s" % bev_path)
            else:
                bev_path = None
            visualization_ms = (perf_counter() - stage_start) * 1000.0

        total_ms = (
            preprocess_ms + inference_ms + postprocess_ms + visualization_ms
        )
        timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "visualization_ms": visualization_ms if visualize else None,
            "total_ms": total_ms,
        }
        record = {
            "image": str(image_path.resolve()),
            "calibration": str(calibration_path.resolve()),
            "extrinsic": (
                str(resolved_extrinsic.resolve())
                if resolved_extrinsic is not None
                else None
            ),
            "extrinsic_source": (
                "per_frame_txt" if resolved_extrinsic is not None else "calib_json"
            ),
            "detections": int(result["scores"].numel()),
            "visualization": str(camera_path) if camera_path else None,
            "bev": str(bev_path) if bev_path else None,
            "raw_output": str(raw_path) if raw_path else None,
            "timing": timing,
            "outputs": output_statistics,
        }
        records.append(record)
        visualization_text = (
            "%.2f ms" % visualization_ms if visualize else "off"
        )
        print(
            "[%d/%d] %s boxes=%d | preprocess=%.2f ms inference=%.2f ms "
            "postprocess=%.2f ms visualization=%s total=%.2f ms%s"
            % (
                index,
                len(images),
                image_path,
                record["detections"],
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                visualization_text,
                total_ms,
                " -> %s" % camera_path if camera_path else "",
            )
        )

    timing_keys = (
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
    )
    average = {
        key: float(np.mean([record["timing"][key] for record in records]))
        for key in timing_keys
    }
    if visualize:
        average["visualization_ms"] = float(
            np.mean(
                [record["timing"]["visualization_ms"] for record in records]
            )
        )
    summary = {
        "model": str(model_path.resolve()),
        "providers": session.get_providers(),
        "postprocess_state": state_source,
        "initialization_ms": initialization_ms,
        "num_images": len(records),
        "average_timing": average,
        "images": records,
    }
    if summary_path:
        summary_path = _require_available_output(summary_path, overwrite)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "Average over %d images | preprocess=%.2f ms inference=%.2f ms "
        "postprocess=%.2f ms visualization=%s total=%.2f ms"
        % (
            len(records),
            average["preprocess_ms"],
            average["inference_ms"],
            average["postprocess_ms"],
            "%.2f ms" % average["visualization_ms"]
            if visualize
            else "off",
            average["total_ms"],
        )
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run calibrated FCOS3D ONNX inference and visualization"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint for learned host post-process state; defaults to ONNX sibling .pth",
    )
    parser.add_argument(
        "--image", required=True, help="One image or a directory of images"
    )
    parser.add_argument(
        "--calib-path", "--calib", required=True, dest="calib_path",
        help="Required FrontViewCalibParam JSON shared by input images",
    )
    parser.add_argument(
        "--extri-path", "--extrinsic", dest="extri_path",
        help="Optional shared TXT or directory of per-frame <image-stem>.txt files",
    )
    parser.add_argument(
        "--output",
        help="Visualization image for one input, or directory for folder input",
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Disable visualization (enabled by default)",
    )
    parser.add_argument(
        "--save-bev", action="store_true",
        help="Save an additional bird's-eye-view image",
    )
    parser.add_argument(
        "--raw-output",
        help="Optional .npz for one image, or raw-output directory for folder input",
    )
    parser.add_argument("--summary", help="Optional timing/result JSON")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--provider", action="append", dest="providers",
        help="ONNX Runtime provider; repeat to set fallback order",
    )
    parser.add_argument(
        "--postprocess-device", default="cuda",
        help="Torch device for official post-processing (must be CUDA)",
    )
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--nms-pre", type=int)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--nms-pairwise-chunk-size", type=int)
    parser.add_argument("--max-per-image", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides"
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    evaluation_overrides = {
        "score_threshold": args.score_threshold,
        "nms_pre": args.nms_pre,
        "nms_threshold": args.nms_threshold,
        "nms_pairwise_chunk_size": args.nms_pairwise_chunk_size,
        "max_per_image": args.max_per_image,
    }
    for key, value in evaluation_overrides.items():
        if value is not None:
            config["evaluation"][key] = value
    if not 0.0 <= config["evaluation"]["score_threshold"] <= 1.0:
        raise ValueError("score threshold must be in [0, 1]")
    if not 0.0 <= config["evaluation"]["nms_threshold"] <= 1.0:
        raise ValueError("NMS threshold must be in [0, 1]")
    if config["evaluation"]["nms_pre"] <= 0:
        raise ValueError("nms-pre must be positive")
    if config["evaluation"]["max_per_image"] <= 0:
        raise ValueError("max-per-image must be positive")
    if config["evaluation"].get("nms_pairwise_chunk_size", 32) <= 0:
        raise ValueError("nms-pairwise-chunk-size must be positive")

    infer_model(
        config,
        args.model,
        args.image,
        args.calib_path,
        checkpoint=args.checkpoint,
        extri_path=args.extri_path,
        output=args.output,
        visualize=not args.no_visualize,
        save_bev=args.save_bev,
        recursive=args.recursive,
        max_images=args.max_images,
        raw_output=args.raw_output,
        summary_path=args.summary,
        providers=args.providers,
        postprocess_device=args.postprocess_device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
