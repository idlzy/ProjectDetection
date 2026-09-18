#!/usr/bin/env python3
"""Render reproducible ONNX predictions sampled from every test ann_batch."""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties

from project_detection.config import load_config
from project_detection.engine import load_checkpoint
from project_detection.models import build_model
from project_detection.task import FCOS3DPostProcessor
from project_detection.visualization import draw_camera_view
if __package__:
    from tools.export_and_infer_onnx import (
        _synchronize,
        load_postprocess_model,
        prepare_sample,
        reconstruct_outputs,
        select_providers,
        validate_input,
    )
else:
    from export_and_infer_onnx import (
        _synchronize,
        load_postprocess_model,
        prepare_sample,
        reconstruct_outputs,
        select_providers,
        validate_input,
    )


CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def record_ann_batch(record):
    """Return a useful batch name, including legacy ONCE manifest records."""
    ann_batch = record.get("ann_batch")
    if ann_batch not in (None, ""):
        return str(ann_batch)
    split_group = str(record.get("split_group") or "")
    if split_group.startswith("ONCE_cam"):
        return split_group.split("__", 1)[0]
    return "<missing ann_batch>"


def read_test_groups(data_root):
    manifest = Path(data_root) / "splits" / "test_frames.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError("Missing test manifest: %s" % manifest)
    groups = defaultdict(list)
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at %s:%d" % (manifest, line_number)
                ) from error
            missing = [
                key for key in ("image_rel", "calib_rel")
                if not record.get(key)
            ]
            if missing:
                raise ValueError(
                    "Invalid test record at %s:%d; missing: %s"
                    % (manifest, line_number, ", ".join(missing))
                )
            groups[record_ann_batch(record)].append(record)
    if not groups:
        raise ValueError("Test manifest is empty: %s" % manifest)
    return dict(groups)


def select_records(groups, samples_per_batch=4, seed=20260918):
    if samples_per_batch <= 0:
        raise ValueError("--samples-per-batch must be positive")
    selected = {}
    random_generator = random.Random(seed)
    for ann_batch in sorted(groups):
        records = groups[ann_batch]
        if len(records) < samples_per_batch:
            raise ValueError(
                "ann_batch %r has only %d records; %d requested"
                % (ann_batch, len(records), samples_per_batch)
            )
        selected[ann_batch] = random_generator.sample(
            records, samples_per_batch
        )
    return selected


def _font_properties(font_path=None):
    candidates = ([font_path] if font_path else []) + list(
        CJK_FONT_CANDIDATES
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return FontProperties(fname=str(candidate))
    raise FileNotFoundError(
        "No CJK font found; pass --font-path with a Chinese-capable TTF/TTC"
    )


def _resize_tile(image, tile_width):
    height, width = image.shape[:2]
    tile_height = max(1, int(round(height * tile_width / width)))
    interpolation = cv2.INTER_AREA if width > tile_width else cv2.INTER_LINEAR
    return cv2.resize(image, (tile_width, tile_height), interpolation=interpolation)


def _wrapped_title(value, width=24):
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def model_format(model_path):
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pth", ".pt"}:
        return "pytorch"
    raise ValueError("--model must end in .onnx, .pth, or .pt")


def collage_output_paths(output, ann_batch_count, columns):
    if columns <= 0:
        raise ValueError("--columns must be positive")
    if ann_batch_count <= 0:
        raise ValueError("ann_batch_count must be positive")
    output = Path(output)
    page_count = int(math.ceil(ann_batch_count / float(columns)))
    if page_count == 1:
        return [output]
    return [
        output.with_name(
            "%s_page_%02d%s" % (output.stem, page_index, output.suffix)
        )
        for page_index in range(1, page_count + 1)
    ]


def save_collage(
    rendered_by_batch,
    output,
    columns=5,
    tile_width=600,
    font_path=None,
    overwrite=False,
):
    """Save one page: one ann_batch per column and one sample per row."""
    if columns <= 0:
        raise ValueError("--columns must be positive")
    if tile_width <= 0:
        raise ValueError("--tile-width must be positive")
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(
            "%s already exists; pass --overwrite to replace it" % output
        )
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("--output must end in .jpg, .jpeg, or .png")
    output.parent.mkdir(parents=True, exist_ok=True)

    ann_batches = list(rendered_by_batch)
    if not ann_batches:
        raise ValueError("No rendered batches to compose")
    samples_per_batch = len(rendered_by_batch[ann_batches[0]])
    if not samples_per_batch:
        raise ValueError("Rendered batches must contain images")
    for name, items in rendered_by_batch.items():
        if len(items) != samples_per_batch:
            raise ValueError("Batch %r has an inconsistent sample count" % name)

    if len(ann_batches) > columns:
        raise ValueError(
            "One collage page accepts at most %d ann_batches" % columns
        )
    tile_height = int(round(tile_width * 9.0 / 16.0))
    figure_width = columns * tile_width / 100.0
    figure_height = (150 + samples_per_batch * tile_height) / 100.0
    figure = plt.figure(
        figsize=(figure_width, figure_height), dpi=100, facecolor="#111820"
    )
    ratios = [150.0 / tile_height] + [1.0] * samples_per_batch
    grid = figure.add_gridspec(
        samples_per_batch + 1,
        columns,
        height_ratios=ratios,
        left=0.006,
        right=0.994,
        bottom=0.004,
        top=0.996,
        wspace=0.018,
        hspace=0.035,
    )
    font = _font_properties(font_path)

    for column in range(columns):
        header_axis = figure.add_subplot(grid[0, column])
        header_axis.set_facecolor("#1D2A38")
        header_axis.set_xticks([])
        header_axis.set_yticks([])
        for spine in header_axis.spines.values():
            spine.set_visible(False)
        if column >= len(ann_batches):
            header_axis.axis("off")
            for sample_index in range(samples_per_batch):
                axis = figure.add_subplot(
                    grid[sample_index + 1, column]
                )
                axis.axis("off")
            continue

        ann_batch = ann_batches[column]
        header_axis.text(
            0.5,
            0.5,
            _wrapped_title(ann_batch),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontproperties=font,
            linespacing=1.15,
        )
        for sample_index, item in enumerate(rendered_by_batch[ann_batch]):
            image = _resize_tile(item["image"], tile_width)
            axis = figure.add_subplot(grid[sample_index + 1, column])
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.axis("off")
            axis.text(
                0.012,
                0.025,
                "%s  |  %d boxes"
                % (item["stem"], item["detections"]),
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.58,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )

    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(figure)
    if not cv2.imwrite(str(output), bgr):
        raise OSError("Cannot write collage: %s" % output)
    return output


def _configure_evaluation(config, args):
    overrides = {
        "score_threshold": args.score_threshold,
        "nms_pre": args.nms_pre,
        "nms_threshold": args.nms_threshold,
        "nms_pairwise_chunk_size": args.nms_pairwise_chunk_size,
        "max_per_image": args.max_per_image,
    }
    for key, value in overrides.items():
        if value is not None:
            config["evaluation"][key] = value
    if not 0.0 <= config["evaluation"]["score_threshold"] <= 1.0:
        raise ValueError("score threshold must be in [0, 1]")
    if not 0.0 <= config["evaluation"]["nms_threshold"] <= 1.0:
        raise ValueError("NMS threshold must be in [0, 1]")
    positive_defaults = {
        "nms_pre": 1000,
        "nms_pairwise_chunk_size": 32,
        "max_per_image": 100,
    }
    for key, default in positive_defaults.items():
        if config["evaluation"].get(key, default) <= 0:
            raise ValueError("%s must be positive" % key.replace("_", "-"))


def render_test_batches(args):
    data_root = Path(args.data_root)
    model_path = Path(args.model)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(
        ".json"
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    backend = model_format(model_path)
    groups = read_test_groups(data_root)
    selected = select_records(groups, args.samples_per_batch, args.seed)
    output_paths = collage_output_paths(
        output_path, len(selected), args.columns
    )
    for path in output_paths + [summary_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                "%s already exists; pass --overwrite to replace it" % path
            )
    config = load_config(args.config, args.overrides)
    _configure_evaluation(config, args)

    device = torch.device(args.postprocess_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Official rotated-NMS post-processing requires an available CUDA device"
        )
    session = None
    output_names = None
    pytorch_model = None
    if backend == "onnx":
        import onnxruntime as ort

        providers = select_providers(
            ort.get_available_providers(), args.providers
        )
        session = ort.InferenceSession(str(model_path), providers=providers)
        output_names = [item.name for item in session.get_outputs()]
        postprocess_model, state_source = load_postprocess_model(
            config, model_path, args.checkpoint, device
        )
        processor = FCOS3DPostProcessor(postprocess_model, config)
        runtime_providers = session.get_providers()
    else:
        pytorch_model = build_model(config).to(device)
        load_checkpoint(model_path, pytorch_model, strict=True)
        pytorch_model.eval()
        processor = FCOS3DPostProcessor(pytorch_model, config)
        state_source = str(model_path.resolve())
        runtime_providers = None

    summary_records = []
    total = sum(len(records) for records in selected.values())
    completed = 0
    selected_items = list(selected.items())
    for page_index, output in enumerate(output_paths):
        page_items = selected_items[
            page_index * args.columns:(page_index + 1) * args.columns
        ]
        rendered = {}
        for ann_batch, records in page_items:
            rendered[ann_batch] = []
            for record in records:
                completed += 1
                image_path = data_root / record["image_rel"]
                calibration_path = data_root / record["calib_rel"]
                extrinsic_rel = record.get("extrinsic_rel")
                extrinsic_path = (
                    data_root / extrinsic_rel if extrinsic_rel else None
                )
                started = perf_counter()
                original, tensor, target = prepare_sample(
                    image_path,
                    calibration_path,
                    config,
                    extrinsic_path=extrinsic_path,
                )
                with torch.no_grad():
                    if backend == "onnx":
                        input_name = validate_input(session, tensor)
                        runtime_outputs = session.run(
                            None, {input_name: tensor}
                        )
                        predictions = reconstruct_outputs(
                            output_names, runtime_outputs, config, device
                        )
                    else:
                        model_input = torch.from_numpy(tensor).to(device)
                        predictions = pytorch_model(model_input)
                    result = processor(predictions, [target])[0]
                _synchronize(device)
                camera = draw_camera_view(
                    original.copy(),
                    target,
                    result,
                    config["data"]["classes"],
                    config["evaluation"]["max_per_image"],
                    draw_ground_truth=False,
                )
                elapsed_ms = (perf_counter() - started) * 1000.0
                item = {
                    "image": camera,
                    "stem": str(record.get("stem") or image_path.stem),
                    "detections": int(result["scores"].numel()),
                }
                rendered[ann_batch].append(item)
                summary_records.append(
                    {
                        "ann_batch": ann_batch,
                        "sample_token": record.get("sample_token"),
                        "image": str(image_path.resolve()),
                        "calibration": str(calibration_path.resolve()),
                        "extrinsic": (
                            str(extrinsic_path.resolve())
                            if extrinsic_path else None
                        ),
                        "detections": item["detections"],
                        "elapsed_ms": elapsed_ms,
                    }
                )
                print(
                    "[%d/%d] %s | %s | boxes=%d | %.2f ms"
                    % (
                        completed,
                        total,
                        ann_batch,
                        item["stem"],
                        item["detections"],
                        elapsed_ms,
                    )
                )
        save_collage(
            rendered,
            output,
            columns=args.columns,
            tile_width=args.tile_width,
            font_path=args.font_path,
            overwrite=args.overwrite,
        )
        print(
            "Saved page %d/%d: %s"
            % (page_index + 1, len(output_paths), output)
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": str(model_path.resolve()),
        "backend": backend,
        "data_root": str(data_root.resolve()),
        "split": "test",
        "seed": args.seed,
        "samples_per_batch": args.samples_per_batch,
        "columns": args.columns,
        "ann_batch_count": len(selected),
        "image_count": total,
        "ann_batches": list(selected),
        "providers": runtime_providers,
        "postprocess_state": state_source,
        "outputs": [str(path.resolve()) for path in output_paths],
        "images": summary_records,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Saved %d ann_batches / %d predictions in %d pages"
        % (len(selected), total, len(output_paths))
    )
    print("Selection summary: %s" % summary_path)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample and render ONNX/PyTorch predictions from every "
            "test ann_batch into paginated multi-column collages"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", required=True, help="Raw-head .onnx or checkpoint .pth/.pt"
    )
    parser.add_argument(
        "--checkpoint", help="Optional host post-process .pth for ONNX only"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output",
        default="outputs/test_ann_batch_predictions.png",
        help="Output base name; multiple pages add _page_XX",
    )
    parser.add_argument("--summary", help="Selection/result JSON path")
    parser.add_argument("--samples-per-batch", type=int, default=4)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--font-path", help="Chinese-capable TTF/TTC font")
    parser.add_argument(
        "--provider", action="append", dest="providers",
        help="ONNX Runtime provider; repeat to set fallback order",
    )
    parser.add_argument("--postprocess-device", default="cuda")
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
    render_test_batches(build_parser().parse_args())


if __name__ == "__main__":
    main()
