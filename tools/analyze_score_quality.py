#!/usr/bin/env python3
"""Diagnose whether post-NMS detection scores rank accurate MW3D boxes higher.

Matching mirrors Mw3dMetric._accumulate_class: per class, descending score,
nearest unmatched same-class GT in camera BEV [x, z], strict distance cutoff.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from project_detection.config import load_config
from project_detection.engine import build_loader, load_checkpoint
from project_detection.models import build_model
from project_detection.task import FCOS3DPostProcessor


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_bin(depth, intervals):
    if depth is None:
        return "no_same_class_gt"
    for lower, upper in intervals:
        if float(lower) <= depth < float(upper):
            return "%g-%gm" % (lower, upper) if upper < 1000000 else "%gm+" % lower
    return "outside_bins"


def _match_class_at_threshold(boxes, scores, pred_indices, gt_boxes, gt_indices, threshold):
    """Mirror Mw3dMetric's score-ordered nearest-unmatched matching."""
    matches = {}
    matched_gt = set()
    for index in sorted(pred_indices, key=lambda item: (-float(scores[item]), int(item))):
        candidates = [int(item) for item in gt_indices if int(item) not in matched_gt]
        if not candidates:
            continue
        distances = np.linalg.norm(
            gt_boxes[candidates][:, [0, 2]] - boxes[index, [0, 2]], axis=1
        )
        position = int(np.argmin(distances))
        if float(distances[position]) < threshold:
            matches[int(index)] = candidates[position]
            matched_gt.add(candidates[position])
    return matches


def match_frame(prediction, target, classes, threshold, intervals,
                distance_thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Return one record per evaluated prediction; unmatched duplicates are FP."""
    boxes = prediction["boxes3d"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    gt_boxes = target["boxes3d"].detach().cpu().numpy()
    gt_labels = target["labels"].detach().cpu().numpy()
    rows = []
    for class_id, class_name in enumerate(classes):
        pred_indices = np.where(labels == class_id)[0]
        gt_indices = np.where(gt_labels == class_id)[0]
        primary_matches = _match_class_at_threshold(
            boxes, scores, pred_indices, gt_boxes, gt_indices, threshold
        )
        matches_by_distance = {
            "%g" % value: _match_class_at_threshold(
                boxes, scores, pred_indices, gt_boxes, gt_indices, float(value)
            ) for value in distance_thresholds
        }
        for index in sorted(pred_indices, key=lambda item: (-float(scores[item]), int(item))):
            box = boxes[index]
            nearest_index = None
            nearest_distance = None
            if len(gt_indices):
                distances = np.linalg.norm(
                    gt_boxes[gt_indices][:, [0, 2]] - box[[0, 2]], axis=1
                )
                nearest_position = int(np.argmin(distances))
                nearest_index = int(gt_indices[nearest_position])
                nearest_distance = float(distances[nearest_position])
            match_index = primary_matches.get(int(index))
            matched_box = gt_boxes[match_index] if match_index is not None else None
            reference_box = matched_box if matched_box is not None else (
                gt_boxes[nearest_index] if nearest_index is not None else None
            )
            reference_depth = float(reference_box[2]) if reference_box is not None else None
            rows.append({
                "image_id": str(target["sample_token"]),
                "image_path": str(target["image_path"]),
                "class": class_name,
                "prediction_index": int(index),
                "score": float(scores[index]),
                "pred_x": float(box[0]),
                "pred_y": float(box[1]),
                "pred_z": float(box[2]),
                "matched": match_index is not None,
                "matched_gt_index": match_index,
                "gt_x": float(matched_box[0]) if matched_box is not None else None,
                "gt_y": float(matched_box[1]) if matched_box is not None else None,
                "gt_z": float(matched_box[2]) if matched_box is not None else None,
                "bev_center_error_m": float(np.linalg.norm(box[[0, 2]] - matched_box[[0, 2]]))
                if matched_box is not None else None,
                "center_3d_error_m": float(np.linalg.norm(box[:3] - matched_box[:3]))
                if matched_box is not None else None,
                "depth_abs_error_m": abs(float(box[2] - matched_box[2]))
                if matched_box is not None else None,
                "nearest_same_class_gt_bev_m": nearest_distance,
                "nearest_gt_depth_m": reference_depth,
                "nearest_gt_depth_bin": depth_bin(reference_depth, intervals),
                "matched_gt_depth_bin": depth_bin(float(matched_box[2]), intervals)
                if matched_box is not None else None,
                "hits_by_distance_m": {
                    key: int(index) in matches for key, matches in matches_by_distance.items()
                },
            })
    return rows


def _ranks(values):
    """Average ranks for ties, matching the usual Spearman definition."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    for indices in np.split(order, np.where(np.diff(values[order]) != 0)[0] + 1):
        ranks[indices] = start + (len(indices) - 1) / 2.0
        start += len(indices)
    return ranks


def spearman_matched(rows, error_key="bev_center_error_m"):
    pairs = [(row["score"], row[error_key]) for row in rows if row["matched"]]
    if len(pairs) < 2:
        return {"n": len(pairs), "rho": None}
    score_rank = _ranks([item[0] for item in pairs])
    error_rank = _ranks([item[1] for item in pairs])
    if np.std(score_rank) == 0 or np.std(error_rank) == 0:
        return {"n": len(pairs), "rho": None}
    return {"n": len(pairs), "rho": float(np.corrcoef(score_rank, error_rank)[0, 1])}


def hit_rate(rows, distance_thresholds=(0.5, 1.0, 2.0, 4.0)):
    count = len(rows)
    hits = sum(bool(row["matched"]) for row in rows)
    result = {
        "predictions": count,
        "hits_2m": hits,
        "hit_rate": hits / count if count else None,
        "false_positive_rate": (count - hits) / count if count else None,
    }
    result["distance_hit_rates"] = {}
    for threshold in distance_thresholds:
        key = "%g" % threshold
        threshold_hits = sum(row["hits_by_distance_m"][key] for row in rows)
        result["distance_hit_rates"][key] = {
            "hits": threshold_hits,
            "hit_rate": threshold_hits / count if count else None,
        }
    return result


def summarize(rows, bins=10, distance_thresholds=(0.5, 1.0, 2.0, 4.0)):
    ordered = sorted(rows, key=lambda row: -row["score"])
    by_class = sorted(set(row["class"] for row in rows))
    by_depth = sorted(set(row["nearest_gt_depth_bin"] for row in rows))
    matched_depths = sorted(set(
        row["matched_gt_depth_bin"] for row in rows if row["matched"]
    ))
    deciles = []
    for bucket in range(bins):
        subset = ordered[bucket * len(ordered) // bins:(bucket + 1) * len(ordered) // bins]
        if subset:
            deciles.append({
                "score_quantile_high_to_low": bucket + 1,
                "score_min": min(row["score"] for row in subset),
                "score_max": max(row["score"] for row in subset),
                **hit_rate(subset, distance_thresholds),
            })
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)
    top_k = {}
    for k in (20, 50, 100):
        selected = []
        for image_rows in by_image.values():
            selected.extend(sorted(image_rows, key=lambda row: -row["score"])[:k])
        top_k[str(k)] = {"selection": "per_image_then_aggregate", "images_with_predictions": len(by_image),
                         **hit_rate(selected, distance_thresholds)}
    return {
        "all": {**hit_rate(rows, distance_thresholds), "matched_spearman_score_vs_bev_error": spearman_matched(rows),
                "matched_spearman_score_vs_3d_error": spearman_matched(rows, "center_3d_error_m")},
        "score_quantiles": deciles,
        "top_k": top_k,
        "by_class": {name: {
            **hit_rate([row for row in rows if row["class"] == name], distance_thresholds),
            "matched_spearman_score_vs_bev_error": spearman_matched(
                [row for row in rows if row["class"] == name]
            ),
        } for name in by_class},
        "by_nearest_gt_depth_bin": {name: hit_rate([
            row for row in rows if row["nearest_gt_depth_bin"] == name
        ], distance_thresholds) for name in by_depth},
        "matched_only_by_gt_depth_bin": {name: {
            "matched_predictions": sum(
                row["matched_gt_depth_bin"] == name for row in rows
            ),
            "spearman_score_vs_bev_error": spearman_matched([
                row for row in rows if row["matched_gt_depth_bin"] == name
            ]),
        } for name in matched_depths},
    }


def run_checkpoint(config, checkpoint, loader, device, output_dir, split, manifest_hash):
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, strict=True)
    model.eval()
    processor = FCOS3DPostProcessor(model, config)
    evaluation = config["evaluation"]
    classes = config["data"]["classes"]
    evaluated = set(evaluation.get("classes") or classes)
    intervals = evaluation.get("depth_bins") or ((0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 1000000))
    threshold = float(evaluation.get("tp_distance_threshold", 2.0))
    rows = []
    with torch.no_grad():
        for images, targets in loader:
            predictions = processor(model(images.to(device)), targets)
            for prediction, target in zip(predictions, targets):
                rows.extend(row for row in match_frame(
                    prediction, target, classes, threshold, intervals,
                    evaluation["distance_thresholds"],
                ) if row["class"] in evaluated)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else [
            "image_id", "image_path", "class", "prediction_index", "score",
            "pred_x", "pred_y", "pred_z", "matched", "matched_gt_index",
            "gt_x", "gt_y", "gt_z", "bev_center_error_m", "center_3d_error_m",
            "depth_abs_error_m", "nearest_same_class_gt_bev_m",
            "nearest_gt_depth_m", "nearest_gt_depth_bin", "matched_gt_depth_bin",
            "hits_by_distance_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["hits_by_distance_m"] = json.dumps(row["hits_by_distance_m"])
            writer.writerow(csv_row)
    report = {
        "protocol": "score_ordered_nearest_unmatched_same_class_bev_strict_threshold",
        "note": "Scores are ranking signals, not calibrated probabilities. Spearman uses matched predictions only; quantiles and Top-K include false positives. Unmatched depth bins use nearest same-class GT as a proxy.",
        "split": split,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": manifest_hash,
        "evaluation_config": {key: evaluation.get(key) for key in (
            "classes", "score_threshold", "nms_pre", "nms_threshold",
            "nms_pairwise_chunk_size", "max_per_image", "distance_thresholds",
            "tp_distance_threshold", "depth_bins"
        )},
        "num_frames": len(loader.dataset),
        "summary": summarize(rows, distance_thresholds=evaluation["distance_thresholds"]),
        "predictions_csv": str(csv_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Repeat for a same-config, same-manifest comparison")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    config = load_config(args.config, args.overrides)
    ready_root = Path(config["data"].get("ready_root") or config["data"]["data_root"])
    manifest = ready_root / "splits" / (args.split + "_frames.jsonl")
    manifest_hash = sha256_file(manifest)
    device = torch.device(
        "cuda" if config["runtime"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    loader = build_loader(config, args.split, max_samples=args.max_samples)
    output_dir = Path(args.output_dir)
    reports = []
    for index, checkpoint in enumerate(args.checkpoint):
        name = "%02d_%s" % (index + 1, Path(checkpoint).stem)
        report = run_checkpoint(
            config, checkpoint, loader, device, output_dir / name,
            args.split, manifest_hash,
        )
        reports.append({
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "summary_json": str((output_dir / name / "summary.json").resolve()),
            "summary": report["summary"]["all"],
        })
        print("%s: %s" % (name, reports[-1]["summary_json"]))
    comparison = {
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "config_overrides": args.overrides,
        "split": args.split,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "max_samples": args.max_samples,
        "device": str(device),
        "evaluation_config": {key: config["evaluation"].get(key) for key in (
            "classes", "score_threshold", "nms_pre", "nms_threshold",
            "nms_pairwise_chunk_size", "max_per_image", "distance_thresholds",
            "tp_distance_threshold", "depth_bins"
        )},
        "data_classes": config["data"]["classes"],
        "checkpoints": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
