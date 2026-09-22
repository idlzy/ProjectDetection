#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from project_detection.config import load_config
from project_detection.engine import evaluate_checkpoint
from project_detection.reporting import write_test_report


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Dataset split evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", help="Write the complete test report JSON here")
    parser.add_argument("--plot-dir", help="Write metric JSON/PNG plots here")
    parser.add_argument(
        "--head-level-stats", action="store_true",
        help="Report P3-P7 contribution metrics and confusion matrices",
    )
    parser.add_argument(
        "--cop-stats", action="store_true",
        help="Report adaptive CoP branch-selection statistics",
    )
    parser.add_argument(
        "--cop-ablation", action="store_true",
        help="Compare adaptive, parallel and chain branches from one forward pass",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    config["evaluation"]["head_level_stats"] = args.head_level_stats
    config["evaluation"]["cop_stats"] = args.cop_stats
    config["evaluation"]["cop_ablation"] = args.cop_ablation
    metrics = evaluate_checkpoint(config, args.checkpoint, args.split)
    if args.output or args.plot_dir:
        experiment_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
        output = Path(args.output) if args.output else experiment_dir / "metrics" / (args.split + "_report.json")
        plot_dir = Path(args.plot_dir) if args.plot_dir else output.parent / (output.stem + "_plots")
        checkpoint_path = Path(args.checkpoint).resolve()
        ready_root = Path(config["data"].get("ready_root") or config["data"]["data_root"]).resolve()
        split_manifest = ready_root / "splits" / (args.split + "_frames.jsonl")
        metadata = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "data_root": str(Path(config["data"]["data_root"]).resolve()),
            "ready_root": str(ready_root),
            "split": args.split,
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
            "score_threshold": config["evaluation"]["score_threshold"],
            "distance_thresholds": config["evaluation"]["distance_thresholds"],
            "head_level_stats": args.head_level_stats,
            "cop_stats": args.cop_stats,
            "cop_ablation": args.cop_ablation,
        }
        if args.split == "test":
            metadata["test_manifest"] = metadata["split_manifest"]
            metadata["test_manifest_sha256"] = metadata["split_manifest_sha256"]
        report = write_test_report(metrics, output, plot_dir, metadata)
        print(json.dumps({
            "protocol": report["protocol"],
            "metric_json": str(output),
            "plot_dir": str(plot_dir),
            "summary": {
                key: metrics.get(key) for key in (
                    "NDS", "mAP", "mATE", "mASE", "mAOE", "mADE",
                    "mRecall", "mPrecision", "F1", "num_samples", "num_gt",
                    "num_predictions", "num_tp",
                )
            },
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
