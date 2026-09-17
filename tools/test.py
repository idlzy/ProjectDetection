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
    parser = argparse.ArgumentParser(description="Final test evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Write the complete test report JSON here")
    parser.add_argument("--plot-dir", help="Write metric JSON/PNG plots here")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    metrics = evaluate_checkpoint(config, args.checkpoint, "test")
    if args.output or args.plot_dir:
        experiment_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
        output = Path(args.output) if args.output else experiment_dir / "metrics" / "test_report.json"
        plot_dir = Path(args.plot_dir) if args.plot_dir else output.parent / (output.stem + "_plots")
        checkpoint_path = Path(args.checkpoint).resolve()
        ready_root = Path(config["data"].get("ready_root") or config["data"]["data_root"]).resolve()
        test_manifest = ready_root / "splits" / "test_frames.jsonl"
        report = write_test_report(
            metrics,
            output,
            plot_dir,
            {
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "data_root": str(Path(config["data"]["data_root"]).resolve()),
                "ready_root": str(ready_root),
                "test_manifest": str(test_manifest),
                "test_manifest_sha256": sha256_file(test_manifest),
                "score_threshold": config["evaluation"]["score_threshold"],
                "distance_thresholds": config["evaluation"]["distance_thresholds"],
            },
        )
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
