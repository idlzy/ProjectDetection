#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from project_detection.config import load_config
from project_detection.engine import evaluate_checkpoint
from project_detection.reporting import write_test_report


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
        report = write_test_report(
            metrics,
            output,
            plot_dir,
            {
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "score_threshold": config["evaluation"]["score_threshold"],
                "distance_thresholds": config["evaluation"]["distance_thresholds"],
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
