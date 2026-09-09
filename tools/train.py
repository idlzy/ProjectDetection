#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from project_detection.config import load_config
from project_detection.engine import train
from project_detection.logging_utils import LOGGER_NAME, set_process_name


def main():
    set_process_name("DetectionTrain")
    parser = argparse.ArgumentParser(description="Train FCOS3D/PGDA")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", nargs="+", action="append", default=[], dest="override_groups"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    overrides = [item for group in args.override_groups for item in group]
    config = load_config(args.config, overrides)
    if args.auto_resume:
        checkpoint_dir = (
            Path(config["experiment"]["output_dir"])
            / config["experiment"]["name"]
            / "checkpoints"
        )
        candidates = [
            path for path in (
                checkpoint_dir / "recovery.pth",
                checkpoint_dir / "last.pth",
            ) if path.is_file()
        ]
        if candidates:
            config["train"]["resume"] = str(max(candidates, key=lambda path: path.stat().st_mtime))
            config["train"]["pretrain"] = None
    if args.validate_only:
        pretrain = config["train"].get("pretrain")
        if pretrain and not Path(pretrain).is_file():
            raise FileNotFoundError("Missing pretrained checkpoint: %s" % pretrain)
        ready_root = Path(
            config["data"].get("ready_root") or config["data"]["data_root"]
        )
        for split in ("train", "val"):
            manifest = ready_root / "splits" / (split + "_frames.jsonl")
            if not manifest.is_file():
                raise FileNotFoundError("Missing split manifest: %s" % manifest)
        print("Training configuration and checkpoint paths are valid")
        return
    try:
        train(config)
    except BaseException:
        logging.getLogger(LOGGER_NAME).exception("Training terminated with an error")
        raise


if __name__ == "__main__": main()
