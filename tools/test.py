#!/usr/bin/env python3
import argparse
import json

from project_detection.config import load_config
from project_detection.engine import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Final test evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(load_config(args.config, args.overrides), args.checkpoint, "test"), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
