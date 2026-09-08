#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from project_detection.config import load_config


def read_records(root, split):
    with (root / "splits" / (split + "_frames.jsonl")).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Inspect manifests and detect split leakage")
    parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = Path(config["data"]["ready_root"])
    records = {split: read_records(root, split) for split in ("train", "val", "test")}
    for split, items in records.items():
        batches = Counter(item.get("ann_batch", item.get("split_group", "unknown")) for item in items)
        print("%s: frames=%d ann_batches=%d" % (split, len(items), len(batches)))
    failed = False
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("sample_token", "split_group"):
            overlap = {item.get(key) for item in records[left]} & {item.get(key) for item in records[right]}
            overlap.discard(None)
            print("%s/%s %s overlap=%d" % (left, right, key, len(overlap)))
            failed |= bool(overlap)
    if failed: raise SystemExit("Dataset leakage detected")


if __name__ == "__main__": main()
