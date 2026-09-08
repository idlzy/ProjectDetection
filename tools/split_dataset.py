#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_detection.data.split import (
    assert_no_leakage,
    read_jsonl,
    split_records_by_group,
    summarize_splits,
    summarize_strata,
    validate_records,
    write_splits,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe MW3D train/val/test manifests"
    )
    parser.add_argument(
        "--data-root",
        default="/home/gy-zb-a-luziyang/datasets/mw3d",
        help="Root containing frames.jsonl, images/ and annotation/",
    )
    parser.add_argument("--manifest", default="frames.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-key", default="split_group")
    parser.add_argument(
        "--stratify-key",
        default="image_source",
        help="image_source uses the images/<source>/... directory; a record field is also accepted",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = data_root / manifest
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "splits"
    )

    records = read_jsonl(manifest)
    validate_records(records, data_root, check_files=not args.skip_file_check)
    splits = split_records_by_group(
        records,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        group_key=args.group_key,
        stratify_key=args.stratify_key,
    )
    assert_no_leakage(splits, args.group_key)
    summary = summarize_splits(splits, args.group_key)
    strata_summary = summarize_strata(splits, args.group_key, args.stratify_key)
    print(json.dumps({"total": summary, "strata": strata_summary}, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run: no files were written")
        return
    write_splits(splits, output_dir, args.force)
    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_manifest": str(manifest),
                "seed": args.seed,
                "group_key": args.group_key,
                "stratify_key": args.stratify_key,
                "ratios": {
                    "train": args.train_ratio,
                    "val": args.val_ratio,
                    "test": args.test_ratio,
                },
                "splits": summary,
                "strata": strata_summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("Wrote split manifests to %s" % output_dir)


if __name__ == "__main__":
    main()
