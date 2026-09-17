#!/usr/bin/env python3
"""Set visibility=0 for MW3D objects beyond a BEV distance threshold."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_THRESHOLD_METRES = 80.0
DEFAULT_EXAMPLE_LIMIT = 20
SPLITS = ("train", "val", "test")


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at %s:%d" % (path, line_number)
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    "Manifest record must be an object at %s:%d"
                    % (path, line_number)
                )
            records.append(record)
    return records


def _default_manifests(data_root):
    complete = data_root / "frames.jsonl"
    if complete.is_file():
        return [complete]
    split_manifests = [
        data_root / "splits" / ("%s_frames.jsonl" % split)
        for split in SPLITS
    ]
    existing = [path for path in split_manifests if path.is_file()]
    if existing:
        return existing
    raise FileNotFoundError(
        "No frames.jsonl or splits/<split>_frames.jsonl found under %s"
        % data_root
    )


def collect_annotation_paths(data_root, manifests=None):
    manifest_paths = list(manifests or _default_manifests(data_root))
    annotations = {}
    invalid_records = []
    for manifest in manifest_paths:
        for index, record in enumerate(_read_jsonl(manifest), 1):
            relative = record.get("ann_rel") or record.get("anno_rel")
            if not relative:
                invalid_records.append("%s:%d" % (manifest, index))
                continue
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "Annotation path must stay inside data-root at %s:%d: %s"
                    % (manifest, index, relative)
                )
            path = data_root / relative
            # Deduplicate records repeated between frames.jsonl and split files.
            annotations[str(path.resolve())] = (path, Path(relative))
    if invalid_records:
        raise ValueError(
            "%d manifest records have no ann_rel/anno_rel; first: %s"
            % (len(invalid_records), invalid_records[0])
        )
    missing = [str(path) for path, _ in annotations.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "%d annotation files are missing; first: %s"
            % (len(missing), missing[0])
        )
    return manifest_paths, [annotations[key] for key in sorted(annotations)]


def _load_annotation(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        objects = payload
    elif isinstance(payload, dict) and isinstance(payload.get("objects"), list):
        objects = payload["objects"]
    else:
        raise ValueError(
            "Annotation must be an object list or contain an objects list: %s"
            % path
        )
    if not all(isinstance(obj, dict) for obj in objects):
        raise ValueError("Annotation contains a non-object entry: %s" % path)
    return payload, objects


def _bev_distance(obj):
    center = obj.get("center")
    if not isinstance(center, dict):
        return None
    try:
        x = float(center["x"])
        y = float(center["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return math.hypot(x, y)


def _atomic_write_json(path, payload):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def process_dataset(
    data_root,
    threshold=DEFAULT_THRESHOLD_METRES,
    apply=False,
    manifests=None,
    example_limit=DEFAULT_EXAMPLE_LIMIT,
):
    manifest_paths, annotation_paths = collect_annotation_paths(
        data_root, manifests
    )
    backup_root = None
    if apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_root = data_root / ".visibility_backups" / stamp

    summary = {
        "mode": "apply" if apply else "preview",
        "data_root": str(data_root),
        "distance_definition": "sqrt(center.x^2 + center.y^2)",
        "distance_threshold_metres": threshold,
        "comparison": "distance > threshold",
        "manifests": [str(path) for path in manifest_paths],
        "annotation_files": len(annotation_paths),
        "files_with_far_objects": 0,
        "files_changed": 0,
        "objects_scanned": 0,
        "objects_beyond_threshold": 0,
        "objects_already_zero": 0,
        "objects_to_change": 0,
        "objects_changed": 0,
        "invalid_centers": 0,
        "changes_by_class": Counter(),
        "original_visibility": Counter(),
        "examples": [],
        "backup_root": str(backup_root) if backup_root else None,
    }

    for annotation, relative in annotation_paths:
        payload, objects = _load_annotation(annotation)
        file_far = 0
        file_changes = 0
        for obj in objects:
            summary["objects_scanned"] += 1
            distance = _bev_distance(obj)
            if distance is None:
                summary["invalid_centers"] += 1
                continue
            if distance <= threshold:
                continue
            file_far += 1
            summary["objects_beyond_threshold"] += 1
            if obj.get("visibility") == 0:
                summary["objects_already_zero"] += 1
                continue

            file_changes += 1
            summary["objects_to_change"] += 1
            label = str(obj.get("label", "<missing>"))
            summary["changes_by_class"][label] += 1
            original = obj.get("visibility", "<missing>")
            summary["original_visibility"][str(original)] += 1
            if len(summary["examples"]) < example_limit:
                summary["examples"].append(
                    {
                        "annotation": str(relative),
                        "object_id": obj.get("id"),
                        "uuid": obj.get("uuid"),
                        "label": obj.get("label"),
                        "distance_metres": round(distance, 3),
                        "old_visibility": original,
                        "new_visibility": 0,
                    }
                )
            obj["visibility"] = 0

        if file_far:
            summary["files_with_far_objects"] += 1
        if not file_changes:
            continue
        summary["files_changed"] += 1
        if apply:
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation, backup)
            _atomic_write_json(annotation, payload)
            summary["objects_changed"] += file_changes

    summary["changes_by_class"] = dict(
        sorted(summary["changes_by_class"].items())
    )
    summary["original_visibility"] = dict(
        sorted(summary["original_visibility"].items())
    )
    if apply and summary["files_changed"] == 0:
        summary["backup_root"] = None
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview or set visibility=0 for MW3D objects whose vehicle-BEV "
            "distance exceeds a threshold"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing frames.jsonl or splits/",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_THRESHOLD_METRES,
        help="BEV distance threshold in metres (default: 80)",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help=(
            "Explicit manifest; repeat to use multiple manifests. Relative "
            "paths are resolved against data-root"
        ),
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLE_LIMIT,
        help="Maximum changed-object examples in the summary (default: 20)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes; without this flag the command is preview-only",
    )
    args = parser.parse_args()
    if not math.isfinite(args.distance_threshold) or args.distance_threshold <= 0:
        parser.error("--distance-threshold must be a positive finite number")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    return args


def main():
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit("Dataset root does not exist: %s" % data_root)
    manifests = None
    if args.manifest:
        manifests = []
        for path in args.manifest:
            path = path.expanduser()
            if not path.is_absolute():
                path = data_root / path
            path = path.resolve()
            if not path.is_file():
                raise SystemExit("Manifest does not exist: %s" % path)
            manifests.append(path)

    summary = process_dataset(
        data_root=data_root,
        threshold=args.distance_threshold,
        apply=args.apply,
        manifests=manifests,
        example_limit=args.examples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply and summary["objects_to_change"]:
        print(
            "\nPreview only: no files were modified. Re-run with --apply "
            "to write these changes."
        )


if __name__ == "__main__":
    main()
