#!/usr/bin/env python3
"""Build an independent mw3d_1020 aggregate dataset.

mw3d_1020 = (mw3d_v5 + mw3d_0911 of mw3d_0915, KITTI/ONCE removed)
          + new_data_1020 delivery (5 new annotation batches)
          with 6 re-exported annotation batches replacing the old ones.

All payload files are real copies: the output dataset never symlinks back to
the source roots and does not depend on them after the build.
Existing split assignments are preserved verbatim; only split groups that did
not exist before are assigned with the usual 80/10/10 group split (seed 3407).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_detection.data.split import (  # noqa: E402
    SPLIT_NAMES,
    assert_no_leakage,
    read_jsonl,
    split_records_by_group,
    validate_records,
)

BASE_SOURCES = ("mw3d_v5", "mw3d_0911")
NEW_SOURCE_DATASET = "mw3d_1020"
REMOVED_SOURCES = ("KITTI", "ONCE_cam01", "ONCE_cam03")
UPDATED_BATCH_PREFIXES = ("0911-", "0914-")
STAGING_PREFIX = ".mw3d_1020.tmp."


def read_records(path: Path):
    if not path.is_file():
        raise FileNotFoundError("Missing manifest: %s" % path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=shutil.copy2, symlinks=False)


def discover_image_sessions(image_root: Path):
    sessions = {}
    for batch_root in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for session_root in sorted(path for path in batch_root.iterdir() if path.is_dir()):
            calibration = session_root / "sensor_params" / "FrontViewCalibParam.json"
            if not calibration.is_file():
                continue
            session = session_root.name
            if session in sessions:
                raise ValueError("Duplicate session directory name: %s" % session)
            sessions[session] = {"batch": batch_root.name, "root": session_root}
    return sessions


def annotation_members(zip_file: ZipFile):
    for info in zip_file.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".json"):
            continue
        parts = PurePosixPath(info.filename).parts
        if len(parts) != 3 or parts[1] != "json" or any(p in ("", ".", "..") for p in parts):
            raise ValueError("Unexpected annotation member path: %s" % info.filename)
        yield info, parts[0], Path(parts[2]).stem


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError("Missing source file: %s" % source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base_root = args.base_root.expanduser().resolve()
    delivery_root = args.delivery_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit("Output already exists: %s" % output_root)

    image_root = delivery_root / "new_images"
    annotation_root = delivery_root / "new_ann"
    for required in (base_root, image_root, annotation_root):
        if not required.is_dir():
            raise SystemExit("Missing required directory: %s" % required)

    report = {"output_root": str(output_root), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------------------------------------------------------- old side
    base_frames = read_jsonl(base_root / "frames.jsonl")
    base_splits = {
        split: read_records(base_root / "splits" / (split + "_frames.jsonl"))
        for split in SPLIT_NAMES
    }
    token_split = {}
    for split in SPLIT_NAMES:
        for record in base_splits[split]:
            token_split[record["sample_token"]] = split
    log("base frames=%d splits=%s" % (len(base_frames), {k: len(v) for k, v in base_splits.items()}))

    base_records = []
    stale_split = 0
    for record in base_frames:
        if record.get("source_dataset") not in BASE_SOURCES:
            continue
        token = record["sample_token"]
        if token not in token_split:
            raise ValueError("Base frame missing from split manifests: %s" % token)
        if record.get("split") != token_split[token]:
            stale_split += 1
        record = dict(record)
        record["split"] = token_split[token]
        base_records.append(record)
    log("kept mw3d base frames=%d (stale split field repaired: %d)" % (len(base_records), stale_split))
    report["base_frames_kept"] = len(base_records)
    report["stale_split_fields_repaired"] = stale_split

    kept_tokens = {r["sample_token"] for r in base_records}
    base_splits = {
        split: [r for r in base_splits[split] if r["sample_token"] in kept_tokens]
        for split in SPLIT_NAMES
    }
    if sum(len(v) for v in base_splits.values()) != len(base_records):
        raise ValueError("Base split manifests do not cover every kept frame")

    image_dirs = sorted({r["image_rel"].split("/")[1] for r in base_records})
    annotation_dirs = sorted({r["ann_rel"].split("/")[1] for r in base_records})
    for removed in REMOVED_SOURCES:
        if removed in image_dirs or removed in annotation_dirs:
            raise ValueError("Removed source still referenced: %s" % removed)
    log("base image dirs=%d annotation dirs=%d" % (len(image_dirs), len(annotation_dirs)))

    # ---------------------------------------------------------------- new side
    archives = sorted(annotation_root.glob("*.zip"))
    updated_batches = sorted(a.stem for a in archives if a.stem.startswith(UPDATED_BATCH_PREFIXES))
    new_batches = sorted(a.stem for a in archives if not a.stem.startswith(UPDATED_BATCH_PREFIXES))
    log("updated annotation batches=%d new annotation batches=%d" % (len(updated_batches), len(new_batches)))
    report["updated_annotation_batches"] = updated_batches
    report["new_annotation_batches"] = new_batches

    sessions = discover_image_sessions(image_root)
    log("new image sessions=%d" % len(sessions))
    report["new_image_sessions"] = len(sessions)

    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=str(output_root.parent)))
    log("staging=%s" % staging)
    try:
        # 1. copy mw3d payload of the base dataset (real files, no symlinks)
        for name in image_dirs:
            copy_tree(base_root / "images" / name, staging / "images" / name)
            log("copied images/%s" % name)
        for name in annotation_dirs:
            if name in updated_batches:
                continue  # replaced below by the re-exported annotations
            copy_tree(base_root / "annotation" / name, staging / "annotation" / name)
            log("copied annotation/%s" % name)

        # 2. new + re-exported annotations and FrontLeft payloads
        base_index = {}
        base_batch_counts = Counter()
        for record in base_records:
            session = record["split_group"].split("__", 1)[1]
            base_index[(record["ann_batch"], session, record["stem"])] = record
            base_batch_counts[record["ann_batch"]] += 1
        rewritten = {}
        new_records = []
        source_counts = Counter()
        batch_counts = Counter()
        calib_copied = set()
        for archive in archives:
            batch = archive.stem
            rewritten[batch] = 0
            with ZipFile(archive) as zip_file:
                for info, session, stem in annotation_members(zip_file):
                    annotation_rel = Path("annotation", batch, session, "json", stem + ".json")
                    if batch in updated_batches:
                        # Images/calibration come from the copied base dataset.
                        if (batch, session, stem) not in base_index:
                            raise ValueError(
                                "Re-exported annotation has no base frame: %s/%s/%s"
                                % (batch, session, stem)
                            )
                        target = staging / annotation_rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zip_file.open(info) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        rewritten[batch] += 1
                        continue
                    if session not in sessions:
                        raise FileNotFoundError("Annotation session without images: %s" % session)
                    details = sessions[session]
                    image_batch = details["batch"]
                    raw_session = details["root"]
                    image_rel = Path("images", image_batch, session, "camera",
                                     "AIU_Camera_FrontLeft", stem + ".jpg")
                    extrinsic_rel = Path("images", image_batch, session, "camera_extrinsic",
                                         "camera_front_left", stem + ".txt")
                    calib_rel = Path("images", image_batch, session, "sensor_params",
                                     "FrontViewCalibParam.json")

                    copy_file(raw_session / "camera" / "AIU_Camera_FrontLeft" / (stem + ".jpg"),
                              staging / image_rel)
                    copy_file(raw_session / "camera_extrinsic" / "camera_front_left" / (stem + ".txt"),
                              staging / extrinsic_rel)
                    key = (image_batch, session)
                    if key not in calib_copied:
                        copy_file(raw_session / "sensor_params" / "FrontViewCalibParam.json",
                                  staging / calib_rel)
                        calib_copied.add(key)

                    target = staging / annotation_rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zip_file.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

                    token = "%s__%s__%s__%s" % (NEW_SOURCE_DATASET, batch, session, stem)
                    new_records.append({
                        "split_group": batch + "__" + session,
                        "sample_token": token,
                        "stem": stem,
                        "image_rel": image_rel.as_posix(),
                        "ann_rel": annotation_rel.as_posix(),
                        "anno_rel": annotation_rel.as_posix(),
                        "calib_rel": calib_rel.as_posix(),
                        "extrinsic_rel": extrinsic_rel.as_posix(),
                        "ann_batch": batch,
                        "source_dataset": NEW_SOURCE_DATASET,
                    })
                    source_counts[image_batch] += 1
                    batch_counts[batch] += 1
                    if len(new_records) % 500 == 0:
                        log("imported %d new frames" % len(new_records))
        log("imported %d new frames; copied %d new images" % (len(new_records), sum(source_counts.values())))
        report["new_frames"] = len(new_records)
        report["new_images_copied"] = sum(source_counts.values())
        for batch in updated_batches:
            if rewritten[batch] != base_batch_counts[batch]:
                raise ValueError(
                    "Re-exported batch %s has %d frames, base has %d"
                    % (batch, rewritten[batch], base_batch_counts[batch])
                )
        log("re-exported annotation files: %s"
            % {b: rewritten[b] for b in updated_batches})

        # 3. split new groups with the historical policy
        new_splits = split_records_by_group(
            new_records,
            ratios=(0.8, 0.1, 0.1),
            seed=3407,
            group_key="split_group",
            stratify_key="image_source",
        )
        assert_no_leakage(new_splits, "split_group")
        by_token = {r["sample_token"]: r for records in new_splits.values() for r in records}
        new_records = [by_token[r["sample_token"]] for r in new_records]

        final_records = base_records + new_records
        final_splits = {split: base_splits[split] + list(new_splits[split]) for split in SPLIT_NAMES}
        assert_no_leakage(final_splits, "split_group")
        split_tokens = {r["sample_token"] for v in final_splits.values() for r in v}
        if split_tokens != {r["sample_token"] for r in final_records}:
            raise ValueError("Split manifests and frames.jsonl disagree on frame coverage")
        if sum(len(v) for v in final_splits.values()) != len(final_records):
            raise ValueError("Split manifests contain duplicated or extra frames")
        if len({r["sample_token"] for r in final_records}) != len(final_records):
            raise ValueError("Duplicate sample tokens in final manifest")

        # verify preserved assignments before anything is published
        for split in SPLIT_NAMES:
            old_tokens = {r["sample_token"] for r in base_splits[split]}
            new_tokens = {r["sample_token"] for r in final_splits[split]}
            if not old_tokens <= new_tokens:
                raise ValueError("Base %s assignments are not preserved" % split)
        validate_records(final_records, staging, check_files=True)
        log("validated %d frames, all referenced files present" % len(final_records))

        write_jsonl(staging / "frames.jsonl", final_records)
        for split in SPLIT_NAMES:
            write_jsonl(staging / "splits" / (split + "_frames.jsonl"), final_splits[split])

        group_split = {}
        for split in SPLIT_NAMES:
            for record in final_splits[split]:
                group = record["split_group"]
                if group_split.setdefault(group, split) != split:
                    raise ValueError("Group spans multiple splits: %s" % group)

        summary = {
            "schema_version": 2,
            "name": output_root.name,
            "strategy": "copy_base_mw3d_replace_updated_batches_append_new_delivery",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_roots": {
                "base": str(base_root),
                "delivery": str(delivery_root),
                "delivery_ann": str(annotation_root),
                "delivery_images": str(image_root),
            },
            "storage": "real_copies_no_symlinks",
            "removed_external_sources": list(REMOVED_SOURCES),
            "updated_annotation_batches": updated_batches,
            "new_annotation_batches": new_batches,
            "split_policy": {
                "existing_frames": "preserved_verbatim_from_base_splits",
                "new_groups": {"ratios": [0.8, 0.1, 0.1], "seed": 3407,
                               "group_key": "split_group", "stratify_key": "image_source"},
                "group_key": "ann_batch__session",
            },
            "total_frames": len(final_records),
            "split_counts": {split: len(final_splits[split]) for split in SPLIT_NAMES},
            "split_groups": {
                split: len({r["split_group"] for r in final_splits[split]}) for split in SPLIT_NAMES
            },
            "source_counts": {
                "mw3d_v5": sum(1 for r in base_records if r["source_dataset"] == "mw3d_v5"),
                "mw3d_0911": sum(1 for r in base_records if r["source_dataset"] == "mw3d_0911"),
                "mw3d_1020": len(new_records),
            },
            "new_image_batch_counts": dict(sorted(source_counts.items())),
            "new_annotation_batch_counts": dict(sorted(batch_counts.items())),
        }
        write_json(staging / "summary.json", summary)
        report["summary"] = summary

        # 4. publish atomically (staging is a real directory tree on the same fs)
        os.replace(staging, output_root)
        log("published %s" % output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
