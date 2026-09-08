from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Sequence


SPLIT_NAMES = ("train", "val", "test")


def record_stratum(record: dict, stratify_key: str) -> str:
    if stratify_key == "image_source":
        parts = PurePosixPath(record.get("image_rel", "")).parts
        if len(parts) < 2 or parts[0] != "images":
            raise ValueError(
                "image_rel must use images/<source>/... for image_source stratification: %s"
                % record.get("image_rel")
            )
        return parts[1]
    value = record.get(stratify_key)
    if not value:
        raise ValueError("Record is missing stratification field: %s" % stratify_key)
    return str(value)


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_number, error))
            records.append(record)
    if not records:
        raise ValueError("No records found in %s" % path)
    return records


def validate_records(records: Sequence[dict], data_root: Path, check_files: bool = True) -> None:
    required = ("sample_token", "image_rel", "ann_rel", "calib_rel")
    seen_tokens = set()
    missing_files = []
    for index, record in enumerate(records, 1):
        missing_fields = [key for key in required if not record.get(key)]
        if missing_fields:
            raise ValueError(
                "Record %d is missing required fields: %s" % (index, ", ".join(missing_fields))
            )
        token = record["sample_token"]
        if token in seen_tokens:
            raise ValueError("Duplicate sample_token at record %d: %s" % (index, token))
        seen_tokens.add(token)
        if check_files:
            for key in ("image_rel", "ann_rel", "calib_rel"):
                path = data_root / record[key]
                if not path.is_file() and len(missing_files) < 20:
                    missing_files.append("%s=%s" % (key, path))
    if missing_files:
        raise FileNotFoundError(
            "Dataset contains missing files (showing at most 20):\n" + "\n".join(missing_files)
        )


def split_records_by_group(
    records: Sequence[dict],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 3407,
    group_key: str = "split_group",
    stratify_key: str = "image_source",
) -> Dict[str, List[dict]]:
    if len(ratios) != len(SPLIT_NAMES) or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("at least one split ratio must be positive")
    normalized = [ratio / ratio_sum for ratio in ratios]

    strata = defaultdict(lambda: defaultdict(list))
    group_strata = {}
    for index, record in enumerate(records, 1):
        group = record.get(group_key) or record.get("ann_batch")
        if not group:
            raise ValueError("Record %d has neither %s nor ann_batch" % (index, group_key))
        group = str(group)
        stratum = record_stratum(record, stratify_key)
        previous_stratum = group_strata.setdefault(group, stratum)
        if previous_stratum != stratum:
            raise ValueError(
                "Group %s spans multiple strata: %s and %s"
                % (group, previous_stratum, stratum)
            )
        strata[stratum][group].append(record)

    result = {name: [] for name in SPLIT_NAMES}
    enabled = [name for name, ratio in zip(SPLIT_NAMES, normalized) if ratio > 0]
    random_generator = random.Random(seed)

    for stratum in sorted(strata):
        groups = list(strata[stratum].items())
        random_generator.shuffle(groups)
        # Place large groups first. The preceding shuffle remains the stable
        # random tie-breaker for equally sized groups.
        groups.sort(key=lambda item: len(item[1]), reverse=True)
        stratum_size = sum(len(group_records) for _, group_records in groups)
        targets = {
            name: stratum_size * ratio
            for name, ratio in zip(SPLIT_NAMES, normalized)
        }
        counts = {name: 0 for name in SPLIT_NAMES}
        group_counts = {name: 0 for name in SPLIT_NAMES}
        for group_index, (_, group_records) in enumerate(groups):
            empty_splits = [name for name in enabled if group_counts[name] == 0]
            remaining_groups = len(groups) - group_index
            candidates = (
                empty_splits
                if empty_splits and remaining_groups == len(empty_splits)
                else enabled
            )

            def allocation_error(destination):
                projected = dict(counts)
                projected[destination] += len(group_records)
                return sum(
                    ((projected[name] - targets[name]) ** 2) / max(targets[name], 1.0)
                    for name in enabled
                )

            destination = min(candidates, key=allocation_error)
            for record in group_records:
                item = dict(record)
                item["split"] = destination
                result[destination].append(item)
            counts[destination] += len(group_records)
            group_counts[destination] += 1
    return result


def summarize_splits(splits: Mapping[str, Sequence[dict]], group_key: str) -> dict:
    summary = {}
    for name in SPLIT_NAMES:
        records = splits[name]
        groups = {
            record.get(group_key) or record.get("ann_batch") for record in records
        }
        summary[name] = {"frames": len(records), "groups": len(groups)}
    return summary


def summarize_strata(
    splits: Mapping[str, Sequence[dict]], group_key: str, stratify_key: str
) -> dict:
    strata = defaultdict(lambda: {name: {"frames": 0, "groups": set()} for name in SPLIT_NAMES})
    for split_name in SPLIT_NAMES:
        for record in splits[split_name]:
            stratum = record_stratum(record, stratify_key)
            group = record.get(group_key) or record.get("ann_batch")
            strata[stratum][split_name]["frames"] += 1
            strata[stratum][split_name]["groups"].add(group)
    return {
        stratum: {
            split_name: {
                "frames": values["frames"],
                "groups": len(values["groups"]),
            }
            for split_name, values in split_values.items()
        }
        for stratum, split_values in sorted(strata.items())
    }


def write_splits(splits: Mapping[str, Sequence[dict]], output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / (name + "_frames.jsonl") for name in SPLIT_NAMES]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Split manifests already exist; pass --force to replace them: %s"
            % ", ".join(str(path) for path in existing)
        )
    for name, path in zip(SPLIT_NAMES, targets):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in splits[name]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)


def assert_no_leakage(splits: Mapping[str, Sequence[dict]], group_key: str) -> None:
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            left_tokens = {item["sample_token"] for item in splits[left]}
            right_tokens = {item["sample_token"] for item in splits[right]}
            if left_tokens & right_tokens:
                raise RuntimeError("sample_token leakage between %s and %s" % (left, right))
            left_groups = {item.get(group_key) or item.get("ann_batch") for item in splits[left]}
            right_groups = {item.get(group_key) or item.get("ann_batch") for item in splits[right]}
            if left_groups & right_groups:
                raise RuntimeError("group leakage between %s and %s" % (left, right))
