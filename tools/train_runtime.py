#!/usr/bin/env python3
"""Atomic runtime registry for concurrent training jobs."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path

ACTIVE_STATES = {
    "reserved", "running", "crashed", "restarting", "stopping", "imported"
}


def utc_timestamp():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def process_start_ticks(pid):
    try:
        stat = Path("/proc/%d/stat" % int(pid)).read_text(encoding="utf-8")
        # The command name may contain spaces and parentheses. Everything
        # after its final ')' starts at proc field 3; starttime is field 22.
        fields_after_command = stat.rsplit(")", 1)[1].strip().split()
        if fields_after_command[0] == "Z":
            return None
        return fields_after_command[19]
    except (OSError, TypeError, ValueError, IndexError):
        return None


def process_matches(pid, expected_start_ticks=None):
    current = process_start_ticks(pid)
    if current is None:
        return False
    return expected_start_ticks is None or str(current) == str(expected_start_ticks)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


@contextmanager
def registry_lock(runtime_root):
    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "registry.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def parse_train_arguments(arguments):
    # Keep list/lookup/stop usable with only the Python standard library.
    from project_detection.config import load_config

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="+", action="append", default=[])
    parser.add_argument("--auto-resume", action="store_true")
    parsed, unknown = parser.parse_known_args(arguments)
    if unknown:
        raise ValueError("Unsupported training arguments: %s" % " ".join(unknown))
    overrides = [item for group in parsed.set for item in group]
    config = load_config(parsed.config, overrides)
    experiment = str(config["experiment"]["name"])
    if not experiment or any(character in experiment for character in "\r\n\0"):
        raise ValueError("experiment.name must be a non-empty single-line string")
    output_dir = (
        Path(config["experiment"]["output_dir"]) / experiment
    ).resolve()
    return config, experiment, output_dir


def visible_gpu_tokens(value=None):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "") if value is None else value
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    return tokens or ["0"]


def gpu_index_uuid_map():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in completed.stdout.splitlines():
        if "," not in line:
            continue
        index, uuid = (part.strip() for part in line.split(",", 1))
        result[index] = uuid
    return result


def canonical_gpu_resources(tokens):
    index_to_uuid = gpu_index_uuid_map()
    resources = []
    for token in tokens:
        if token in index_to_uuid:
            resources.append(index_to_uuid[token])
        elif token.startswith("GPU-") or token.startswith("MIG-"):
            resources.append(token)
        else:
            resources.append("visible:%s" % token)
    return resources


def job_identifier(experiment):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment).strip("._-")
    slug = (slug or "experiment")[:72]
    digest = hashlib.sha256(experiment.encode("utf-8")).hexdigest()[:10]
    return "%s-%s" % (slug, digest)


def load_metadata(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metadata_is_active(metadata):
    if not metadata or metadata.get("state") not in ACTIVE_STATES:
        return False
    pid = metadata.get("supervisor_pid")
    if pid and process_matches(pid, metadata.get("supervisor_start_ticks")):
        return True
    child_pid = metadata.get("child_pid")
    if child_pid and process_matches(child_pid, metadata.get("child_start_ticks")):
        return True
    for worker in metadata.get("worker_processes", ()):
        if process_matches(worker.get("pid"), worker.get("start_ticks")):
            return True
    if metadata.get("state") != "reserved":
        return False
    try:
        created = dt.datetime.fromisoformat(metadata["created_at"])
        age = dt.datetime.now(created.tzinfo) - created
        return age.total_seconds() < 120
    except (KeyError, TypeError, ValueError):
        return False


def iter_jobs(runtime_root):
    jobs_root = Path(runtime_root) / "jobs"
    if not jobs_root.is_dir():
        return
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir():
            continue
        metadata = load_metadata(job_dir / "metadata.json")
        if metadata is not None:
            yield job_dir, metadata


def mark_stale_jobs(runtime_root):
    jobs = []
    for job_dir, metadata in iter_jobs(runtime_root) or ():
        active = metadata_is_active(metadata)
        if not active and metadata.get("state") in ACTIVE_STATES:
            metadata["state"] = "stale"
            metadata["updated_at"] = utc_timestamp()
            atomic_json(job_dir / "metadata.json", metadata)
        jobs.append((job_dir, metadata, active))
    return jobs


def check_legacy_runtime(runtime_root):
    pid_path = Path(runtime_root) / "DetectionTrain.pid"
    if not pid_path.is_file():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not process_matches(pid):
        return None
    return pid


def ensure_no_conflict(jobs, experiment, output_dir, gpu_resources):
    requested = set(gpu_resources)
    for job_dir, metadata, active in jobs:
        if not active:
            continue
        if metadata.get("experiment") == experiment:
            raise RuntimeError(
                "experiment is already running: %s (job=%s)"
                % (experiment, job_dir.name)
            )
        if metadata.get("output_dir") == str(output_dir):
            raise RuntimeError(
                "output directory is already in use: %s" % output_dir
            )
        overlap = requested & set(metadata.get("gpu_resources", ()))
        if overlap:
            raise RuntimeError(
                "GPU is already reserved by %s: %s"
                % (metadata.get("experiment", job_dir.name), ",".join(sorted(overlap)))
            )


def reserve_job(runtime_root, train_arguments, visible_devices=None):
    _, experiment, output_dir = parse_train_arguments(train_arguments)
    gpu_tokens = visible_gpu_tokens(visible_devices)
    resources = canonical_gpu_resources(gpu_tokens)
    runtime_root = Path(runtime_root).resolve()
    with registry_lock(runtime_root):
        legacy_pid = check_legacy_runtime(runtime_root)
        if legacy_pid is not None:
            raise RuntimeError(
                "legacy DetectionTrain PID %d is active; stop the legacy "
                "training process before starting a managed job"
                % legacy_pid
            )
        jobs = mark_stale_jobs(runtime_root)
        ensure_no_conflict(jobs, experiment, output_dir, resources)
        identifier = job_identifier(experiment)
        job_dir = runtime_root / "jobs" / identifier
        job_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "job_id": identifier,
            "experiment": experiment,
            "output_dir": str(output_dir),
            "visible_devices": gpu_tokens,
            "gpu_resources": resources,
            "launch_mode": "ddp" if len(gpu_tokens) > 1 else "single",
            "world_size": len(gpu_tokens),
            "state": "reserved",
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "supervisor_pid": None,
            "supervisor_start_ticks": None,
            "child_pid": None,
            "child_start_ticks": None,
            "worker_processes": [],
            "train_arguments": list(train_arguments),
        }
        atomic_json(job_dir / "metadata.json", metadata)
    return job_dir, metadata


def activate_job(job_dir, pid):
    job_dir = Path(job_dir)
    runtime_root = job_dir.parent.parent
    with registry_lock(runtime_root):
        metadata = load_metadata(job_dir / "metadata.json")
        if metadata is None:
            raise RuntimeError("job metadata is missing: %s" % job_dir)
        start_ticks = process_start_ticks(pid)
        if start_ticks is None:
            raise RuntimeError("supervisor PID is not alive: %s" % pid)
        pending_state = metadata.pop("pending_state", "running")
        metadata.update(
            state=pending_state,
            updated_at=utc_timestamp(),
            supervisor_pid=int(pid),
            supervisor_start_ticks=start_ticks,
        )
        atomic_json(job_dir / "metadata.json", metadata)
        (job_dir / "pid").write_text("%d\n" % int(pid), encoding="utf-8")
    return metadata


def update_job_state(job_dir, state):
    job_dir = Path(job_dir)
    runtime_root = job_dir.parent.parent
    with registry_lock(runtime_root):
        metadata = load_metadata(job_dir / "metadata.json")
        if metadata is None:
            raise RuntimeError("job metadata is missing: %s" % job_dir)
        # The supervisor can write its first status between fork() and the
        # launcher's activate call. Preserve the reservation (and therefore
        # conflict protection) until its PID identity has been recorded.
        if metadata.get("state") == "reserved" and not metadata.get("supervisor_pid"):
            metadata["pending_state"] = state
        else:
            metadata["state"] = state
        metadata["updated_at"] = utc_timestamp()
        atomic_json(job_dir / "metadata.json", metadata)


def update_child_process(job_dir, pid=None):
    job_dir = Path(job_dir)
    runtime_root = job_dir.parent.parent
    with registry_lock(runtime_root):
        metadata = load_metadata(job_dir / "metadata.json")
        if metadata is None:
            raise RuntimeError("job metadata is missing: %s" % job_dir)
        if pid is None:
            metadata["child_pid"] = None
            metadata["child_start_ticks"] = None
        else:
            start_ticks = process_start_ticks(pid)
            if start_ticks is None:
                raise RuntimeError("training child PID is not alive: %s" % pid)
            metadata["child_pid"] = int(pid)
            metadata["child_start_ticks"] = start_ticks
        metadata["updated_at"] = utc_timestamp()
        atomic_json(job_dir / "metadata.json", metadata)


def update_worker_processes(job_dir, pids=()):
    job_dir = Path(job_dir)
    runtime_root = job_dir.parent.parent
    with registry_lock(runtime_root):
        metadata = load_metadata(job_dir / "metadata.json")
        if metadata is None:
            raise RuntimeError("job metadata is missing: %s" % job_dir)
        workers = []
        for pid in pids:
            start_ticks = process_start_ticks(pid)
            if start_ticks is not None:
                workers.append({"pid": int(pid), "start_ticks": start_ticks})
        metadata["worker_processes"] = workers
        metadata["updated_at"] = utc_timestamp()
        atomic_json(job_dir / "metadata.json", metadata)


def active_jobs(runtime_root):
    with registry_lock(runtime_root):
        return [item for item in mark_stale_jobs(runtime_root) if item[2]]


def find_active_job(runtime_root, selector=None):
    jobs = active_jobs(runtime_root)
    if selector:
        matches = [
            item for item in jobs
            if item[0].name == selector or item[1].get("experiment") == selector
        ]
        if not matches:
            raise RuntimeError("training job is not running: %s" % selector)
        if len(matches) > 1:
            raise RuntimeError("ambiguous training job: %s" % selector)
        return matches[0]
    if not jobs:
        raise RuntimeError("no training jobs are running")
    if len(jobs) > 1:
        names = ", ".join(item[1]["experiment"] for item in jobs)
        raise RuntimeError("multiple jobs are running; specify one of: %s" % names)
    return jobs[0]


def print_job_lines(job_dir, metadata):
    values = (
        str(job_dir),
        metadata["experiment"],
        metadata["job_id"],
        ",".join(metadata["visible_devices"]),
        ",".join(metadata["gpu_resources"]),
        metadata["output_dir"],
        str(metadata.get("supervisor_pid") or ""),
        str(metadata.get("supervisor_start_ticks") or ""),
        str(metadata.get("child_pid") or ""),
        str(metadata.get("child_start_ticks") or ""),
        metadata.get("launch_mode", "single"),
        str(metadata.get("world_size") or 1),
    )
    print("\n".join(values))


def _display_width(value):
    return sum(
        2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
        for character in str(value)
    )


def _pad_table_value(value, width, right_aligned=False):
    value = str(value)
    padding = " " * max(width - _display_width(value), 0)
    return padding + value if right_aligned else value + padding


def format_job_table(rows):
    """Format runtime rows without relying on tab-stop positions."""
    headers = ("EXPERIMENT", "GPU", "WORLD", "MODE", "PID", "ROLE", "STATE", "OUTPUT")
    normalized = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max([_display_width(headers[index])] + [
            _display_width(row[index]) for row in normalized
        ])
        for index in range(len(headers) - 1)
    ]
    right_aligned = {2, 4}

    def format_row(row):
        cells = [
            _pad_table_value(
                row[index], widths[index], index in right_aligned
            )
            for index in range(len(widths))
        ]
        return "  ".join(cells + [str(row[-1])])

    return "\n".join([format_row(headers)] + [format_row(row) for row in normalized])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--runtime-root", required=True)
    reserve.add_argument("train_arguments", nargs=argparse.REMAINDER)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--job-dir", required=True)
    activate.add_argument("--pid", required=True, type=int)

    state = subparsers.add_parser("state")
    state.add_argument("--job-dir", required=True)
    state.add_argument("--value", required=True)

    child_start = subparsers.add_parser("child-start")
    child_start.add_argument("--job-dir", required=True)
    child_start.add_argument("--pid", required=True, type=int)

    child_clear = subparsers.add_parser("child-clear")
    child_clear.add_argument("--job-dir", required=True)

    workers_set = subparsers.add_parser("workers-set")
    workers_set.add_argument("--job-dir", required=True)
    workers_set.add_argument("--pids", default="")

    listing = subparsers.add_parser("list")
    listing.add_argument("--runtime-root", required=True)

    lookup = subparsers.add_parser("lookup")
    lookup.add_argument("--runtime-root", required=True)
    lookup.add_argument("selector", nargs="?")

    args = parser.parse_args()
    try:
        if args.command == "reserve":
            train_arguments = list(args.train_arguments)
            if train_arguments[:1] == ["--"]:
                train_arguments = train_arguments[1:]
            job_dir, metadata = reserve_job(
                args.runtime_root, train_arguments
            )
            print_job_lines(job_dir, metadata)
        elif args.command == "activate":
            activate_job(args.job_dir, args.pid)
        elif args.command == "state":
            update_job_state(args.job_dir, args.value)
        elif args.command == "child-start":
            update_child_process(args.job_dir, args.pid)
        elif args.command == "child-clear":
            update_child_process(args.job_dir)
        elif args.command == "workers-set":
            pids = [int(value) for value in args.pids.split(",") if value]
            update_worker_processes(args.job_dir, pids)
        elif args.command == "list":
            jobs = active_jobs(args.runtime_root)
            rows = []
            for _, metadata, _ in jobs:
                supervisor_alive = process_matches(
                    metadata.get("supervisor_pid"),
                    metadata.get("supervisor_start_ticks"),
                ) if metadata.get("supervisor_pid") else False
                child_alive = bool(
                    metadata.get("child_pid")
                    and process_matches(
                        metadata.get("child_pid"),
                        metadata.get("child_start_ticks"),
                    )
                )
                live_workers = [
                    worker for worker in metadata.get("worker_processes", ())
                    if process_matches(worker.get("pid"), worker.get("start_ticks"))
                ]
                if supervisor_alive:
                    active_pid = metadata.get("supervisor_pid")
                    role = "supervisor"
                elif child_alive:
                    active_pid = metadata.get("child_pid")
                    role = "orphan-launcher"
                elif live_workers:
                    active_pid = live_workers[0]["pid"]
                    role = "orphan-worker"
                else:
                    active_pid = "-"
                    role = "stale"
                rows.append(
                    (
                        metadata["experiment"],
                        ",".join(metadata["visible_devices"]),
                        metadata.get("world_size", 1),
                        metadata.get("launch_mode", "single"),
                        active_pid,
                        role,
                        metadata["state"],
                        metadata["output_dir"],
                    )
                )
            print(format_job_table(rows))
        elif args.command == "lookup":
            job_dir, metadata, _ = find_active_job(
                args.runtime_root, args.selector
            )
            print_job_lines(job_dir, metadata)
    except (RuntimeError, ValueError, OSError) as error:
        print("train-runtime: %s" % error, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
