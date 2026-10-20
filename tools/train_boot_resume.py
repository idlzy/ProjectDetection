#!/usr/bin/env python3
"""Explicit, fail-closed boot recovery for managed training jobs (Linux)."""
import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

try:
    from tools import train_runtime as runtime
    from tools.train_checkpoint import select_checkpoint
except ModuleNotFoundError:
    import train_runtime as runtime
    from train_checkpoint import select_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("train-boot-resume")


def check_service_ready():
    """Never claim unattended recovery works without a persistent user manager."""
    username = pwd.getpwuid(os.getuid()).pw_name
    subprocess.run(["systemctl", "--user", "show-environment"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
    result = subprocess.run(
        ["loginctl", "show-user", username, "-p", "Linger", "--value"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    if result.stdout.strip() != "yes":
        raise RuntimeError(
            "One-time server setup required: sudo loginctl enable-linger %s. "
            "Then repeat the same start_train.sh command. To explicitly train "
            "WITHOUT boot recovery, set TRAIN_BOOT_RESUME=0." % username
        )


def install_for_job(job_dir, python_bin, max_boot_attempts=3):
    """Idempotently enable (never start) an owned unit, then arm this launch."""
    job_dir = Path(job_dir).resolve()
    check_service_ready()
    # Do not replace a record while a boot coordinator is still using it.
    with (job_dir / "boot_resume.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = runtime.load_metadata(job_dir / "metadata.json")
        if not metadata:
            raise RuntimeError("Missing managed job metadata")
        config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        if not config_home.is_absolute():
            raise RuntimeError("XDG_CONFIG_HOME must be absolute")
        directory = config_home / "systemd/user"
        directory.mkdir(parents=True, exist_ok=True)
        name = "projectdetection-%s.service" % metadata["job_id"]
        unit = directory / name
        expected = service_text(job_dir, python_bin)
        if unit.is_symlink():
            raise RuntimeError("Refusing to overwrite a symlink unit: %s" % unit)
        if unit.exists():
            if unit.read_text() != expected:
                raise RuntimeError("Existing unit was modified; inspect it first: %s" % unit)
        else:
            # Exclusive creation: another installer cannot overwrite this unit.
            with unit.open("x") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=15)
        subprocess.run(["systemctl", "--user", "enable", name], check=True, timeout=15)
        record = register(job_dir, python_bin, max_boot_attempts)
        return record


def mount_identity(path):
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,UUID"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    entry = json.loads(result.stdout)["filesystems"][0]
    return {"target": entry["target"], "uuid": entry.get("uuid")}


def register(job_dir, python_bin, max_boot_attempts=3):
    job_dir = Path(job_dir).resolve()
    if max_boot_attempts < 1:
        raise ValueError("max_boot_attempts must be positive")
    with runtime.registry_lock(job_dir.parent.parent):
        metadata = runtime.load_metadata(job_dir / "metadata.json")
        if not metadata:
            raise RuntimeError("No managed training job: %s" % job_dir)
        if metadata["state"] == "completed":
            raise RuntimeError("Completed jobs cannot be armed")
        if not runtime.metadata_is_active(metadata) and not (job_dir / "boot_resume.json").exists():
            raise RuntimeError("Register a running managed job, not a historical experiment")
        config = runtime.load_metadata(job_dir / "launch_config.json")
        if not config:
            raise RuntimeError("Missing launch snapshot; restart through updated start_train.sh first")
        if not Path(python_bin).is_file() or not os.access(python_bin, os.X_OK):
            raise RuntimeError("Python interpreter is not executable")
        if metadata.get("python") and Path(metadata["python"]).resolve() != Path(python_bin).resolve():
            raise RuntimeError("Register using the same Python environment as the running job")
        data_paths = {str(Path(config["data"][key]).resolve())
                      for key in ("data_root", "ready_root") if config["data"].get(key)}
        data_paths.add(str(Path(metadata["output_dir"]).resolve()))
        mounts = []
        for path in sorted(data_paths):
            if not Path(path).is_dir():
                raise RuntimeError("Required directory does not exist: %s" % path)
            mounts.append({"path": path, **mount_identity(path)})
        devices = runtime.canonical_gpu_resources(metadata["visible_devices"])
        if any(not device.startswith("GPU-") for device in devices):
            raise RuntimeError("Boot resume requires resolvable physical GPU UUIDs; MIG is not supported")
        config_bytes = (json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n").encode()
        snapshot = job_dir / "boot_config.json"
        runtime.atomic_json(snapshot, config, durable=True)
        record = {
            "schema_version": 1, "desired_running": True,
            "job_id": metadata["job_id"], "experiment": metadata["experiment"],
            "output_dir": metadata["output_dir"], "project_root": str(PROJECT_ROOT),
            "python": str(Path(python_bin).absolute()), "gpu_uuids": devices,
            "config": str(snapshot),
            "config_hash": hashlib.sha256(config_bytes).hexdigest(),
            "mounts": mounts, "boot_attempts": 0,
            "max_boot_attempts": max_boot_attempts,
            # Enabling never starts a second training in the current boot.
            "last_attempt_boot_id": runtime.boot_id(),
            "supervisor_max_restarts": 2,
            "environment": {key: os.environ[key] for key in (
                "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_DEVICE_ORDER",
                "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
            ) if key in os.environ},
            "created_at": runtime.utc_timestamp(), "last_result": "armed",
        }
        runtime.atomic_json(job_dir / "boot_resume.json", record, durable=True)
    return record


def read_record(job_dir):
    record = runtime.load_metadata(Path(job_dir) / "boot_resume.json")
    if not record:
        raise RuntimeError("Job has not been explicitly armed")
    return record


def claim_attempt(job_dir):
    with runtime.registry_lock(Path(job_dir).parent.parent):
        record = read_record(job_dir)
        if not record["desired_running"]:
            return None
        if record["last_attempt_boot_id"] == runtime.boot_id():
            LOGGER.info("Already armed/attempted in this boot; not launching again")
            return None
        if record["boot_attempts"] >= record["max_boot_attempts"]:
            raise RuntimeError("Persistent boot-attempt budget exhausted; inspect and explicitly re-arm")
        record["boot_attempts"] += 1
        record["last_attempt_boot_id"] = runtime.boot_id()
        record["last_result"] = "preflight"
        runtime.atomic_json(Path(job_dir) / "boot_resume.json", record, durable=True)
        return record


def preflight(record):
    from project_detection.config import load_config

    if Path.cwd() != Path(record["project_root"]):
        raise RuntimeError("Project working directory changed")
    config = load_config(record["config"])
    digest = hashlib.sha256((json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest()
    if digest != record["config_hash"]:
        raise RuntimeError("Frozen boot configuration has been modified; explicitly re-register")
    output = (Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]).resolve()
    if str(output) != record["output_dir"]:
        raise RuntimeError("Snapshot experiment output does not match registered job")
    for expected in record["mounts"]:
        path = Path(expected["path"])
        if not path.is_dir() or not os.path.ismount(expected["target"]):
            raise RuntimeError("Required mount/directory is unavailable: %s" % path)
        actual = mount_identity(path)
        if actual != {key: expected[key] for key in ("target", "uuid")}:
            raise RuntimeError("Required filesystem identity changed: %s" % path)
    ready = Path(config["data"].get("ready_root") or config["data"]["data_root"])
    for split in ("train", "val"):
        if not (ready / "splits" / (split + "_frames.jsonl")).is_file():
            raise RuntimeError("Missing %s split manifest" % split)
    distillation = config["train"].get("distillation", {})
    if distillation.get("enabled"):
        teacher = distillation.get("teacher", {}).get("checkpoint")
        if not teacher or not Path(teacher).is_file():
            raise RuntimeError("Distillation teacher checkpoint is missing")
    available = set(runtime.gpu_index_uuid_map().values())
    if not set(record["gpu_uuids"]) <= available:
        raise RuntimeError("Registered GPUs are not ready")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    for line in processes.stdout.splitlines():
        if line.split(",", 1)[0].strip() in record["gpu_uuids"]:
            raise RuntimeError("Registered GPU is occupied: %s" % line)
    jobs = runtime.active_jobs(PROJECT_ROOT / ".runtime")
    runtime.ensure_no_conflict(jobs, record["experiment"], output, record["gpu_uuids"])
    # Detect a directly launched (unregistered) tools/train.py process too.
    with (output / ".training.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return config


def save_result(job_dir, result):
    with runtime.registry_lock(Path(job_dir).parent.parent):
        record = read_record(job_dir)
        record.update(last_result=result, updated_at=runtime.utc_timestamp())
        runtime.atomic_json(Path(job_dir) / "boot_resume.json", record, durable=True)


def run(job_dir, wait_seconds=300):
    job_dir = Path(job_dir).resolve()
    # Lock retained for the service lifetime; restarting systemd cannot create
    # another coordinator for the same experiment.
    with (job_dir / "boot_resume.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = claim_attempt(job_dir)
        if record is None:
            return 0
        child = None
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        deadline = time.monotonic() + wait_seconds
        try:
            while True:
                if stopping or not read_record(job_dir)["desired_running"]:
                    return 0
                try:
                    config = preflight(record)
                    break
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    LOGGER.warning("Preflight not ready: %s", error)
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
            checkpoint = select_checkpoint(config, len(record["gpu_uuids"]))
            LOGGER.info("Boot recovery | attempt=%d/%d | checkpoint=%s",
                        record["boot_attempts"], record["max_boot_attempts"], checkpoint)
            environment = os.environ.copy()
            environment.update(record.get("environment", {}))
            environment.update(TRAIN_PYTHON=record["python"],
                               TRAIN_BOOT_JOB_DIR=str(job_dir),
                               CUDA_VISIBLE_DEVICES=",".join(record["gpu_uuids"]),
                               TRAIN_MAX_RESTARTS=str(record["supervisor_max_restarts"]),
                               TRAIN_RESTART_DELAY="30")
            command = ["bash", str(PROJECT_ROOT / "scripts/start_train.sh"),
                       "--foreground", "--config", record["config"], "--auto-resume"]
            # Serialize disarm vs launch. If stop wins, no child is created.
            with runtime.registry_lock(job_dir.parent.parent):
                if stopping or not read_record(job_dir)["desired_running"]:
                    return 0
                child = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=environment)
            if stopping:
                child.send_signal(signal.SIGTERM)
            result = child.wait()
            if stopping:
                save_result(job_dir, "service_shutdown")
                return 0
            save_result(job_dir, "completed" if result == 0 else "failed:%d" % result)
            return result
        except Exception as error:
            save_result(job_dir, "blocked:%s" % error)
            raise


def service_text(job_dir, python_bin):
    def quote(value):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'
    return "\n".join([
        "[Unit]", "Description=ProjectDetection explicit boot recovery",
        "", "[Service]", "Type=simple",
        "WorkingDirectory=" + str(PROJECT_ROOT).replace("\\", "\\\\").replace("%", "%%"),
        "ExecStart=" + " ".join(quote(v) for v in [python_bin, PROJECT_ROOT / "tools/train_boot_resume.py", "run", "--job-dir", job_dir]),
        "Restart=no", "KillMode=mixed", "TimeoutStopSec=120", "LimitNOFILE=65535",
        "Environment=PYTHONUNBUFFERED=1", "", "[Install]", "WantedBy=default.target", "",
    ])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-service")
    for name in ("register", "run", "status"):
        command = sub.add_parser(name)
        command.add_argument("--job-dir", required=True, type=Path)
        if name == "register":
            command.add_argument("--max-boot-attempts", type=int, default=3)
            command.add_argument("--quiet", action="store_true")
        elif name == "run":
            command.add_argument("--wait-seconds", type=int, default=300)
    args = parser.parse_args()
    os.chdir(str(PROJECT_ROOT))
    try:
        if args.command == "check-service":
            check_service_ready()
        elif args.command == "register":
            try:
                record = install_for_job(args.job_dir, sys.executable,
                                         args.max_boot_attempts)
            except Exception:
                runtime.disarm_boot_resume(args.job_dir, "installation_failed")
                raise
            if args.quiet:
                print("Boot recovery enabled | job=%s | max_boot_attempts=%d"
                      % (record["job_id"], record["max_boot_attempts"]))
            else:
                print(json.dumps(record, ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(read_record(args.job_dir), ensure_ascii=False, indent=2))
        else:
            return run(args.job_dir, args.wait_seconds)
    except Exception as error:
        LOGGER.error("Recovery stopped safely: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
