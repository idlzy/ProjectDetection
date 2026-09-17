from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
from pathlib import Path


LOGGER_NAME = "project_detection.train"


def set_process_name(name: str = "DetectionTrain") -> None:
    """Set Linux /proc/<pid>/comm without adding a third-party dependency."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(15, name.encode("utf-8")[:15], 0, 0, 0)
    except (AttributeError, OSError):
        pass


def configure_training_logging(output_dir: Path, rank: int = 0, level: str = "INFO"):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("train.log" if rank == 0 else "train.rank%d.log" % rank)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("%s", "=" * 88)
    logger.info(
        "Training process started | name=DetectionTrain | rank=%d | host=%s | python=%s",
        rank,
        platform.node(),
        platform.python_version(),
    )
    logger.info("Application log file: %s", log_path.resolve())
    return logger


def log_runtime_environment(
    logger, torch_module, device, world_size: int, rank: int = 0, local_rank: int = 0
) -> None:
    logger.info(
        "Runtime | torch=%s | launch_mode=%s | rank=%d | local_rank=%d | "
        "device=%s | world_size=%d | cwd=%s",
        torch_module.__version__,
        "ddp" if world_size > 1 else "single",
        rank,
        local_rank,
        device,
        world_size,
        os.getcwd(),
    )
    if device.type == "cuda":
        logger.info(
            "CUDA | version=%s | gpu=%s | device_count=%d",
            torch_module.version.cuda,
            torch_module.cuda.get_device_name(device),
            torch_module.cuda.device_count(),
        )
