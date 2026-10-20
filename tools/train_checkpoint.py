"""Select a trusted local, complete training checkpoint without using mtime."""
import logging
from pathlib import Path


def select_checkpoint(config, world_size=1):
    import torch

    directory = (Path(config["experiment"]["output_dir"])
                 / config["experiment"]["name"] / "checkpoints")
    valid = []
    errors = []
    for path in (directory / "recovery.pth", directory / "last.pth"):
        if not path.is_file():
            continue
        try:
            try:
                saved = torch.load(str(path), map_location="cpu", weights_only=False)
            except TypeError:
                saved = torch.load(str(path), map_location="cpu")
            if not isinstance(saved, dict):
                raise ValueError("not a training checkpoint")
            for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "config",
                        "random_state", "best_metric"):
                if key not in saved:
                    raise ValueError("missing training state: %s" % key)
            for key in ("model", "optimizer", "scheduler", "scaler", "config"):
                if not isinstance(saved[key], dict):
                    raise ValueError("invalid training state: %s" % key)
            if not saved["model"]:
                raise ValueError("empty model state")
            if not isinstance(saved["random_state"], dict) or not {"python", "numpy", "torch"} <= set(saved["random_state"]):
                raise ValueError("incomplete random state")
            if not {"state", "param_groups"} <= set(saved["optimizer"]):
                raise ValueError("incomplete optimizer state")
            previous = saved["config"]
            for key in ("name", "seed"):
                if previous.get("experiment", {}).get(key) != config["experiment"].get(key):
                    raise ValueError("experiment.%s changed" % key)
            if previous.get("model") != config["model"]:
                raise ValueError("model configuration changed")
            # These changes would silently change training semantics on resume.
            for section in ("data", "evaluation"):
                if previous.get(section) != config[section]:
                    raise ValueError("%s configuration changed" % section)
            ignored = {"resume", "pretrain", "recovery_checkpoint_every_steps"}
            for key in set(previous.get("train", {})) | set(config["train"]):
                if key not in ignored and previous.get("train", {}).get(key) != config["train"].get(key):
                    raise ValueError("train.%s changed" % key)
            if config["train"].get("distillation", {}).get("enabled"):
                if not isinstance(saved.get("distillation"), dict):
                    raise ValueError("missing distillation state")
            epoch = saved["epoch"]
            step = saved.get("step_in_epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("invalid epoch")
            if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
                raise ValueError("invalid step_in_epoch")
            topology = saved.get("distributed_state") or {}
            if step is not None:
                if (world_size > 1 and not topology) or (
                    topology and (topology.get("world_size") != world_size
                    or topology.get("batch_size_per_gpu") != config["data"]["batch_size_per_gpu"])
                ):
                    raise ValueError("mid-epoch GPU/batch topology changed")
            # An epoch-boundary checkpoint is later than any mid-epoch save
            # with the same zero-based epoch, regardless of filesystem mtime.
            progress = (epoch, float("inf") if step is None else step)
            valid.append((progress, path))
        except Exception as error:
            errors.append("%s: %s" % (path, error))
            logging.getLogger(__name__).warning("Skipping checkpoint %s: %s", path, error)
        finally:
            saved = None
    if not valid:
        raise RuntimeError("No valid recovery/last checkpoint; refusing fresh training. %s"
                           % "; ".join(errors))
    return str(max(valid, key=lambda item: item[0])[1])
