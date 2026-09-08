from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import Mw3dReadyDataset, collate_detection_batch
from .logging_utils import configure_training_logging, log_runtime_environment
from .metrics import Mw3dMetric
from .models import build_model
from .task import FCOS3DLoss, FCOS3DPostProcessor


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_loader(config, split, world_size=1, max_samples=None):
    data = config["data"]
    dataset = Mw3dReadyDataset(
        data["data_root"],
        data.get("ready_root"),
        split,
        data["classes"],
        data["image_size"],
        max_samples,
        image_mean=data.get("image_mean", (128.0, 128.0, 128.0)),
        image_std=data.get("image_std", (128.0, 128.0, 128.0)),
        pad_value=data.get("pad_value", (0.0, 0.0, 0.0)),
    )
    sampler = DistributedSampler(dataset, shuffle=split == "train") if world_size > 1 else None
    return DataLoader(dataset, batch_size=data["batch_size_per_gpu"], shuffle=split == "train" and sampler is None,
                      sampler=sampler, num_workers=data["num_workers"], pin_memory=True,
                      collate_fn=collate_detection_batch, drop_last=split == "train")


def _json_value(value):
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {key: _json_value(item) for key, item in value.items()}
    return value


@torch.no_grad()
def evaluate(model, loader, config, device):
    model.eval(); raw_model = model.module if hasattr(model, "module") else model
    processor = FCOS3DPostProcessor(raw_model, config)
    metric = Mw3dMetric(config["data"]["classes"], config["evaluation"]["distance_thresholds"])
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        metric.update(processor(model(images), targets), targets)
    if distributed.is_initialized():
        states = [None for _ in range(distributed.get_world_size())]
        distributed.all_gather_object(states, metric.state_dict())
        metric.load_state_dict([item for state in states for item in state])
    return _json_value(metric.compute())


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_metric": best_metric, "config": config}, temporary)
    os.replace(str(temporary), str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, strict=True):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint: scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint, result


def train(config):
    local_rank, world_size = setup_distributed(); rank = distributed.get_rank() if distributed.is_initialized() else 0
    seed_everything(config["experiment"]["seed"] + rank)
    requested_device = config["runtime"]["device"]
    device = torch.device("cuda", local_rank) if requested_device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    logger = configure_training_logging(output_dir, rank, config["runtime"].get("log_level", "INFO"))
    log_runtime_environment(logger, torch, device, world_size)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle: yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        logger.info("Experiment | name=%s | seed=%d | output=%s", config["experiment"]["name"], config["experiment"]["seed"], output_dir.resolve())
        logger.info("Data configuration:\n%s", yaml.safe_dump(config["data"], sort_keys=False, allow_unicode=True).rstrip())
        logger.info("Model configuration:\n%s", yaml.safe_dump(config["model"], sort_keys=False, allow_unicode=True).rstrip())
        logger.info("Training hyperparameters:\n%s", yaml.safe_dump(config["train"], sort_keys=False, allow_unicode=True).rstrip())
        logger.info("Runtime configuration:\n%s", yaml.safe_dump(config["runtime"], sort_keys=False, allow_unicode=True).rstrip())
    model = build_model(config).to(device)
    if rank == 0:
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        logger.info("Model parameters | total=%d | trainable=%d", total_parameters, trainable_parameters)
        logger.info("Model architecture:\n%s", model)
    if world_size > 1 and config["train"]["sync_batch_norm"]: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["amp"] and device.type == "cuda")
    train_loader = build_loader(config, "train", world_size, config["runtime"].get("max_train_samples"))
    val_loader = build_loader(config, "val", world_size, config["runtime"].get("max_val_samples"))
    total_steps = max(config["train"]["epochs"] * len(train_loader), 1)
    warmup_steps = max(int(total_steps * config["train"]["lr_warmup_fraction"]), 1)

    def lr_factor(step):
        if step < warmup_steps:
            return 1.0 + (config["train"]["lr_peak_ratio"] - 1.0) * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        peak = config["train"]["lr_peak_ratio"]
        return peak * ((config["train"]["lr_final_ratio"] / peak) ** progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if rank == 0:
        global_batch_size = config["data"]["batch_size_per_gpu"] * world_size
        logger.info(
            "Dataset scale | train=%d/%d frames | val=%d/%d frames | train_batches=%d | val_batches=%d",
            len(train_loader.dataset),
            train_loader.dataset.available_samples,
            len(val_loader.dataset),
            val_loader.dataset.available_samples,
            len(train_loader),
            len(val_loader),
        )
        logger.info(
            "Training schedule | epochs=%d | steps_per_epoch=%d | total_steps=%d | batch_per_gpu=%d | global_batch=%d | warmup_steps=%d",
            config["train"]["epochs"],
            len(train_loader),
            total_steps,
            config["data"]["batch_size_per_gpu"],
            global_batch_size,
            warmup_steps,
        )
    start_epoch, best = 0, -float("inf")
    resume = config["train"].get("resume")
    if resume:
        checkpoint, _ = load_checkpoint(resume, model, optimizer, scheduler, scaler, strict=True)
        start_epoch = checkpoint.get("epoch", -1) + 1; best = checkpoint.get("best_metric", best)
        logger.info("Resumed checkpoint | path=%s | start_epoch=%d | best_NDS=%.6f", resume, start_epoch, best)
    elif config["train"].get("pretrain"):
        checkpoint, load_result = load_checkpoint(
            config["train"]["pretrain"], model, strict=False
        )
        conversion = checkpoint.get("conversion_report", {})
        logger.info(
            "Loaded pretrained weights | path=%s | loaded_tensors=%s | "
            "backbone_element_ratio=%s | missing=%d | unexpected=%d",
            config["train"]["pretrain"],
            conversion.get("loaded_tensors", "unknown"),
            (
                "%.4f" % conversion["loaded_backbone_element_ratio"]
                if "loaded_backbone_element_ratio" in conversion
                else "unknown"
            ),
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
    if world_size > 1: model = DistributedDataParallel(model, device_ids=[local_rank])
    criterion = FCOS3DLoss(model.module if hasattr(model, "module") else model, len(config["data"]["classes"]),
                           config["model"]["strides"], config["model"]["regress_ranges"],
                           config["model"].get("geometry"), config["data"]["classes"])
    training_started = time.time()
    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            epoch_started = time.time()
            model.train()
            if isinstance(train_loader.sampler, DistributedSampler): train_loader.sampler.set_epoch(epoch)
            if rank == 0:
                logger.info("Epoch %d/%d started", epoch + 1, config["train"]["epochs"])
            for step, (images, targets) in enumerate(train_loader):
                step_started = time.time()
                images = images.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    losses = criterion(model(images), targets)
                scaler.scale(losses["loss_total"]).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip_norm"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                # AMP skips optimizer.step on overflow; the LR schedule must skip too.
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                if rank == 0 and (step % config["runtime"]["log_every"] == 0 or step + 1 == len(train_loader)):
                    loss_text = " ".join("%s=%.5f" % (key, value.item()) for key, value in losses.items())
                    gpu_memory = ""
                    if device.type == "cuda":
                        gpu_memory = " gpu_mem=%.2fGB" % (torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                    logger.info(
                        "epoch=%d/%d step=%d/%d global_step=%d lr=%.8g grad_norm=%.5f step_time=%.3fs%s %s",
                        epoch + 1,
                        config["train"]["epochs"],
                        step + 1,
                        len(train_loader),
                        epoch * len(train_loader) + step + 1,
                        optimizer.param_groups[0]["lr"],
                        float(gradient_norm),
                        time.time() - step_started,
                        gpu_memory,
                        loss_text,
                    )
            should_validate = (epoch + 1) % config["train"]["validate_every"] == 0 or epoch + 1 == config["train"]["epochs"]
            if should_validate:
                validation_started = time.time()
                if rank == 0: logger.info("Validation started | epoch=%d", epoch + 1)
                metrics = evaluate(model, val_loader, config, device)
                if rank == 0:
                    logger.info("Validation finished | seconds=%.2f | metrics=%s", time.time() - validation_started, json.dumps(metrics, ensure_ascii=False))
                    if metrics["NDS"] > best:
                        best = metrics["NDS"]
                        best_path = output_dir / "checkpoints" / "best.pth"
                        save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best, config)
                        logger.info("New best checkpoint | NDS=%.6f | path=%s", best, best_path)
                    metrics_dir = output_dir / "metrics"; metrics_dir.mkdir(exist_ok=True)
                    with (metrics_dir / "val.json").open("w", encoding="utf-8") as handle: json.dump(metrics, handle, ensure_ascii=False, indent=2)
            if rank == 0:
                last_path = output_dir / "checkpoints" / "last.pth"
                save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best, config)
                logger.info("Epoch %d finished | seconds=%.2f | last_checkpoint=%s", epoch + 1, time.time() - epoch_started, last_path)
        if rank == 0:
            logger.info("Training completed | seconds=%.2f | best_NDS=%.6f", time.time() - training_started, best)
    finally:
        if distributed.is_initialized(): distributed.destroy_process_group()


def evaluate_checkpoint(config, checkpoint_path, split):
    device = torch.device("cuda" if config["runtime"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device); load_checkpoint(checkpoint_path, model, strict=True)
    loader = build_loader(config, split, max_samples=config["runtime"].get("max_%s_samples" % split))
    metrics = evaluate(model, loader, config, device)
    output = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"] / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    with (output / (split + ".json")).open("w", encoding="utf-8") as handle: json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
