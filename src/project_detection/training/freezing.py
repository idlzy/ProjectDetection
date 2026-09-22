from __future__ import annotations


def freeze_policy(config):
    freeze = config.get("train", {}).get("freeze", {})
    return {
        "enabled": freeze.get("enabled", False),
        "modules": list(freeze.get("modules", ["backbone"])),
        "epochs": int(freeze.get("epochs", 0)),
        "backbone_lr_multiplier": float(
            freeze.get("backbone_lr_multiplier", 0.1)
        ),
    }


def freeze_state_for_checkpoint(config, epoch):
    policy = freeze_policy(config)
    return {
        **policy,
        "epoch": int(epoch),
        "frozen": bool(policy["enabled"] and epoch < policy["epochs"]),
    }


def apply_freeze_schedule(model, config, epoch):
    policy = freeze_policy(config)
    frozen = bool(policy["enabled"] and epoch < policy["epochs"])
    backbone = model.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad_(not frozen)
    if frozen:
        backbone.eval()
    else:
        backbone.train()
    return {
        **policy,
        "epoch": int(epoch),
        "frozen": frozen,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def build_optimizer_parameters(model, config):
    policy = freeze_policy(config)
    if not policy["enabled"]:
        return model.parameters()

    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    base_learning_rate = float(config["train"]["learning_rate"])
    return [
        {
            "name": "main",
            "params": other_parameters,
            "lr": base_learning_rate,
        },
        {
            "name": "backbone",
            "params": backbone_parameters,
            "lr": base_learning_rate
            * policy["backbone_lr_multiplier"],
        },
    ]


def _checkpoint_state(checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def backbone_pretrain_coverage(model, checkpoint):
    checkpoint_state = _checkpoint_state(checkpoint)
    total_elements = 0
    loaded_elements = 0
    missing = []
    mismatched = []
    for key, value in model.backbone.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        total_elements += value.numel()
        candidate = checkpoint_state.get("backbone." + key)
        if candidate is None:
            missing.append(key)
        elif tuple(candidate.shape) != tuple(value.shape):
            mismatched.append(key)
        else:
            loaded_elements += value.numel()
    ratio = loaded_elements / max(total_elements, 1)
    return {
        "loaded_elements": loaded_elements,
        "total_elements": total_elements,
        "element_ratio": ratio,
        "missing": missing,
        "mismatched": mismatched,
    }


def validate_resume_freeze_policy(checkpoint, config):
    saved_config = checkpoint.get("config")
    current = freeze_policy(config)
    if saved_config is None:
        if current["enabled"]:
            raise RuntimeError(
                "freeze-enabled resume requires a checkpoint with saved config; "
                "use it as train.pretrain for a new experiment instead"
            )
        return
    saved = freeze_policy(saved_config)
    if saved != current:
        raise RuntimeError(
            "resume freeze policy differs from the checkpoint: saved=%s current=%s; "
            "start a new warm-start experiment with train.pretrain instead"
            % (saved, current)
        )
