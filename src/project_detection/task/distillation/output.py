"""Foreground output distillation; shared outputs are evaluated only once."""

import torch
import torch.nn.functional as functional


DEPTH_OUTPUTS = ("depth_distribution", "depth_regression")
SHARED_OUTPUTS = {"classification": "cls", "attribute": "attribute",
                  "centerness": "centerness", "bbox2d": "bbox2d"}
BRANCH_OUTPUTS = {"center_offset": "bbox", "dimensions": "bbox",
                  "orientation": "bbox", "direction": "direction",
                  "geometric_weight": "geo_weight"}
OUTPUT_NAMES = DEPTH_OUTPUTS + tuple(SHARED_OUTPUTS) + tuple(BRANCH_OUTPUTS)


def output_enabled(options, name):
    # Old depth-only configurations retain their original behavior.
    return options.get(name, {}).get("enabled", name in DEPTH_OUTPUTS)


def _zero(outputs):
    return torch.nan_to_num(outputs[0]["cls"].float(), nan=0., posinf=0., neginf=0.).sum() * 0


def _output(prediction, key):
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError("Enabled distillation output is missing: %s" % key)
    return flatten_prediction(value).float()


def _bernoulli_kl(student, teacher, temperature):
    student, teacher = student / temperature, teacher / temperature
    probability = teacher.sigmoid()
    return (probability * (functional.logsigmoid(teacher) - functional.logsigmoid(student))
            + (1 - probability) * (functional.logsigmoid(-teacher)
                                   - functional.logsigmoid(-student))).mean(1) * temperature ** 2


def output_losses(student, teacher, regions, student_head, teacher_head, options):
    losses, stats = depth_output_losses(student, teacher, regions, student_head,
                                       teacher_head, options)
    branches = ("parallel", "chain") if student_head.attribute_prediction_mode == "adaptive" else (None,)
    for name, key in dict(SHARED_OUTPUTS, **BRANCH_OUTPUTS).items():
        if not output_enabled(options, name):
            continue
        value, count = _zero(student), _zero(student).detach().clone()
        selected_branches = (None,) if name in SHARED_OUTPUTS else branches
        settings = options.get(name, {})
        for s_level, t_level, level_regions in zip(student, teacher, regions):
            for branch in selected_branches:
                sp = _output(s_level[branch] if branch else s_level, key)
                tp = _output(t_level[branch] if branch else t_level, key).detach()
                if sp.shape != tp.shape:
                    raise ValueError("Teacher/student shape mismatch: %s" % name)
                if name == "center_offset":
                    sp, tp = sp[..., :2], tp[..., :2]
                elif name == "dimensions":
                    sp, tp = sp[..., 3:6], tp[..., 3:6]
                elif name == "orientation":
                    sp, tp = sp[..., 6:7], tp[..., 6:7]
                for image, region in enumerate(level_regions):
                    mask, _, center, _ = region[:4]
                    if name == "bbox2d":
                        if not hasattr(region, "bbox2d_valid"):
                            raise ValueError("bbox2d distillation requires inside-GT region metadata")
                        mask = mask & region.bbox2d_valid
                    # Filter BEFORE applying nonlinearities/losses, not by multiplying NaNs by zero.
                    s, t = sp[image][mask], tp[image][mask]
                    valid = torch.isfinite(s).all(1) & torch.isfinite(t).all(1)
                    s, t = s[valid], t[valid]
                    weight = center[mask][valid].float()
                    if name in ("classification", "centerness", "geometric_weight"):
                        error = _bernoulli_kl(s, t, settings.get("temperature", 1.0))
                        if name in ("classification", "centerness"):
                            weight = torch.ones_like(weight)
                    elif name in ("attribute", "direction"):
                        temperature = settings.get("temperature", 1.0)
                        error = functional.kl_div(
                            functional.log_softmax(s / temperature, 1),
                            functional.softmax(t / temperature, 1), reduction="none",
                        ).sum(1) * temperature ** 2
                        if name == "attribute":
                            weight = torch.ones_like(weight)
                    else:
                        if name == "orientation":
                            s, t = s.sin() * t.cos(), s.cos() * t.sin()
                        error = functional.smooth_l1_loss(
                            s, t, beta=settings.get("beta", 1.0 / 9.0), reduction="none",
                        ).mean(1)
                    divisor = float(len(selected_branches))
                    value = value + (error * weight).sum() / divisor
                    count = count + valid.sum().detach() / divisor
        losses[name] = value
        stats["loss_kd_output_%s_count" % name] = count
    return losses, stats


def flatten_prediction(tensor):
    return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, tensor.shape[1])


def depth_output_losses(student, teacher, regions, student_head, teacher_head, options):
    zero = _zero(student)
    losses = {name: zero for name in DEPTH_OUTPUTS if output_enabled(options, name)}
    stats = {key: zero.detach().clone() for key in (
        "quality_sum", "teacher_error_sum", "student_error_sum", "teacher_better_sum",
        "quality_count",
    )}
    for name in losses:
        stats["loss_kd_output_%s_count" % name] = zero.detach().clone()
    if not losses:
        return losses, stats
    branches = ("parallel", "chain") if student_head.attribute_prediction_mode == "adaptive" else (None,)
    for s_level, t_level, region in zip(student, teacher, regions):
        for branch in branches:
            s = s_level[branch] if branch else s_level
            t = t_level[branch] if branch else t_level
            sb, tb = _output(s, "bbox"), _output(t, "bbox").detach()
            sl, tl = _output(s, "depth_logits"), _output(t, "depth_logits").detach()
            if sb.shape != tb.shape or sl.shape != tl.shape:
                raise ValueError("Teacher/student shape mismatch: depth")
            for image, entry in enumerate(region):
                mask, gt_depth, center, _ = entry[:4]
                sp, tp = sl[image][mask].float(), tl[image][mask].float()
                sd, _ = student_head.decode_depth_candidates(sb[image][mask, 2].float(), sp)
                td, _ = teacher_head.decode_depth_candidates(tb[image][mask, 2].float(), tp)
                gt = gt_depth[mask].float()
                valid = torch.isfinite(td) & torch.isfinite(sd) & torch.isfinite(tp).all(1)
                valid &= torch.isfinite(sp).all(1) & torch.isfinite(gt) & (gt > 0)
                # Index invalid values BEFORE the loss; zero * NaN is still NaN.
                sd, td, gt, sp, tp = sd[valid], td[valid], gt[valid], sp[valid], tp[valid]
                te, se = (td - gt).abs(), (sd.detach() - gt).abs()
                quality_options = options.get("quality", {})
                quality = torch.ones_like(te)
                if quality_options.get("enabled", True):
                    quality = torch.exp(-te / gt.clamp_min(1e-6)
                                        / quality_options.get("relative_error_scale", 0.1))
                weights = center[mask][valid].float() * quality
                divisor = float(len(branches))
                distribution = options.get("depth_distribution", {})
                if output_enabled(options, "depth_distribution"):
                    temperature = distribution.get("temperature", 1.0)
                    kl = functional.kl_div(
                        functional.log_softmax(sp / temperature, dim=1),
                        functional.softmax(tp / temperature, dim=1), reduction="none",
                    ).sum(1) * temperature ** 2
                    losses["depth_distribution"] = losses["depth_distribution"] + (kl * weights).sum() / divisor
                regression = options.get("depth_regression", {})
                if output_enabled(options, "depth_regression"):
                    residual = functional.smooth_l1_loss(
                        sd, td, beta=regression.get("beta", 1.0), reduction="none"
                    )
                    losses["depth_regression"] = losses["depth_regression"] + (residual * weights).sum() / divisor
                for name in losses:
                    stats["loss_kd_output_%s_count" % name] += te.new_tensor(te.numel()) / divisor
                for name, value in {
                    "quality_sum": quality.sum(), "teacher_error_sum": te.sum(),
                    "student_error_sum": se.sum(), "teacher_better_sum": (te < se).float().sum(),
                    "quality_count": te.new_tensor(te.numel()),
                }.items():
                    stats[name] += value.detach() / divisor
    return losses, stats
