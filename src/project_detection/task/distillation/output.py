"""Branch-corresponding depth output distillation."""

import torch
import torch.nn.functional as functional


def flatten_prediction(tensor):
    return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, tensor.shape[1])


def depth_output_losses(student, teacher, regions, student_head, teacher_head, options):
    zero = student[0]["cls"].float().sum() * 0
    losses = {"depth_distribution": zero, "depth_regression": zero}
    stats = {key: zero.detach().clone() for key in (
        "quality_sum", "teacher_error_sum", "student_error_sum", "teacher_better_sum",
        "quality_count",
    )}
    branches = ("parallel", "chain") if student_head.attribute_prediction_mode == "adaptive" else (None,)
    for s_level, t_level, region in zip(student, teacher, regions):
        for branch in branches:
            s = s_level[branch] if branch else s_level
            t = t_level[branch] if branch else t_level
            sb, tb = flatten_prediction(s["bbox"]), flatten_prediction(t["bbox"])
            sl, tl = flatten_prediction(s["depth_logits"]), flatten_prediction(t["depth_logits"])
            for image, (mask, gt_depth, center, _) in enumerate(region):
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
                if distribution.get("enabled", True):
                    temperature = distribution.get("temperature", 1.0)
                    kl = functional.kl_div(
                        functional.log_softmax(sp / temperature, dim=1),
                        functional.softmax(tp / temperature, dim=1), reduction="none",
                    ).sum(1) * temperature ** 2
                    losses["depth_distribution"] = losses["depth_distribution"] + (kl * weights).sum() / divisor
                regression = options.get("depth_regression", {})
                if regression.get("enabled", True):
                    residual = functional.smooth_l1_loss(
                        sd, td, beta=regression.get("beta", 1.0), reduction="none"
                    )
                    losses["depth_regression"] = losses["depth_regression"] + (residual * weights).sum() / divisor
                for name, value in {
                    "quality_sum": quality.sum(), "teacher_error_sum": te.sum(),
                    "student_error_sum": se.sum(), "teacher_better_sum": (te < se).float().sum(),
                    "quality_count": te.new_tensor(te.numel()),
                }.items():
                    stats[name] += value.detach() / divisor
    return losses, stats
