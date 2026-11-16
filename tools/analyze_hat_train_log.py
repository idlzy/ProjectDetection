#!/usr/bin/env python3
"""Analyze native HAT Epoch[n]/metric[value] training logs without HAT imports.

Epoch summaries take precedence over step losses. Only full (non-Step)
Validation summaries are used for model selection. Original epoch indices
are retained; plots default to the original zero-based epoch indices.
"""
import argparse
import json
import math
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH = re.compile(r"\bEpoch\[(\d+)\]")
NODE = re.compile(r"\bNode\[(\d+)\]")
STEP = re.compile(r"\bStep\[(\d+)\]")
PAIR = re.compile(r"\b([A-Za-z_]\w*(?:\.\d+)?)\[([^\[\]]+)\]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LR = re.compile(r"\blr=(" + NUMBER + r")")
COST = re.compile(r"\bCost Time:\s*(" + NUMBER + r")s")
BEST = re.compile(r"Save best model checkpoint:\s*(.+?)\s*$")


def numeric_pairs(text):
    values = {}
    for name, raw in PAIR.findall(text):
        try:
            value = float(raw)
        except ValueError:
            continue
        values[name] = value if math.isfinite(value) else None
    return values


def parse_hat_train_log(log_path, node=0, epoch_offset=0, metric="NDS", mode="auto"):
    steps, summaries, validation, runtime, learning_rates = {}, {}, {}, {}, []
    checkpoint_events, warnings = [], []
    last_validation_epoch = None
    step_loss_records = partial_validation_records = full_validation_records = 0
    with Path(log_path).open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = ANSI.sub("", raw)
            rank = NODE.search(line)
            if rank and int(rank.group(1)) != node:
                continue
            best = BEST.search(line)
            if best:
                checkpoint_events.append({"raw_epoch": last_validation_epoch,
                    "epoch": (last_validation_epoch + epoch_offset
                              if last_validation_epoch is not None else None),
                    "path": best.group(1), "line": line_number})
            match = EPOCH.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            body = line[match.end():]
            step = STEP.search(body)
            # A repeated epoch Begin indicates a resumed/repeated attempt.
            if re.match(r"\s+Begin\b", body):
                if epoch in summaries or epoch in steps or epoch in validation:
                    warnings.append("Repeated Epoch[%d] Begin: using the latest attempt" % epoch)
                    for mapping in (summaries, steps, validation, runtime):
                        mapping.pop(epoch, None)
                    learning_rates = [x for x in learning_rates if x["raw_epoch"] != epoch]
            if "Validation " in body:
                if step:
                    partial_validation_records += 1
                    continue
                payload = body.partition(":")[2]
                values = numeric_pairs(payload)
                if values:
                    full_validation_records += 1
                    validation[epoch] = values
                    last_validation_epoch = epoch
                continue
            losses = {k: v for k, v in numeric_pairs(body).items() if k.startswith("loss_")}
            if losses:
                if step:
                    step_loss_records += 1
                    steps[epoch] = losses
                else:
                    summaries[epoch] = losses
            lr = LR.search(body)
            if lr:
                global_step = re.search(r"\bGlobalStep\[(\d+)\]", body)
                learning_rates.append({"raw_epoch": epoch, "epoch": epoch + epoch_offset,
                    "step": int(step.group(1)) if step else None,
                    "global_step": int(global_step.group(1)) if global_step else None,
                    "lr": float(lr.group(1))})
            cost = COST.search(body)
            if cost and not re.search(r"\bStep\[", body):
                runtime[epoch] = float(cost.group(1))
    training = {}
    for epoch in sorted(set(steps) | set(summaries)):
        values = summaries.get(epoch, steps.get(epoch, {}))
        training[epoch] = {"raw_epoch": epoch, "epoch": epoch + epoch_offset,
            "source": "epoch_summary" if epoch in summaries else "last_logged_step",
            "losses": values}
    if not training and not validation:
        raise ValueError("No native HAT training/validation records found for Node[%d]" % node)
    train_loss, val_metrics = {}, {}
    for epoch, record in training.items():
        for name, value in record["losses"].items():
            train_loss.setdefault(name, []).append([epoch + epoch_offset, value])
    for epoch, values in sorted(validation.items()):
        for name, value in values.items():
            val_metrics.setdefault(name, []).append([epoch + epoch_offset, value])
    direction = mode
    if direction == "auto":
        direction = "min" if metric in ("mATE", "mASE", "mAOE", "mAVE", "mAAE", "mADE") else "max"
    candidates = [(epoch, values[metric]) for epoch, values in sorted(validation.items())
                  if values.get(metric) is not None]
    best_info = {"metric": metric, "mode": direction, "raw_epoch": None,
                 "epoch": None, "value": None, "source": "logged_full_validation"}
    if candidates:
        choose = max if direction == "max" else min
        epoch, value = choose(candidates, key=lambda x: x[1])
        best_info.update(raw_epoch=epoch, epoch=epoch+epoch_offset, value=value)
    else:
        warnings.append("No finite full-validation values for selection metric %s" % metric)
    if any(record["source"] == "last_logged_step" for record in training.values()):
        warnings.append("Epochs without loss summaries use the last logged step, not an epoch mean")
    return {"log": str(Path(log_path).resolve()), "node": node,
            "epoch_offset": epoch_offset, "training_epochs": list(training.values()),
            "validation_epochs": [{"raw_epoch": epoch, "epoch": epoch+epoch_offset, "metrics": values}
                                  for epoch, values in sorted(validation.items())],
            "train_loss": train_loss, "val_metrics": val_metrics,
            "epoch_seconds": [[epoch+epoch_offset, value] for epoch, value in sorted(runtime.items())],
            "learning_rates": learning_rates, "best_info": best_info,
            "best_checkpoint_events": checkpoint_events,
            "last_saved_best_checkpoint": checkpoint_events[-1] if checkpoint_events else None,
            "record_counts": {"step_losses": step_loss_records,
                              "full_validation": full_validation_records,
                              "ignored_partial_validation": partial_validation_records},
            "notes": ["Loss values are taken as logged; loss_total is not fabricated by summing components.",
                      "Best epoch uses rounded log values; ties choose the earliest epoch. "
                      "Actual saved-best checkpoints are reported separately.",
                      "Non-finite values are preserved as JSON null, never converted to zero."],
            "warnings": warnings}


def draw_report(report, output_dir, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)

    def curve(ax, series, name):
        # Keep None as NaN to show gaps instead of drawing across missing values.
        if any(value is not None for _, value in series):
            ax.plot([x for x, _ in series],
                    [float("nan") if y is None else y for _, y in series],
                    linewidth=1.3, label=name)

    def finish(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.25)
            ax.set_xlabel("Epoch" if report["epoch_offset"] == 0 else "Epoch (raw + 1)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(output_dir / name), dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for name, series in report["train_loss"].items():
        curve(axes[1] if name == "loss_depth" else axes[0], series, name)
    axes[0].set_title("Training losses (excluding loss_depth)")
    axes[1].set_title("Training loss_depth")
    for ax in axes:
        ax.set_ylabel("Logged loss")
    finish(fig, "loss_overview.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for name in ("NDS", "mAP", "mRecall", "mPrec", "F1"):
        curve(axes[0], report["val_metrics"].get(name, []), name)
    for name in ("mATE", "mASE", "mAOE", "mADE"):
        curve(axes[1], report["val_metrics"].get(name, []), name)
    axes[0].set_title("Full validation metrics (higher is better)")
    axes[1].set_title("Full validation errors (lower is better)")
    for ax in axes:
        if report["best_info"]["epoch"] is not None:
            best = report["best_info"]
            ax.axvline(best["epoch"], color="#c62828", linestyle="--",
                       label="Best %s @ %d: %.4f" % (best["metric"], best["epoch"], best["value"]))
        if not report["validation_epochs"]:
            ax.text(.5, .5, "No full validation summaries", transform=ax.transAxes, ha="center")
    finish(fig, "model_selection.png")

    classes = [name for name in report["val_metrics"] if name.startswith("AP_")]
    fig, axes = plt.subplots(max(1, (len(classes)+2)//3), 3,
                             figsize=(14, max(3, ((len(classes)+2)//3)*2.6)), squeeze=False)
    for ax, name in zip(axes.flat, classes):
        curve(ax, report["val_metrics"][name], name)
        ax.set_title(name[3:])
        ax.set_ylabel("AP")
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    finish(fig, "class_ap.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    curve(axes[0], report["epoch_seconds"], "training epoch seconds")
    axes[0].set_title("Training epoch duration (validation excluded)")
    axes[0].set_ylabel("Seconds")
    rates = report["learning_rates"]
    curve(axes[1], [[x["epoch"], x["lr"]] for x in rates], "logged learning rate")
    axes[1].set_title("Learning rate (logged precision)")
    axes[1].set_ylabel("LR")
    finish(fig, "training_runtime.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--epoch-offset", type=int, choices=(0, 1), default=0,
                        help="Display offset from raw epoch index (default: 0)")
    parser.add_argument("--metric", default="NDS")
    parser.add_argument("--mode", choices=("auto", "max", "min"), default="auto")
    parser.add_argument("--show", action="store_true", help="Show interactive figures (default headless)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = parse_hat_train_log(args.log, args.node, args.epoch_offset, args.metric, args.mode)
    output = args.output_dir or Path("outputs/train_log_analysis") / args.log.stem
    names = ("analysis.json", "loss_overview.png", "model_selection.png", "class_ap.png", "training_runtime.png")
    for name in names:
        if (output / name).exists() and not args.overwrite:
            raise FileExistsError("%s exists; use --overwrite" % (output / name))
    output.mkdir(parents=True, exist_ok=True)
    draw_report(report, output, args.show)
    (output / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output.resolve()),
                      "training_epochs": len(report["training_epochs"]),
                      "validation_epochs": len(report["validation_epochs"]),
                      "best_info": report["best_info"],
                      "record_counts": report["record_counts"],
                      "warnings": report["warnings"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
