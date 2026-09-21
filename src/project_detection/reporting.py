from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#3972D5"
CYAN = "#27A7B8"
ORANGE = "#F29E4C"
GREEN = "#3BA272"
RED = "#D85A5A"
PURPLE = "#7568B5"
GRID = "#DCE3EC"
TEXT = "#263445"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#C7D0DC",
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": "#536273",
    "ytick.color": "#536273",
    "font.size": 10,
    "axes.titleweight": "bold",
})


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _finish_figure(figure, output_path):
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _decorate_axis(axis, grid_axis="x"):
    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _horizontal_bars(values, title, output_path):
    labels = list(values)
    numeric = np.asarray([
        float(values[label]) if values[label] is not None else np.nan
        for label in labels
    ])
    figure, axis = plt.subplots(figsize=(10.5, max(3.4, 0.42 * len(labels) + 1.5)))
    positions = np.arange(len(labels))
    widths = np.nan_to_num(numeric, nan=0.0)
    bars = axis.barh(positions, widths, color=BLUE, alpha=0.9, height=0.62)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(1.0, float(np.nanmax(numeric, initial=1.0)) * 1.12))
    axis.set_xlabel("Score")
    axis.set_title(title, loc="left", pad=14)
    _decorate_axis(axis)
    for bar, value in zip(bars, numeric):
        text = "N/A" if not np.isfinite(value) else f"{value:.4f}"
        axis.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            color=TEXT,
            fontsize=9,
        )
    _finish_figure(figure, output_path)


def _error_bars(values, title, output_path):
    labels = list(values)
    numeric = np.asarray([
        float(values[label]) if values[label] is not None else np.nan
        for label in labels
    ])
    figure, axis = plt.subplots(figsize=(9.5, max(3.2, 0.6 * len(labels) + 1.4)))
    positions = np.arange(len(labels))
    widths = np.nan_to_num(numeric, nan=0.0)
    bars = axis.barh(positions, widths, color=[ORANGE, RED, PURPLE][:len(labels)])
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    maximum = max(float(np.nanmax(numeric, initial=1.0)), 1e-6)
    axis.set_xlim(0.0, maximum * 1.18)
    axis.set_xlabel("Error (lower is better)")
    axis.set_title(title, loc="left", pad=14)
    _decorate_axis(axis)
    for bar, value in zip(bars, numeric):
        text = "N/A" if not np.isfinite(value) else f"{value:.4f}"
        axis.text(bar.get_width() + maximum * 0.018, bar.get_y() + bar.get_height() / 2,
                  text, va="center", color=TEXT, fontsize=9)
    _finish_figure(figure, output_path)


def _map_matrix(metrics, output_path):
    matrix = metrics.get("per_class_AP_by_distance", {})
    classes = list(matrix)
    thresholds = list(metrics.get("mAP_by_distance", {}))
    values = np.asarray([
        [np.nan if matrix[name].get(threshold) is None else matrix[name][threshold]
         for threshold in thresholds]
        for name in classes
    ], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8.8, max(4.5, 0.48 * len(classes) + 1.8)))
    image = axis.imshow(np.nan_to_num(values), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(thresholds)), [f"{item} m" for item in thresholds])
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_xlabel("Center-distance threshold")
    axis.set_title("AP by class and distance threshold", loc="left", pad=14)
    for row in range(len(classes)):
        for column in range(len(thresholds)):
            value = values[row, column]
            axis.text(column, row, "N/A" if not np.isfinite(value) else f"{value:.3f}",
                      ha="center", va="center", color="white", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="AP")
    _finish_figure(figure, output_path)


def _confusion_matrix_plot(payload, output_path):
    labels = payload.get("labels", [])
    counts = np.asarray(payload.get("counts", []), dtype=np.int64)
    normalized = np.asarray(payload.get("normalized_by_gt", []), dtype=np.float64)
    size = max(8.5, 0.72 * len(labels))
    figure, axis = plt.subplots(figsize=(size + 1.2, size))
    if counts.ndim != 2 or counts.size == 0:
        axis.text(0.5, 0.5, "Confusion-matrix data unavailable", ha="center", va="center",
                  transform=axis.transAxes, color=TEXT, fontsize=13)
        axis.set_axis_off()
        _finish_figure(figure, output_path)
        return
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=48, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    axis.set_title(
        f"Detection confusion matrix · < {payload.get('distance_threshold', 2.0):g} m",
        loc="left", pad=14,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            ratio = normalized[row, column]
            if counts[row, column] == 0 and ratio == 0:
                continue
            color = "white" if ratio >= 0.48 else TEXT
            axis.text(column, row, f"{ratio:.0%}", ha="center", va="center",
                      color=color, fontsize=7.2, fontweight="medium")
    figure.colorbar(image, ax=axis, fraction=0.038, pad=0.025, label="Row-normalized rate")
    _finish_figure(figure, output_path)


def _pr_curves(metrics, output_path):
    curves = metrics.get("per_class_PR", {})
    threshold = float(metrics.get("tp_distance_threshold", 2.0))
    threshold_key = str(threshold)
    aps = metrics.get("per_class_AP_by_distance", {})
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    valid_precision = []
    palette = (
        "#0057B8", "#FF7A00", "#D7263D", "#00A6A6", "#2E8B57",
        "#C9A000", "#7B2CBF", "#FF4FA3", "#7F4F24", "#4D4D4D",
        "#00B4D8", "#AACC00",
    )
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p")
    valid_index = 0
    for class_name, curve in curves.items():
        class_ap = aps.get(class_name, {}).get(threshold_key)
        if class_ap is None:
            continue
        recall = np.asarray(curve["recall"], dtype=np.float64)
        precision = np.asarray(curve["precision"], dtype=np.float64)
        valid_precision.append(precision)
        axis.plot(
            recall, precision, linewidth=1.8, alpha=0.92,
            color=palette[valid_index % len(palette)],
            marker=markers[valid_index % len(markers)], markevery=10, markersize=3.8,
            label=f"{class_name} · AP@{threshold:g}m {class_ap:.3f}",
        )
        valid_index += 1
    if valid_precision:
        macro_precision = np.mean(np.stack(valid_precision), axis=0)
        axis.plot(np.linspace(0, 1, len(macro_precision)), macro_precision,
                  color="#101820", linewidth=3.0, label="Macro-average precision")
    axis.scatter([metrics.get("mRecall", 0.0)], [metrics.get("mPrecision", 0.0)],
                 s=85, color=RED, edgecolor="white", linewidth=1.4,
                 zorder=5, label="Evaluation operating point")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        f"Per-class precision–recall curves · {threshold:g} m",
        loc="left", pad=14,
    )
    _decorate_axis(axis, "both")
    axis.legend(loc="upper right", fontsize=7.4, frameon=True, ncol=2)
    _finish_figure(figure, output_path)


def _ap_by_distance_bars(metrics, output_path):
    matrix = metrics.get("per_class_AP_by_distance", {})
    thresholds = list(metrics.get("mAP_by_distance", {}))
    class_names = list(matrix)
    positions = np.arange(len(class_names), dtype=np.float64)
    colors = ("#0057B8", "#FF8C00", "#00A676", "#D7263D")
    bar_height = min(0.18, 0.78 / max(len(thresholds), 1))
    figure, axis = plt.subplots(figsize=(12.5, max(5.2, 0.62 * len(class_names) + 1.8)))
    for index, threshold in enumerate(thresholds):
        offset = (index - (len(thresholds) - 1) / 2.0) * bar_height
        raw_values = [matrix[name].get(threshold) for name in class_names]
        values = [0.0 if value is None else float(value) for value in raw_values]
        bars = axis.barh(
            positions + offset, values, height=bar_height * 0.88,
            color=colors[index % len(colors)], label=f"{threshold} m",
        )
        for bar, raw_value in zip(bars, raw_values):
            if raw_value is None:
                continue
            axis.text(
                bar.get_width() + 0.007,
                bar.get_y() + bar.get_height() / 2,
                f"{float(raw_value):.3f}",
                va="center", ha="left", fontsize=7.2, color=TEXT,
            )
    axis.set_yticks(positions, class_names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Average precision")
    axis.set_title("Per-class AP across center-distance thresholds", loc="left", pad=14)
    axis.legend(title="Match threshold", loc="lower right", ncol=len(thresholds))
    _decorate_axis(axis)
    _finish_figure(figure, output_path)


def _performance_overview(metrics, output_path):
    per_class = metrics.get("per_class_AP", {})
    depth_recall = metrics.get("recall_by_depth_bin", {})
    threshold_recall = metrics.get("recall_by_distance", {})
    depth_counts = metrics.get("depth_bin_counts", {})
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.45, 1, 1]})

    class_names = list(per_class)
    class_ap = np.asarray([np.nan if per_class[name] is None else per_class[name] for name in class_names])
    positions = np.arange(len(class_names))
    axes[0].barh(positions, np.nan_to_num(class_ap), color=BLUE, height=0.65)
    axes[0].set_yticks(positions, class_names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Mean AP")
    axes[0].set_title("Average AP by class", loc="left")
    for y, value in enumerate(class_ap):
        axes[0].text(0.012 if not np.isfinite(value) else value + 0.012, y,
                     "N/A" if not np.isfinite(value) else f"{value:.3f}", va="center", fontsize=8)
    _decorate_axis(axes[0])

    depth_labels = list(depth_recall)
    depth_values = [depth_recall[label] for label in depth_labels]
    bars = axes[1].bar(np.arange(len(depth_labels)), depth_values, color=CYAN, width=0.68)
    axes[1].set_xticks(np.arange(len(depth_labels)), depth_labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall by GT depth bin", loc="left")
    for bar, label, value in zip(bars, depth_labels, depth_values):
        counts = depth_counts.get(label, {})
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                     f"{value:.2f}\n{counts.get('tp', 0)}/{counts.get('gt', 0)}",
                     ha="center", va="bottom", fontsize=7.5)
    _decorate_axis(axes[1], "y")

    threshold_labels = list(threshold_recall)
    threshold_values = [threshold_recall[label] for label in threshold_labels]
    x_values = np.arange(len(threshold_labels))
    axes[2].plot(x_values, threshold_values, color=GREEN, marker="o", markersize=8, linewidth=2.5)
    axes[2].fill_between(x_values, threshold_values, color=GREEN, alpha=0.12)
    axes[2].set_xticks(x_values, [f"{label} m" for label in threshold_labels])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Recall by match threshold", loc="left")
    for x, value in zip(x_values, threshold_values):
        axes[2].text(x, value + 0.035, f"{value:.3f}", ha="center", fontsize=9)
    _decorate_axis(axes[2], "y")

    figure.suptitle("MW3D detection performance overview", x=0.03, ha="left", fontsize=17,
                    fontweight="bold", color=TEXT)
    figure.subplots_adjust(top=0.84, wspace=0.33)
    _finish_figure(figure, output_path)


def _class_diagnostics(metrics, output_path):
    diagnostics = metrics.get("per_class_detection", {})
    labels = list(diagnostics)
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, max(4.5, 0.48 * len(labels) + 1.7)))
    height = 0.23
    for offset, key, color in ((-height, "AP", BLUE), (0, "recall", GREEN), (height, "precision", ORANGE)):
        values = [0.0 if diagnostics[label].get(key) is None else diagnostics[label].get(key, 0.0)
                  for label in labels]
        axis.barh(positions + offset, values, height=height, label=key, color=color)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title("Class-level AP, recall and precision", loc="left", pad=14)
    axis.legend(loc="lower right", ncol=3)
    _decorate_axis(axis)
    _finish_figure(figure, output_path)


def _confidence_curve(thresholds, values, title, output_path, color=BLUE):
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(thresholds, values, color=color, linewidth=1.8)
    axis.fill_between(thresholds, values, color=color, alpha=0.14)
    axis.set_xlim(0.01, 1.0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Confidence bin upper bound")
    axis.set_ylabel("Predicted bbox count per 0.01 bin")
    axis.set_title(title, loc="left", pad=14)
    _decorate_axis(axis, grid_axis="both")
    _finish_figure(figure, output_path)


def _safe_plot_name(value):
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return name or "class"


def _write_confidence_distribution(metrics, plot_dir):
    payload = metrics.get("confidence_distribution")
    if not payload:
        return None
    output_dir = Path(plot_dir) / "confidence_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "distribution.json", payload)
    thresholds = payload["bin_upper_bounds"]
    generated = [str(output_dir / "distribution.json")]
    total_path = output_dir / "total.png"
    _confidence_curve(
        thresholds, payload["total"]["bin_counts"],
        "All classes · confidence distribution", total_path,
    )
    generated.append(str(total_path))
    colors = (ORANGE, GREEN, RED, PURPLE, CYAN, BLUE)
    for index, (class_name, distribution) in enumerate(payload["per_class"].items()):
        output_path = output_dir / ("class_%s.png" % _safe_plot_name(class_name))
        _confidence_curve(
            thresholds, distribution["bin_counts"],
            "%s · confidence distribution" % class_name, output_path,
            colors[index % len(colors)],
        )
        generated.append(str(output_path))
    return generated


def write_test_report(metrics, metric_path, plot_dir, metadata):
    metric_path, plot_dir = Path(metric_path), Path(plot_dir)
    split = metadata.get("split", "test")
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_values = {
        "NDS": metrics["NDS"], "mAP": metrics["mAP"],
        "mRecall@2m": metrics.get("mRecall", 0.0),
        "mPrecision@2m": metrics.get("mPrecision", 0.0),
        "F1@2m": metrics.get("F1", 0.0),
    }
    tp_error_values = {
        "mATE (m)": metrics.get("mATE", float("nan")),
        "mASE": metrics.get("mASE", float("nan")),
        "mAOE (rad)": metrics.get("mAOE", float("nan")),
    }
    diagnostic_error_values = {"mADE (m)": metrics.get("mADE", float("nan"))}
    files = {
        "summary": [str(plot_dir / "summary_bars.json"), str(plot_dir / "summary_bars.png")],
        "tp_errors": [str(plot_dir / "tp_errors.json"), str(plot_dir / "tp_errors.png")],
        "diagnostic_errors": [str(plot_dir / "diagnostic_errors.json"), str(plot_dir / "diagnostic_errors.png")],
        "per_class_ap": [str(plot_dir / "per_class_ap.json"), str(plot_dir / "per_class_ap.png")],
        "map_matrix": [str(plot_dir / "map_matrix.json"), str(plot_dir / "map_matrix.png")],
        "ap_by_distance_bars": [str(plot_dir / "ap_by_distance_bars.json"), str(plot_dir / "ap_by_distance_bars.png")],
        "confusion_matrix": [str(plot_dir / "confusion_matrix.json"), str(plot_dir / "confusion_matrix.png")],
        "pr_curves": [str(plot_dir / "pr_curves.json"), str(plot_dir / "pr_curves.png")],
        "performance_overview": [str(plot_dir / "performance_overview.json"), str(plot_dir / "performance_overview.png")],
        "class_diagnostics": [str(plot_dir / "class_diagnostics.json"), str(plot_dir / "class_diagnostics.png")],
    }
    _write_json(plot_dir / "summary_bars.json", summary_values)
    _horizontal_bars(summary_values, "MW3D %s summary" % split, plot_dir / "summary_bars.png")
    _write_json(plot_dir / "tp_errors.json", tp_error_values)
    _error_bars(tp_error_values, "NDS true-positive errors · 3 terms", plot_dir / "tp_errors.png")
    _write_json(plot_dir / "diagnostic_errors.json", diagnostic_error_values)
    _error_bars(diagnostic_error_values, "MW3D diagnostic errors · not part of NDS", plot_dir / "diagnostic_errors.png")
    _write_json(plot_dir / "per_class_ap.json", metrics.get("per_class_AP", {}))
    _horizontal_bars(metrics.get("per_class_AP", {}), "Per-class mean AP", plot_dir / "per_class_ap.png")
    map_payload = {
        "class_names": list(metrics.get("per_class_AP_by_distance", {})),
        "distance_thresholds": list(metrics.get("mAP_by_distance", {})),
        "ap": metrics.get("per_class_AP_by_distance", {}),
    }
    _write_json(plot_dir / "map_matrix.json", map_payload)
    _map_matrix(metrics, plot_dir / "map_matrix.png")
    _write_json(plot_dir / "ap_by_distance_bars.json", map_payload)
    _ap_by_distance_bars(metrics, plot_dir / "ap_by_distance_bars.png")
    confusion = metrics.get("confusion_matrix", {})
    _write_json(plot_dir / "confusion_matrix.json", confusion)
    _confusion_matrix_plot(confusion, plot_dir / "confusion_matrix.png")
    _write_json(plot_dir / "pr_curves.json", metrics.get("per_class_PR", {}))
    _pr_curves(metrics, plot_dir / "pr_curves.png")
    overview = {
        "per_class_AP": metrics.get("per_class_AP", {}),
        "recall_by_depth_bin": metrics.get("recall_by_depth_bin", {}),
        "depth_bin_counts": metrics.get("depth_bin_counts", {}),
        "recall_by_distance": metrics.get("recall_by_distance", {}),
    }
    _write_json(plot_dir / "performance_overview.json", overview)
    _performance_overview(metrics, plot_dir / "performance_overview.png")
    _write_json(plot_dir / "class_diagnostics.json", metrics.get("per_class_detection", {}))
    _class_diagnostics(metrics, plot_dir / "class_diagnostics.png")
    confidence_files = _write_confidence_distribution(metrics, plot_dir)
    if confidence_files is not None:
        files["confidence_distribution"] = confidence_files
    plot_index = {"backend": "matplotlib", "plot_dir": str(plot_dir), "files": files}
    _write_json(plot_dir / "plots_index.json", plot_index)
    report_metadata = dict(metadata)
    report_metadata.update({
        "nds_variant": metrics.get("nds_variant", "nusc_cvpr2019_3tp"),
        "nds_tp_terms": metrics.get("nds_tp_terms", ["mATE", "mASE", "mAOE"]),
        "min_recall": metrics.get("min_recall", 0.1),
        "min_precision": metrics.get("min_precision", 0.1),
        "mean_ap_weight": metrics.get("mean_ap_weight", 5.0),
        "evaluated_classes": metrics.get("evaluated_classes", list(metrics.get("per_class_AP", {}))),
        "confusion_matrix_note": "Rows=GT, columns=prediction; background includes unmatched GT/predictions.",
    })
    report = {
        "protocol": "nusc_cvpr2019_formulas_on_mw3d_camera_bev",
        "note": (
            "MW3D data/classes evaluated in camera BEV [x,z] with NuScenes CVPR 2019 "
            "formulas. NDS uses ATE/ASE/AOE only; it is not directly comparable to "
            "the NuScenes leaderboard. mADE is diagnostic and is not an NDS term."
        ),
        "split": split, "metadata": report_metadata, "metrics": metrics,
        "plot_dir": str(plot_dir), "plots": files,
    }
    _write_json(metric_path, report)
    return report
