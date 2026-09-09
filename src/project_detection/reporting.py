from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _horizontal_bars(values, title, output_path, width=1200):
    labels = list(values)
    canvas_height = max(360, 85 + 38 * len(labels))
    canvas = np.full((canvas_height, width, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    label_width, chart_left, chart_right = 245, 270, width - 55
    chart_width = chart_right - chart_left
    for index, label in enumerate(labels):
        value = float(values[label])
        y = 75 + index * 38
        cv2.putText(canvas, str(label), (20, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (chart_left, y), (chart_right, y + 22), (225, 225, 225), -1)
        end = chart_left + int(np.clip(value, 0.0, 1.0) * chart_width)
        cv2.rectangle(canvas, (chart_left, y), (end, y + 22), (65, 145, 235), -1)
        cv2.putText(canvas, "%.4f" % value, (chart_right + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def _map_matrix(metrics, output_path):
    matrix = metrics.get("per_class_AP_by_distance", {})
    classes = list(matrix)
    thresholds = list(metrics.get("mAP_by_distance", {}))
    cell_w, cell_h = 125, 42
    left, top = 245, 80
    canvas = np.full(
        (top + cell_h * len(classes) + 40, left + cell_w * len(thresholds) + 35, 3),
        250,
        dtype=np.uint8,
    )
    cv2.putText(canvas, "AP by class and center-distance threshold", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
    for column, threshold in enumerate(thresholds):
        cv2.putText(canvas, "%sm" % threshold, (left + column * cell_w + 40, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    for row, class_name in enumerate(classes):
        y = top + row * cell_h
        cv2.putText(canvas, class_name, (15, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
        for column, threshold in enumerate(thresholds):
            value = matrix[class_name].get(threshold)
            score = 0.0 if value is None else float(value)
            color = cv2.applyColorMap(np.uint8([[int(np.clip(score, 0, 1) * 255)]]), cv2.COLORMAP_VIRIDIS)[0, 0]
            x = left + column * cell_w
            cv2.rectangle(canvas, (x, y), (x + cell_w - 3, y + cell_h - 3), tuple(int(item) for item in color), -1)
            text = "N/A" if value is None else "%.3f" % score
            cv2.putText(canvas, text, (x + 37, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def write_test_report(metrics, metric_path, plot_dir, metadata):
    metric_path, plot_dir = Path(metric_path), Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_values = {
        "NDS": metrics["NDS"],
        "mAP": metrics["mAP"],
        "mRecall@2m": metrics.get("mRecall", 0.0),
        "mPrecision@2m": metrics.get("mPrecision", 0.0),
        "F1@2m": metrics.get("F1", 0.0),
    }
    files = {
        "summary": [str(plot_dir / "summary_bars.json"), str(plot_dir / "summary_bars.png")],
        "per_class_ap": [str(plot_dir / "per_class_ap.json"), str(plot_dir / "per_class_ap.png")],
        "map_matrix": [str(plot_dir / "map_matrix.json"), str(plot_dir / "map_matrix.png")],
    }
    _write_json(plot_dir / "summary_bars.json", summary_values)
    _horizontal_bars(summary_values, "MW3D test summary", plot_dir / "summary_bars.png")
    _write_json(plot_dir / "per_class_ap.json", metrics.get("per_class_AP", {}))
    _horizontal_bars(metrics.get("per_class_AP", {}), "Per-class mean AP", plot_dir / "per_class_ap.png")
    map_payload = {
        "class_names": list(metrics.get("per_class_AP_by_distance", {})),
        "distance_thresholds": list(metrics.get("mAP_by_distance", {})),
        "ap": metrics.get("per_class_AP_by_distance", {}),
    }
    _write_json(plot_dir / "map_matrix.json", map_payload)
    _map_matrix(metrics, plot_dir / "map_matrix.png")
    plot_index = {"backend": "opencv", "plot_dir": str(plot_dir), "files": files}
    _write_json(plot_dir / "plots_index.json", plot_index)
    report = {
        "protocol": "mw3d_camera_bev_center_distance",
        "note": "NuScenes-style AP/TP/NDS formulas on MW3D camera BEV; not leaderboard-comparable.",
        "split": "test",
        "metadata": metadata,
        "metrics": metrics,
        "plot_dir": str(plot_dir),
        "plots": files,
    }
    _write_json(metric_path, report)
    return report
