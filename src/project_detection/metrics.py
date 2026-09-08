from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


def _average_precision(tp, fp, positives):
    if positives == 0: return float("nan")
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recall = tp / positives; precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]))


class Mw3dMetric:
    """Camera-BEV center-distance metric with multi-threshold AP and TP errors."""

    def __init__(self, class_names, distance_thresholds=(0.5, 1, 2, 4), tp_threshold=2.0):
        self.class_names = class_names
        self.distance_thresholds = list(distance_thresholds)
        self.tp_threshold = tp_threshold
        self.samples = []

    def update(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            self.samples.append((
                prediction["boxes3d"].detach().cpu().numpy(),
                prediction["scores"].detach().cpu().numpy(),
                prediction["labels"].detach().cpu().numpy(),
                target["boxes3d"].detach().cpu().numpy(),
                target["labels"].detach().cpu().numpy(),
            ))

    def state_dict(self): return self.samples
    def load_state_dict(self, state): self.samples = list(state)

    def _class_at_threshold(self, class_id, threshold):
        detections, gt_by_sample = [], {}
        positives = 0
        for sample_id, (boxes, scores, labels, gt_boxes, gt_labels) in enumerate(self.samples):
            gt_indices = np.where(gt_labels == class_id)[0]
            gt_by_sample[sample_id] = {int(index): False for index in gt_indices}
            positives += len(gt_indices)
            for index in np.where(labels == class_id)[0]:
                detections.append((float(scores[index]), sample_id, boxes[index]))
        detections.sort(key=lambda item: item[0], reverse=True)
        tp, fp, errors = [], [], []
        for _, sample_id, box in detections:
            gt_boxes = self.samples[sample_id][3]
            candidates = [idx for idx, used in gt_by_sample[sample_id].items() if not used]
            if candidates:
                distances = [np.linalg.norm(box[[0, 2]] - gt_boxes[idx][[0, 2]]) for idx in candidates]
                best_pos = int(np.argmin(distances)); best_idx = candidates[best_pos]
            else:
                distances, best_pos, best_idx = [], -1, -1
            if candidates and distances[best_pos] <= threshold:
                gt_by_sample[sample_id][best_idx] = True; tp.append(1); fp.append(0)
                gt = gt_boxes[best_idx]
                scale_iou = np.prod(np.minimum(box[3:6], gt[3:6])) / max(np.prod(np.maximum(box[3:6], gt[3:6])), 1e-6)
                angle = abs((float(box[6] - gt[6]) + math.pi) % (2*math.pi) - math.pi)
                errors.append((distances[best_pos], 1-scale_iou, angle))
            else:
                tp.append(0); fp.append(1)
        return _average_precision(np.asarray(tp), np.asarray(fp), positives), errors

    def compute(self):
        aps, tp_errors = [], []
        per_class = {}
        for class_id, name in enumerate(self.class_names):
            class_aps = []
            for threshold in self.distance_thresholds:
                ap, errors = self._class_at_threshold(class_id, threshold)
                if not np.isnan(ap): class_aps.append(ap)
                if threshold == self.tp_threshold: tp_errors.extend(errors)
            if class_aps:
                per_class[name] = float(np.mean(class_aps)); aps.extend(class_aps)
        mean_ap = float(np.mean(aps)) if aps else 0.0
        if tp_errors:
            mean_errors = np.mean(np.asarray(tp_errors), axis=0)
            mate, mase, maoe = map(float, mean_errors)
        else:
            mate = mase = maoe = float("nan")
        tp_scores = [max(0.0, 1.0-value) if np.isfinite(value) else 0.0 for value in (mate, mase, maoe)]
        nds = (5.0 * mean_ap + sum(tp_scores)) / 8.0
        return {"NDS": nds, "mAP": mean_ap, "mATE": mate, "mASE": mase, "mAOE": maoe, "per_class_AP": per_class}
