from __future__ import annotations

import math

import numpy as np


PROTOCOL = "nusc_cvpr2019_formulas_on_mw3d_camera_bev"
NDS_VARIANT = "nusc_cvpr2019_3tp"
NDS_TP_TERMS = ("mATE", "mASE", "mAOE")
DEFAULT_DEPTH_BINS = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 1000000))


def _to_owned_numpy(tensor):
    """Copy a tensor into NumPy-owned memory.

    DataLoader worker tensors can be backed by shared-memory file descriptors.
    Keeping a zero-copy NumPy view alive would retain that descriptor for the
    complete evaluation epoch.
    """
    return tensor.detach().cpu().numpy().copy()


def _cummean(values):
    """Cumulative mean with the same empty/non-finite behavior as nuScenes."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.ones(1, dtype=np.float64)
    finite = np.isfinite(values)
    cumulative = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite.astype(np.float64))
    return cumulative / np.maximum(counts, 1.0)


def _angle_diff(predicted, target):
    """Smallest absolute yaw difference using a 2*pi period."""
    return abs((float(predicted) - float(target) + math.pi) % (2.0 * math.pi) - math.pi)


def _scale_iou_aligned(predicted_dims, target_dims):
    """Volume IoU after aligning box centers and orientations."""
    predicted = np.maximum(np.asarray(predicted_dims, dtype=np.float64), 0.0)
    target = np.maximum(np.asarray(target_dims, dtype=np.float64), 0.0)
    intersection = float(np.prod(np.minimum(predicted, target)))
    union = float(np.prod(predicted) + np.prod(target) - intersection)
    return intersection / union if union > 0.0 else 0.0


def _calc_ap(accumulated, min_recall=0.1, min_precision=0.1):
    """nuScenes CVPR 2019 AP from a 101-point interpolated precision curve."""
    if accumulated["num_gt"] == 0:
        return float("nan")
    first_index = int(round(100 * min_recall)) + 1
    precision = accumulated["precision"][first_index:].copy()
    if precision.size == 0:
        return 0.0
    precision = np.maximum(precision - min_precision, 0.0)
    return float(np.mean(precision) / (1.0 - min_precision))


def _calc_tp(accumulated, metric_name, min_recall=0.1):
    """nuScenes TP curve mean, returning maximal error if recall is too low."""
    first_index = int(round(100 * min_recall)) + 1
    last_index = accumulated["max_recall_index"]
    if last_index < first_index:
        return 1.0
    return float(np.mean(accumulated[metric_name][first_index:last_index + 1]))


class Mw3dMetric:
    """NuScenes-formula metrics on MW3D camera-coordinate BEV ``[x, z]``."""

    def __init__(
        self,
        class_names,
        distance_thresholds=(0.5, 1.0, 2.0, 4.0),
        tp_distance_threshold=2.0,
        min_recall=0.1,
        min_precision=0.1,
        mean_ap_weight=5.0,
        depth_bins=DEFAULT_DEPTH_BINS,
        evaluated_classes=None,
        tp_threshold=None,
    ):
        # ``tp_threshold`` remains accepted for callers using the old API.
        if tp_threshold is not None:
            tp_distance_threshold = tp_threshold
        self.all_class_names = list(class_names)
        self.class_names = (
            list(evaluated_classes)
            if evaluated_classes is not None else list(self.all_class_names)
        )
        if not self.class_names:
            raise ValueError("evaluated_classes must not be empty")
        unknown_classes = set(self.class_names) - set(self.all_class_names)
        if unknown_classes:
            raise ValueError(
                "Unknown evaluated classes: %s" % ", ".join(sorted(unknown_classes))
            )
        self.class_ids = [self.all_class_names.index(name) for name in self.class_names]
        self.distance_thresholds = [float(value) for value in distance_thresholds]
        self.tp_distance_threshold = float(tp_distance_threshold)
        self.tp_threshold = self.tp_distance_threshold
        self.min_recall = float(min_recall)
        self.min_precision = float(min_precision)
        self.mean_ap_weight = float(mean_ap_weight)
        self.depth_bins = [tuple(map(float, interval)) for interval in (depth_bins or DEFAULT_DEPTH_BINS)]
        self.samples = []

    def update(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            self.samples.append(
                (
                    _to_owned_numpy(prediction["boxes3d"]),
                    _to_owned_numpy(prediction["scores"]),
                    _to_owned_numpy(prediction["labels"]),
                    _to_owned_numpy(target["boxes3d"]),
                    _to_owned_numpy(target["labels"]),
                )
            )

    def state_dict(self):
        return self.samples

    def load_state_dict(self, state):
        self.samples = list(state)

    def _accumulate_class(self, class_id, distance_threshold):
        detections = []
        ground_truth = {}
        num_gt = 0
        for sample_id, (boxes, scores, labels, gt_boxes, gt_labels) in enumerate(self.samples):
            indices = np.where(gt_labels == class_id)[0]
            ground_truth[sample_id] = {int(index): False for index in indices}
            num_gt += len(indices)
            for index in np.where(labels == class_id)[0]:
                detections.append((float(scores[index]), sample_id, boxes[index]))
        detections.sort(key=lambda item: item[0], reverse=True)

        true_positives = []
        false_positives = []
        match_confidence = []
        match_errors = {"mATE": [], "mASE": [], "mAOE": [], "mADE": []}
        gt_depths = []
        for sample_id, indices in ground_truth.items():
            gt_boxes = self.samples[sample_id][3]
            gt_depths.extend(float(gt_boxes[index][2]) for index in indices)
        matched_gt_depths = []
        for confidence, sample_id, box in detections:
            gt_boxes = self.samples[sample_id][3]
            candidates = [
                index for index, matched in ground_truth[sample_id].items()
                if not matched
            ]
            best_index = None
            best_distance = float("inf")
            for index in candidates:
                distance = float(np.linalg.norm(box[[0, 2]] - gt_boxes[index][[0, 2]]))
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            # NuScenes uses a strict center-distance comparison.
            if best_index is not None and best_distance < distance_threshold:
                ground_truth[sample_id][best_index] = True
                true_positives.append(1.0)
                false_positives.append(0.0)
                target = gt_boxes[best_index]
                matched_gt_depths.append(float(target[2]))
                match_confidence.append(confidence)
                match_errors["mATE"].append(best_distance)
                match_errors["mASE"].append(
                    1.0 - _scale_iou_aligned(box[3:6], target[3:6])
                )
                match_errors["mAOE"].append(_angle_diff(box[6], target[6]))
                match_errors["mADE"].append(abs(float(box[2] - target[2])))
            else:
                true_positives.append(0.0)
                false_positives.append(1.0)

        tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
        fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
        if len(detections):
            recall = tp / max(num_gt, 1)
            precision = tp / np.maximum(tp + fp, 1.0)
            confidence = np.asarray([item[0] for item in detections], dtype=np.float64)
        else:
            recall = precision = confidence = np.empty(0, dtype=np.float64)

        recall_grid = np.linspace(0.0, 1.0, 101)
        interpolated_precision = np.interp(
            recall_grid, recall, precision, right=0.0
        ) if recall.size else np.zeros(101, dtype=np.float64)
        interpolated_confidence = np.interp(
            recall_grid, recall, confidence, right=0.0
        ) if recall.size else np.zeros(101, dtype=np.float64)
        nonzero_confidence = np.nonzero(interpolated_confidence)[0]
        max_recall_index = int(nonzero_confidence[-1]) if nonzero_confidence.size else 0

        accumulated = {
            "num_gt": int(num_gt),
            "num_predictions": len(detections),
            "num_tp": int(tp[-1]) if tp.size else 0,
            "recall": recall_grid,
            "precision": interpolated_precision,
            "confidence": interpolated_confidence,
            "max_recall_index": max_recall_index,
            "matched_errors": match_errors,
            "gt_depths": gt_depths,
            "matched_gt_depths": matched_gt_depths,
        }
        for metric_name in NDS_TP_TERMS:
            errors = np.asarray(match_errors[metric_name], dtype=np.float64)
            if errors.size:
                cumulative_error = _cummean(errors)
                accumulated[metric_name] = np.interp(
                    interpolated_confidence[::-1],
                    np.asarray(match_confidence, dtype=np.float64)[::-1],
                    cumulative_error[::-1],
                )[::-1]
            else:
                accumulated[metric_name] = np.ones(101, dtype=np.float64)
        return accumulated

    def _confusion_matrix(self):
        """Score-ordered, class-agnostic 2m matching for diagnostic confusion."""
        class_count = len(self.class_names)
        background = class_count
        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(self.class_ids)
        }
        matrix = np.zeros((class_count + 1, class_count + 1), dtype=np.int64)
        for boxes, scores, labels, gt_boxes, gt_labels in self.samples:
            pred_indices = np.where(np.isin(labels, self.class_ids))[0]
            gt_indices = np.where(np.isin(gt_labels, self.class_ids))[0]
            selected_boxes = boxes[pred_indices]
            selected_scores = scores[pred_indices]
            selected_labels = labels[pred_indices]
            selected_gt_boxes = gt_boxes[gt_indices]
            selected_gt_labels = gt_labels[gt_indices]
            matched_gt = np.zeros(len(selected_gt_boxes), dtype=bool)
            if len(selected_boxes) and len(selected_gt_boxes):
                pairwise_distance = np.linalg.norm(
                    selected_boxes[:, None, :][:, :, [0, 2]]
                    - selected_gt_boxes[None, :, :][:, :, [0, 2]],
                    axis=2,
                )
            else:
                pairwise_distance = np.empty((len(selected_boxes), len(selected_gt_boxes)))
            for pred_index in np.argsort(-selected_scores, kind="stable"):
                candidates = np.where(~matched_gt)[0]
                best_index = None
                best_distance = float("inf")
                if candidates.size:
                    candidate_distances = pairwise_distance[pred_index, candidates]
                    best_position = int(np.argmin(candidate_distances))
                    best_index = int(candidates[best_position])
                    best_distance = float(candidate_distances[best_position])
                pred_label = global_to_local[int(selected_labels[pred_index])]
                if best_index is not None and best_distance < self.tp_distance_threshold:
                    matched_gt[best_index] = True
                    gt_label = global_to_local[int(selected_gt_labels[best_index])]
                    matrix[gt_label, pred_label] += 1
                else:
                    matrix[background, pred_label] += 1
            for gt_index in np.where(~matched_gt)[0]:
                gt_label = global_to_local[int(selected_gt_labels[gt_index])]
                matrix[gt_label, background] += 1
        return matrix

    @staticmethod
    def _depth_bin_label(lower, upper):
        if upper >= 1000000:
            return "%gm+" % lower
        return "%g-%gm" % (lower, upper)

    def compute(self):
        per_class_ap = {}
        per_class_ap_by_distance = {}
        ap_by_distance = {str(value): [] for value in self.distance_thresholds}
        counts_by_distance = {
            str(value): {"tp": 0, "gt": 0, "pred": 0}
            for value in self.distance_thresholds
        }
        class_tp_errors = {name: [] for name in NDS_TP_TERMS}
        matched_depth_errors = []
        per_class_pr = {}
        per_class_detection = {}
        depth_counts = {
            self._depth_bin_label(lower, upper): {"tp": 0, "gt": 0}
            for lower, upper in self.depth_bins
        }

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_aps = []
            class_by_distance = {}
            class_has_gt = False
            for threshold in self.distance_thresholds:
                accumulated = self._accumulate_class(class_id, threshold)
                threshold_key = str(threshold)
                ap = _calc_ap(accumulated, self.min_recall, self.min_precision)
                class_by_distance[threshold_key] = None if np.isnan(ap) else ap
                counts = counts_by_distance[threshold_key]
                counts["tp"] += accumulated["num_tp"]
                counts["gt"] += accumulated["num_gt"]
                counts["pred"] += accumulated["num_predictions"]
                if accumulated["num_gt"] > 0:
                    class_has_gt = True
                    class_aps.append(ap)
                    ap_by_distance[threshold_key].append(ap)
                if math.isclose(threshold, self.tp_distance_threshold):
                    per_class_pr[class_name] = {
                        "recall": accumulated["recall"].tolist(),
                        "precision": accumulated["precision"].tolist(),
                        "confidence": accumulated["confidence"].tolist(),
                    }
                    per_class_detection[class_name] = {
                        "tp": accumulated["num_tp"],
                        "gt": accumulated["num_gt"],
                        "pred": accumulated["num_predictions"],
                        "recall": accumulated["num_tp"] / max(accumulated["num_gt"], 1),
                        "precision": accumulated["num_tp"] / max(accumulated["num_predictions"], 1),
                    }
                    if accumulated["num_gt"] > 0:
                        for metric_name in NDS_TP_TERMS:
                            class_tp_errors[metric_name].append(
                                _calc_tp(accumulated, metric_name, self.min_recall)
                            )
                    matched_depth_errors.extend(
                        accumulated["matched_errors"]["mADE"]
                    )
                    for lower, upper in self.depth_bins:
                        label = self._depth_bin_label(lower, upper)
                        depth_counts[label]["gt"] += sum(
                            lower <= depth < upper for depth in accumulated["gt_depths"]
                        )
                        depth_counts[label]["tp"] += sum(
                            lower <= depth < upper
                            for depth in accumulated["matched_gt_depths"]
                        )
            per_class_ap_by_distance[class_name] = class_by_distance
            per_class_ap[class_name] = (
                float(np.mean(class_aps)) if class_has_gt else None
            )

        valid_class_aps = [value for value in per_class_ap.values() if value is not None]
        mean_ap = float(np.mean(valid_class_aps)) if valid_class_aps else 0.0
        tp_errors = {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in class_tp_errors.items()
        }
        tp_scores = [
            max(0.0, 1.0 - tp_errors[name])
            if np.isfinite(tp_errors[name]) else 0.0
            for name in NDS_TP_TERMS
        ]
        nds = (
            self.mean_ap_weight * mean_ap + sum(tp_scores)
        ) / (self.mean_ap_weight + len(NDS_TP_TERMS))

        map_by_distance = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in ap_by_distance.items()
        }
        recall_by_distance = {
            key: values["tp"] / max(values["gt"], 1)
            for key, values in counts_by_distance.items()
        }
        tp_key = str(self.tp_distance_threshold)
        tp_counts = counts_by_distance[tp_key]
        mean_recall = tp_counts["tp"] / max(tp_counts["gt"], 1)
        mean_precision = tp_counts["tp"] / max(tp_counts["pred"], 1)
        f1 = 2.0 * mean_precision * mean_recall / max(
            mean_precision + mean_recall, 1e-12
        )
        made = (
            float(np.mean(matched_depth_errors))
            if matched_depth_errors else float("nan")
        )
        for class_name, value in per_class_ap.items():
            per_class_detection.setdefault(
                class_name,
                {"tp": 0, "gt": 0, "pred": 0, "recall": 0.0, "precision": 0.0},
            )["AP"] = value
        recall_by_depth_bin = {
            label: counts["tp"] / max(counts["gt"], 1)
            for label, counts in depth_counts.items()
        }
        confusion = self._confusion_matrix()
        row_totals = confusion.sum(axis=1, keepdims=True)
        normalized_confusion = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_totals != 0,
        )
        return {
            "protocol": PROTOCOL,
            "nds_variant": NDS_VARIANT,
            "nds_tp_terms": list(NDS_TP_TERMS),
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "mean_ap_weight": self.mean_ap_weight,
            "tp_distance_threshold": self.tp_distance_threshold,
            "evaluated_classes": list(self.class_names),
            "NDS": nds,
            "mAP": mean_ap,
            "mATE": tp_errors["mATE"],
            "mASE": tp_errors["mASE"],
            "mAOE": tp_errors["mAOE"],
            "mADE": made,
            "mRecall": mean_recall,
            "mPrecision": mean_precision,
            "F1": f1,
            "num_samples": len(self.samples),
            "num_gt": tp_counts["gt"],
            "num_predictions": tp_counts["pred"],
            "num_tp": tp_counts["tp"],
            "mAP_by_distance": map_by_distance,
            "recall_by_distance": recall_by_distance,
            "per_class_AP": per_class_ap,
            "per_class_AP_by_distance": per_class_ap_by_distance,
            "per_class_PR": per_class_pr,
            "per_class_detection": per_class_detection,
            "recall_by_depth_bin": recall_by_depth_bin,
            "depth_bin_counts": depth_counts,
            "confusion_matrix": {
                "labels": self.class_names + ["background"],
                "rows": "ground_truth",
                "columns": "prediction",
                "distance_threshold": self.tp_distance_threshold,
                "counts": confusion.tolist(),
                "normalized_by_gt": normalized_confusion.tolist(),
            },
        }
