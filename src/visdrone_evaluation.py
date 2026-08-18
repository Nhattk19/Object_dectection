"""Python-compatible port of the official VisDrone DET MATLAB evaluator.

The matching, ignore-region filtering, IoU thresholds, VOC-style AP integral,
and max-detection limits mirror VisDrone2018-DET-toolkit version 1.0.4.
Metrics are returned in the 0..1 range to match pycocotools conventions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
MAX_DETECTIONS = (1, 10, 100, 500)


def export_visdrone_predictions(
    model,
    data_loader,
    dataset,
    device,
    output_root: Path | str,
) -> Path:
    """Run inference and write one official-format TXT file per image."""

    import torch
    from tqdm.auto import tqdm

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    image_names = {
        int(item["id"]): Path(item["file_name"]).stem for item in dataset.images
    }
    model.eval()
    with torch.inference_mode():
        for images, targets in tqdm(data_loader, desc="VisDrone validation", leave=False):
            predictions = model(
                [image.to(device, non_blocking=True) for image in images]
            )
            for prediction, target in zip(predictions, targets):
                image_id = int(target["image_id"].item())
                lines: list[str] = []
                for box, score, label in zip(
                    prediction["boxes"].cpu(),
                    prediction["scores"].cpu(),
                    prediction["labels"].cpu(),
                ):
                    class_id = int(label)
                    if not 1 <= class_id <= 10:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    lines.append(
                        f"{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},"
                        f"{float(score):.8f},{class_id},-1,-1\n"
                    )
                (output / f"{image_names[image_id]}.txt").write_text(
                    "".join(lines), encoding="utf-8"
                )
    return output


def _read_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            values = [item.strip() for item in row if item.strip()]
            if values:
                rows.append([float(item) for item in values[:8]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def _matlab_round_positive(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int64)


def _drop_objects_in_ignore_regions(
    ground_truth: np.ndarray,
    detections: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``dropObjectsInIgr.m`` including its integral-image convention."""

    ignored_regions = np.maximum(1, ground_truth[ground_truth[:, 5] == 0, :4])
    current_gt = ground_truth[ground_truth[:, 5] != 0].copy()
    if not len(ignored_regions):
        return current_gt, detections

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for x, y, width, height in ignored_regions:
        x0 = max(1, int(x)) - 1
        y0 = max(1, int(y)) - 1
        x1 = min(image_width, int(x + width))
        y1 = min(image_height, int(y + height))
        mask[y0:y1, x0:x1] = 1
    integral = mask.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)

    def retained(rows: np.ndarray) -> np.ndarray:
        keep: list[int] = []
        for index, row in enumerate(rows):
            x, y, width, height = np.maximum(
                1, _matlab_round_positive(row[:4])
            )
            x = max(1, min(image_width, int(x)))
            y = max(1, min(image_height, int(y)))
            x2 = min(image_width, x + int(width))
            y2 = min(image_height, y + int(height))
            tl = integral[y - 1, x - 1]
            tr = integral[y - 1, x2 - 1]
            bl = integral[y2 - 1, x - 1]
            br = integral[y2 - 1, x2 - 1]
            ignored_area = float(tl + br - tr - bl)
            if ignored_area / max(1.0, float(width * height)) < 0.5:
                keep.append(index)
        return rows[keep]

    return retained(current_gt), retained(detections)


def _overlap_matrix(detections: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if not len(detections) or not len(ground_truth):
        return np.zeros((len(detections), len(ground_truth)), dtype=np.float64)
    det_xy1 = detections[:, None, :2]
    gt_xy1 = ground_truth[None, :, :2]
    det_xy2 = det_xy1 + detections[:, None, 2:4]
    gt_xy2 = gt_xy1 + ground_truth[None, :, 2:4]
    intersection_wh = np.maximum(
        0.0, np.minimum(det_xy2, gt_xy2) - np.maximum(det_xy1, gt_xy1)
    )
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    det_area = detections[:, None, 2] * detections[:, None, 3]
    gt_area = ground_truth[None, :, 2] * ground_truth[None, :, 3]
    union = det_area + gt_area - intersection
    union = np.where(ground_truth[None, :, 4].astype(bool), det_area, union)
    return intersection / np.maximum(union, 1e-12)


def _match_image(
    ground_truth: np.ndarray, detections: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the official toolkit's ``evalRes.m`` matching."""

    gt = ground_truth[np.argsort(ground_truth[:, 4], kind="stable")].copy()
    dt = detections[np.argsort(-detections[:, 4], kind="stable")].copy()
    gt_match = -gt[:, 4].copy()
    dt_match = np.zeros(len(dt), dtype=np.float64)
    overlaps = _overlap_matrix(dt, gt)
    for det_index in range(len(dt)):
        best_overlap = threshold
        best_gt = -1
        best_match = 0.0
        for gt_index in range(len(gt)):
            match = gt_match[gt_index]
            if match == 1:
                continue
            if best_match != 0 and match == -1:
                break
            if overlaps[det_index, gt_index] < best_overlap:
                continue
            best_overlap = overlaps[det_index, gt_index]
            best_gt = gt_index
            best_match = 1.0 if match == 0 else -1.0
        if best_match == -1:
            dt_match[det_index] = -1
        elif best_match == 1:
            gt_match[best_gt] = 1
            dt_match[det_index] = 1
    return gt_match, np.column_stack((dt[:, 4], dt_match))


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[changes] - mrec[changes - 1]) * mpre[changes]))


def evaluate_visdrone(
    ground_truth_root: Path | str,
    detection_root: Path | str,
    image_root: Path | str,
) -> dict[str, float]:
    """Evaluate VisDrone TXT predictions with the official toolkit semantics."""

    gt_root = Path(ground_truth_root)
    det_root = Path(detection_root)
    images = Path(image_root)
    annotation_paths = sorted(gt_root.glob("*.txt"))
    if not annotation_paths:
        raise FileNotFoundError(f"No raw VisDrone annotations found in {gt_root}")

    all_gt: list[np.ndarray] = []
    all_det: list[np.ndarray] = []
    for annotation_path in annotation_paths:
        image_path = images / f"{annotation_path.stem}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        gt = _read_rows(annotation_path)
        det = _read_rows(det_root / annotation_path.name)
        gt, det = _drop_objects_in_ignore_regions(gt, det, height, width)
        # VisDrone columns: score 1 -> evaluated (ignore=0), score 0 -> ignore=1.
        gt[:, 4] = np.where(gt[:, 4] == 0, 1, 0)
        all_gt.append(gt)
        all_det.append(det)

    ap = np.zeros((10, len(IOU_THRESHOLDS)), dtype=np.float64)
    ar = np.zeros((10, len(IOU_THRESHOLDS), len(MAX_DETECTIONS)), dtype=np.float64)
    # The official MATLAB code repeats a class once per image containing it.
    class_weights = np.asarray(
        [sum(np.any(gt[:, 5] == class_id) for gt in all_gt) for class_id in range(1, 11)],
        dtype=np.float64,
    )

    for class_index, class_id in enumerate(range(1, 11)):
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            for max_index, max_detections in enumerate(MAX_DETECTIONS):
                gt_matches: list[np.ndarray] = []
                det_matches: list[np.ndarray] = []
                for gt, det in zip(all_gt, all_det):
                    class_gt = gt[gt[:, 5] == class_id, :5]
                    limited_det = det[: min(len(det), max_detections)]
                    class_det = limited_det[limited_det[:, 5] == class_id, :5]
                    image_gt, image_det = _match_image(class_gt, class_det, threshold)
                    gt_matches.append(image_gt)
                    det_matches.append(image_det)
                merged_gt = np.concatenate(gt_matches) if gt_matches else np.empty(0)
                merged_det = (
                    np.concatenate(det_matches) if det_matches else np.empty((0, 2))
                )
                order = np.argsort(-merged_det[:, 0], kind="stable") if len(merged_det) else []
                matches = merged_det[order, 1] if len(merged_det) else np.empty(0)
                true_positives = np.cumsum(matches == 1)
                recall = true_positives / max(1, len(merged_gt))
                if len(recall):
                    ar[class_index, threshold_index, max_index] = recall[-1]
                if max_detections == 500:
                    false_positives = np.cumsum(matches == 0)
                    precision = true_positives / np.maximum(
                        1, true_positives + false_positives
                    )
                    ap[class_index, threshold_index] = _voc_ap(recall, precision)

    valid = class_weights > 0
    weights = class_weights[valid]
    if not len(weights):
        raise ValueError("No VisDrone categories 1..10 were found")

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.average(values[valid], axis=0, weights=weights).mean())

    return {
        "visdrone_ap": weighted_mean(ap),
        "visdrone_ap50": weighted_mean(ap[:, :1]),
        "visdrone_ap75": weighted_mean(ap[:, 5:6]),
        "visdrone_ar1": weighted_mean(ar[:, :, 0]),
        "visdrone_ar10": weighted_mean(ar[:, :, 1]),
        "visdrone_ar100": weighted_mean(ar[:, :, 2]),
        "visdrone_ar500": weighted_mean(ar[:, :, 3]),
        "images": len(annotation_paths),
    }
