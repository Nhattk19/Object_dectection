"""Architecture-independent COCO-style evaluation for VisDrone detections.

The project uses four maximum-detection settings: 1, 10, 100, and 500. All AP
and area-specific metrics are reported at maxDets=500. This is intentionally a
COCO-style extension for crowded VisDrone images, rather than the official COCO
maxDets=100 summary.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COCO_MAX_DETS = (1, 10, 100, 500)
COCO_METRIC_NAMES = (
    "AP (50:95)",
    "AP50",
    "AP75",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR@500",
    "AP small",
    "AR small",
    "AP medium",
    "AR medium",
    "AP large",
    "AR large",
)


def _mean_valid(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def _index_of(values: Sequence[Any], target: Any, *, float_value: bool = False) -> int:
    for index, value in enumerate(values):
        if (float_value and np.isclose(float(value), float(target))) or value == target:
            return index
    raise ValueError(f"{target!r} is not present in {list(values)!r}")


def _average_precision(evaluator, *, area: str, max_dets: int, iou: float | None = None) -> float:
    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDets]
    area_index = _index_of(evaluator.params.areaRngLbl, area)
    max_det_index = _index_of(evaluator.params.maxDets, max_dets)
    if iou is None:
        selected = precision[:, :, :, area_index, max_det_index]
    else:
        iou_index = _index_of(evaluator.params.iouThrs, iou, float_value=True)
        selected = precision[iou_index, :, :, area_index, max_det_index]
    return _mean_valid(selected)


def _average_recall(evaluator, *, area: str, max_dets: int) -> float:
    recall = evaluator.eval["recall"]  # [IoU, class, area, maxDets]
    area_index = _index_of(evaluator.params.areaRngLbl, area)
    max_det_index = _index_of(evaluator.params.maxDets, max_dets)
    return _mean_valid(recall[:, :, area_index, max_det_index])


def extract_extended_coco_metrics(evaluator, *, ap_max_dets: int = 500) -> dict[str, float]:
    """Extract the 13 project metrics from an accumulated COCOeval instance."""

    metrics = {
        "AP (50:95)": _average_precision(evaluator, area="all", max_dets=ap_max_dets),
        "AP50": _average_precision(evaluator, area="all", max_dets=ap_max_dets, iou=0.50),
        "AP75": _average_precision(evaluator, area="all", max_dets=ap_max_dets, iou=0.75),
        "AR@1": _average_recall(evaluator, area="all", max_dets=1),
        "AR@10": _average_recall(evaluator, area="all", max_dets=10),
        "AR@100": _average_recall(evaluator, area="all", max_dets=100),
        "AR@500": _average_recall(evaluator, area="all", max_dets=500),
        "AP small": _average_precision(evaluator, area="small", max_dets=ap_max_dets),
        "AR small": _average_recall(evaluator, area="small", max_dets=ap_max_dets),
        "AP medium": _average_precision(evaluator, area="medium", max_dets=ap_max_dets),
        "AR medium": _average_recall(evaluator, area="medium", max_dets=ap_max_dets),
        "AP large": _average_precision(evaluator, area="large", max_dets=ap_max_dets),
        "AR large": _average_recall(evaluator, area="large", max_dets=ap_max_dets),
    }
    return {name: float(metrics[name]) for name in COCO_METRIC_NAMES}


def validate_coco_predictions(
    predictions: Sequence[Mapping[str, Any]], annotation_file: Path | str
) -> list[dict[str, Any]]:
    """Validate and normalize prediction dictionaries before COCO evaluation."""

    annotation_payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    valid_image_ids = {int(image["id"]) for image in annotation_payload["images"]}
    valid_category_ids = {int(category["id"]) for category in annotation_payload["categories"]}
    normalized: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        missing = {"image_id", "category_id", "bbox", "score"} - set(prediction)
        if missing:
            raise ValueError(f"Prediction {index} is missing fields: {sorted(missing)}")
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        bbox = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])
        if image_id not in valid_image_ids:
            raise ValueError(f"Prediction {index} has unknown image_id={image_id}")
        if category_id not in valid_category_ids:
            raise ValueError(f"Prediction {index} has unknown category_id={category_id}")
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Prediction {index} has invalid COCO xywh bbox={bbox}")
        if not np.isfinite([*bbox, score]).all():
            raise ValueError(f"Prediction {index} contains a non-finite value")
        normalized.append(
            {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}
        )
    return normalized


def evaluate_coco_predictions(
    annotation_file: Path | str,
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_ids: Sequence[int] | None = None,
    max_dets: Sequence[int] = COCO_MAX_DETS,
    ap_max_dets: int = 500,
) -> dict[str, float]:
    """Evaluate COCO detections and return metrics in the 0..1 range.

    AP, AP50, AP75, and all scale metrics use ``ap_max_dets``. AR@K uses its
    named K. The default therefore evaluates all AP/scale metrics at 500 final
    detections per image, as requested for crowded VisDrone scenes.
    """

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError("Install pycocotools>=2.0.7 for COCO evaluation") from error

    annotation_file = Path(annotation_file).expanduser().resolve()
    normalized = validate_coco_predictions(predictions, annotation_file)
    if not normalized:
        return {name: 0.0 for name in COCO_METRIC_NAMES}

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_file))
        detections = ground_truth.loadRes(normalized)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = sorted({int(value) for value in max_dets})
        if ap_max_dets not in evaluator.params.maxDets:
            raise ValueError(
                f"ap_max_dets={ap_max_dets} must be included in max_dets={evaluator.params.maxDets}"
            )
        if image_ids is not None:
            evaluator.params.imgIds = sorted({int(value) for value in image_ids})
        evaluator.evaluate()
        evaluator.accumulate()
    return extract_extended_coco_metrics(evaluator, ap_max_dets=ap_max_dets)


def metrics_as_percent(metrics: Mapping[str, float]) -> dict[str, float]:
    """Convert metric fractions to percentages without changing their order."""

    return {
        name: (100.0 * float(metrics[name]) if np.isfinite(metrics[name]) else float("nan"))
        for name in COCO_METRIC_NAMES
    }


def save_coco_predictions(path: Path | str, predictions: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(predictions), indent=2), encoding="utf-8")
    return target


def load_coco_predictions(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, list):
        raise ValueError("COCO prediction JSON must be a list or contain a 'predictions' list")
    return payload


def _checkpoint_payload(path: Path | str):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.6
        return torch.load(path, map_location="cpu")


def run_faster_rcnn_predictions(
    checkpoint: Path | str,
    annotation_file: Path | str,
    image_root: Path | str,
    *,
    device: str | None = None,
    workers: int = 2,
    score_threshold: float = 0.001,
    max_detections: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a project Faster R-CNN checkpoint and return COCO predictions."""

    import torch

    from faster_rcnn import (
        VisDroneCocoDataset,
        build_faster_rcnn,
        detection_collate_fn,
        predictions_to_coco,
    )

    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = _checkpoint_payload(checkpoint)
    state_dict = payload.get("model", payload)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    experiment = str(config.get("experiment", checkpoint.parent.name)).upper()
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_faster_rcnn(
        num_classes=11,
        pretrained=False,
        small_anchors=bool(config.get("small_anchors", experiment == "F2")),
        min_size=int(config.get("min_size", 896 if experiment == "F5" else 800)),
        max_size=int(config.get("max_size", 1493 if experiment == "F5" else 1333)),
        crowded_proposals=bool(config.get("crowded_proposals", experiment in {"F3", "F4", "F5"})),
        box_detections_per_img=max_detections,
        box_score_thresh=score_threshold,
        model_version=str(config.get("model_version", "v2" if experiment in {"F4", "F5"} else "v1")),
    )
    model.load_state_dict(state_dict)
    model.to(selected_device).eval()

    dataset = VisDroneCocoDataset(annotation_file, image_root)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=selected_device.type == "cuda",
        persistent_workers=workers > 0,
    )
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
        torch.cuda.synchronize(selected_device)
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = [image.to(selected_device, non_blocking=True) for image in images]
            outputs = model(images)
            predictions.extend(
                predictions_to_coco(outputs, targets, score_threshold=score_threshold)
            )
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    seconds = time.perf_counter() - started
    metadata = {
        "adapter": "faster_rcnn",
        "checkpoint": str(checkpoint),
        "experiment": experiment,
        "images": len(dataset),
        "predictions": len(predictions),
        "seconds": seconds,
        "images_per_second": len(dataset) / seconds,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(selected_device) / 1024**3
            if selected_device.type == "cuda"
            else 0.0
        ),
    }
    return predictions, metadata


def _normalized_class_name(value: str) -> str:
    return "-".join(str(value).strip().lower().replace("_", "-").split())


def _ultralytics_category_mapping(model_names: Mapping[int, str], annotation_file: Path | str):
    payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    gt_by_name = {
        _normalized_class_name(category["name"]): int(category["id"])
        for category in payload["categories"]
    }
    aliases = {"motorcycle": "motor", "awning-tricycle": "awning-tricycle"}
    mapping: dict[int, int] = {}
    for model_id, model_name in model_names.items():
        normalized = _normalized_class_name(model_name)
        normalized = aliases.get(normalized, normalized)
        if normalized not in gt_by_name:
            raise ValueError(
                f"Cannot map model class {model_id}:{model_name!r} to COCO categories "
                f"{sorted(gt_by_name)}"
            )
        mapping[int(model_id)] = gt_by_name[normalized]
    return mapping


def run_ultralytics_predictions(
    checkpoint: Path | str,
    model_kind: str,
    annotation_file: Path | str,
    image_root: Path | str,
    *,
    imgsz: int = 960,
    device: int | str | None = None,
    batch_size: int = 4,
    score_threshold: float = 0.001,
    nms_iou: float = 0.7,
    max_detections: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Ultralytics YOLO or RT-DETR weights and return COCO predictions."""

    try:
        import torch
        from ultralytics import RTDETR, YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics to evaluate YOLO or RT-DETR weights") from error

    model_kind = model_kind.lower().replace("-", "_")
    if model_kind not in {"yolo", "ultralytics_yolo", "rtdetr", "rt_detr", "ultralytics_rtdetr"}:
        raise ValueError(f"Unsupported Ultralytics model_kind={model_kind!r}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    model = RTDETR(str(checkpoint)) if "detr" in model_kind else YOLO(str(checkpoint))
    annotation_payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    images = sorted(annotation_payload["images"], key=lambda item: int(item["id"]))
    image_root = Path(image_root)
    selected_device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and selected_device != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    class_mapping: dict[int, int] | None = None
    for start in range(0, len(images), batch_size):
        batch_infos = images[start : start + batch_size]
        batch_paths = [str(image_root / info["file_name"]) for info in batch_infos]
        for path in batch_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        results = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            conf=score_threshold,
            iou=nms_iou,
            max_det=max_detections,
            device=selected_device,
            verbose=False,
            stream=False,
        )
        if len(results) != len(batch_infos):
            raise RuntimeError("Ultralytics returned a different number of results than input images")
        for image_info, result in zip(batch_infos, results):
            if class_mapping is None:
                names = result.names
                if isinstance(names, list):
                    names = dict(enumerate(names))
                class_mapping = _ultralytics_category_mapping(names, annotation_file)
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, model_class in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": int(image_info["id"]),
                        "category_id": int(class_mapping[model_class]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    if torch.cuda.is_available() and selected_device != "cpu":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    metadata = {
        "adapter": "rtdetr" if "detr" in model_kind else "yolo",
        "checkpoint": str(checkpoint),
        "imgsz": imgsz,
        "images": len(images),
        "predictions": len(predictions),
        "seconds": seconds,
        "images_per_second": len(images) / seconds,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() and selected_device != "cpu"
            else 0.0
        ),
    }
    return predictions, metadata
