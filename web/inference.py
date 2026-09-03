"""Local YOLO11m loading, SAHI sliced inference, and box rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import get_slice_bboxes
from torchvision.ops import batched_nms
from ultralytics import YOLO


CLASS_COLORS = (
    "#22D3EE", "#A3E635", "#3B82F6", "#FACC15", "#8B5CF6",
    "#F43F5E", "#10B981", "#F97316", "#D946EF", "#FB7185",
)
DEFAULT_SLICE_OVERLAP = 0.20
INCLUDE_STANDARD_PREDICTION = True


def detection_color(label_id: int) -> str:
    """Return the stable semantic color used for one object class."""
    return CLASS_COLORS[int(label_id) % len(CLASS_COLORS)]


@dataclass(frozen=True)
class ModelBundle:
    """Objects required for local YOLO + SAHI inference."""
    model: Any
    device: str
    id2label: dict[int, str]
    image_size: int


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""
    label_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    """Rendered result and its metadata."""
    image: Image.Image
    detections: tuple[Detection, ...]
    elapsed_ms: float
    merged_count: int
    slice_count: int


def select_device() -> str:
    """Return the SAHI/Ultralytics selector for the best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_checkpoint_path(checkpoint_path: Path) -> None:
    """Fail early when the requested local YOLO checkpoint is unavailable."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint YOLO: {checkpoint_path}")
    if checkpoint_path.suffix.lower() != ".pt":
        raise ValueError(f"Checkpoint YOLO phải là file .pt: {checkpoint_path}")


def _normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(label_id): str(label) for label_id, label in names.items()}
    return {label_id: str(label) for label_id, label in enumerate(names)}


def load_model(checkpoint_path: str | Path) -> ModelBundle:
    """Load the fine-tuned YOLO checkpoint through SAHI's Ultralytics adapter."""
    checkpoint_file = Path(checkpoint_path).resolve()
    validate_checkpoint_path(checkpoint_file)
    ultralytics_model = YOLO(str(checkpoint_file))
    id2label = _normalize_names(ultralytics_model.names)
    if not id2label:
        raise ValueError("Checkpoint YOLO không chứa danh sách lớp.")
    image_size = int(ultralytics_model.overrides.get("imgsz", 640))
    device = select_device()
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=ultralytics_model,
        confidence_threshold=0.04,
        device=device,
        image_size=image_size,
    )
    return ModelBundle(model, device, id2label, image_size)


def predict(
    image: Image.Image,
    bundle: ModelBundle,
    threshold: float = 0.04,
    max_detections: int = 500,
    nms_iou_threshold: float = 0.50,
    slice_overlap: float = DEFAULT_SLICE_OVERLAP,
) -> Prediction:
    """Run SAHI tiles plus its standard prediction and merge them class-wise."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Ngưỡng confidence phải nằm trong khoảng 0 đến 1.")
    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("Ngưỡng IoU của NMS phải nằm trong khoảng 0 đến 1.")
    if max_detections < 1:
        raise ValueError("Số kết quả tối đa phải lớn hơn 0.")
    if not 0.0 <= slice_overlap < 1.0:
        raise ValueError("Tỷ lệ chồng lấp tile phải nằm trong khoảng từ 0 đến dưới 1.")

    rgb_image = image.convert("RGB")
    slice_count = count_slices(rgb_image.size, bundle.image_size, slice_overlap)
    if bundle.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = perf_counter()
    result = get_sliced_prediction(
        image=rgb_image,
        detection_model=bundle.model,
        slice_height=bundle.image_size,
        slice_width=bundle.image_size,
        overlap_height_ratio=float(slice_overlap),
        overlap_width_ratio=float(slice_overlap),
        perform_standard_pred=INCLUDE_STANDARD_PREDICTION,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=float(nms_iou_threshold),
        postprocess_class_agnostic=False,
        verbose=0,
        force_postprocess_type=True,
        confidence_threshold=float(threshold),
    )
    if bundle.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - started) * 1000

    merged_detections = sorted(
        (
            Detection(
                label_id=int(item.category.id),
                label=bundle.id2label.get(int(item.category.id), str(item.category.name)),
                score=float(item.score.value),
                box=tuple(float(value) for value in item.bbox.to_xyxy()),
            )
            for item in result.object_prediction_list
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    detections = tuple(merged_detections[:max_detections])

    return Prediction(
        image=draw_detections(rgb_image, detections),
        detections=detections,
        elapsed_ms=elapsed_ms,
        merged_count=len(merged_detections),
        slice_count=slice_count,
    )


def count_slices(
    image_size: tuple[int, int],
    slice_size: int,
    overlap_ratio: float = DEFAULT_SLICE_OVERLAP,
) -> int:
    """Return how many SAHI tiles will cover an image."""
    if slice_size < 1:
        raise ValueError("Kích thước tile phải lớn hơn 0.")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("Tỷ lệ chồng lấp tile phải nằm trong khoảng từ 0 đến dưới 1.")
    width, height = image_size
    return len(
        get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=slice_size,
            slice_width=slice_size,
            auto_slice_resolution=False,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )
    )


def suppress_overlapping_detections(
    detections: tuple[Detection, ...] | list[Detection],
    iou_threshold: float = 0.45,
) -> tuple[Detection, ...]:
    """Apply class-aware NMS to an existing detection collection."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Ngưỡng IoU của NMS phải nằm trong khoảng 0 đến 1.")
    if not detections:
        return ()
    boxes = torch.tensor([item.box for item in detections], dtype=torch.float32)
    scores = torch.tensor([item.score for item in detections], dtype=torch.float32)
    labels = torch.tensor([item.label_id for item in detections], dtype=torch.int64)
    kept = [
        detections[index]
        for index in batched_nms(boxes, scores, labels, iou_threshold).tolist()
    ]
    kept.sort(key=lambda item: item.score, reverse=True)
    return tuple(kept)


def draw_detections(
    image: Image.Image,
    detections: tuple[Detection, ...] | list[Detection],
) -> Image.Image:
    """Draw class-colored boxes with class labels and confidence percentages."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    short_edge = min(canvas.size)
    line_width = max(2, round(short_edge / 320))
    font_size = max(12, round(short_edge / 42))
    font = _load_font(font_size)

    for detection in detections:
        x_min, y_min, x_max, y_max = _clamp_box(detection.box, canvas.size)
        color = detection_color(detection.label_id)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=line_width)

        label_text = f"{detection.label} {detection.score:.0%}"
        text_box = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding_x = max(5, line_width * 2)
        padding_y = max(3, line_width)
        label_height = text_height + padding_y * 2
        label_top = max(0, y_min - label_height)
        label_right = min(canvas.width, x_min + text_width + padding_x * 2)
        draw.rounded_rectangle(
            (x_min, label_top, label_right, label_top + label_height),
            radius=max(3, line_width * 2),
            fill=color,
        )
        draw.text(
            (x_min + padding_x, label_top + padding_y),
            label_text,
            fill="#07110E",
            font=font,
        )
    return canvas


def summarize_detections(detections: tuple[Detection, ...]) -> dict[str, int]:
    """Count detections by class while keeping the output deterministic."""
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def model_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the model facts exposed by an Ultralytics YOLO checkpoint."""
    checkpoint_file = Path(checkpoint_path).resolve()
    validate_checkpoint_path(checkpoint_file)
    model = YOLO(str(checkpoint_file))
    labels = tuple(_normalize_names(model.names).values())
    image_size = int(model.overrides.get("imgsz", 640))
    return {
        "architecture": "Ultralytics YOLO",
        "backbone": type(model.model).__name__,
        "classes": len(labels),
        "weights_mb": checkpoint_file.stat().st_size / (1024 * 1024),
        "labels": labels,
        "image_size": image_size,
    }


def _clamp_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    x_min, y_min, x_max, y_max = box
    return (
        max(0.0, min(x_min, float(width))),
        max(0.0, min(y_min, float(height))),
        max(0.0, min(x_max, float(width))),
        max(0.0, min(y_max, float(height))),
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
