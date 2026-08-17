"""Inspect and preprocess the VisDrone detection dataset.

The original VisDrone annotation format is::

    x, y, width, height, score, category, truncation, occlusion

Categories 1..10 are trainable objects. Category 0 (ignored region), category
11 (others), and rows with score 0 are intentionally excluded. Preprocessing
writes a model-neutral JSONL manifest; format-specific conversion lives in the
YOLO and DETR converter modules. The source dataset is never modified.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

SPLIT_DIRECTORY_NAMES = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Annotation:
    x: float
    y: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int


@dataclass
class PreprocessingStats:
    split: str
    images_found: int = 0
    annotations_found: int = 0
    images_written: int = 0
    records_written: int = 0
    rows_total: int = 0
    objects_written: int = 0
    ignored_score: int = 0
    ignored_category: int = 0
    malformed_rows: int = 0
    non_positive_boxes: int = 0
    outside_image_boxes: int = 0
    too_small_boxes: int = 0
    clipped_boxes: int = 0
    missing_annotations: int = 0
    missing_images: int = 0
    unreadable_images: int = 0


def _contains_images(directory: Path) -> bool:
    image_dir = directory / "images"
    return image_dir.is_dir() and any(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        for item in image_dir.iterdir()
    )


def resolve_split_dir(data_root: Path | str, split: str) -> Path:
    """Locate a split even when the downloaded archive adds a duplicate folder."""
    root = Path(data_root).expanduser().resolve()
    if split not in SPLIT_DIRECTORY_NAMES:
        raise ValueError(f"Unknown split {split!r}. Choose from {tuple(SPLIT_DIRECTORY_NAMES)}")

    expected_name = SPLIT_DIRECTORY_NAMES[split]
    candidates = [root / expected_name, root]
    candidates.extend(root.glob(f"**/{expected_name}"))

    visited: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in visited or not candidate.is_dir():
            continue
        visited.add(candidate)
        if _contains_images(candidate):
            return candidate
        nested = candidate / expected_name
        if nested.is_dir() and _contains_images(nested):
            return nested.resolve()

    raise FileNotFoundError(
        f"Cannot find images for split {split!r} below {root}. "
        f"Expected a directory named {expected_name!r}."
    )


def discover_splits(data_root: Path | str) -> dict[str, Path]:
    """Return all discoverable VisDrone split directories."""
    found: dict[str, Path] = {}
    for split in SPLIT_DIRECTORY_NAMES:
        try:
            found[split] = resolve_split_dir(data_root, split)
        except FileNotFoundError:
            continue
    return found


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_annotation_file(path: Path | str) -> tuple[list[Annotation], int]:
    """Parse one annotation file and return (validly parsed rows, malformed count)."""
    rows: list[Annotation] = []
    malformed = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.reader(handle):
            if not raw or all(not value.strip() for value in raw):
                continue
            if len(raw) < 8:
                malformed += 1
                continue
            try:
                values = [float(value.strip()) for value in raw[:8]]
                rows.append(
                    Annotation(
                        x=values[0],
                        y=values[1],
                        width=values[2],
                        height=values[3],
                        score=int(values[4]),
                        category=int(values[5]),
                        truncation=int(values[6]),
                        occlusion=int(values[7]),
                    )
                )
            except (TypeError, ValueError):
                malformed += 1
    return rows, malformed


def clip_xywh(
    annotation: Annotation, image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    """Clip a top-left xywh box to image bounds, or return None if disjoint."""
    if annotation.width <= 0 or annotation.height <= 0:
        return None
    x1 = max(0.0, annotation.x)
    y1 = max(0.0, annotation.y)
    x2 = min(float(image_width), annotation.x + annotation.width)
    y2 = min(float(image_height), annotation.y + annotation.height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def evaluate_annotation(
    annotation: Annotation,
    image_width: int,
    image_height: int,
    min_box_width: float = 1.0,
    min_box_height: float = 1.0,
) -> tuple[str, tuple[float, float, float, float] | None, bool]:
    """Return (status, clipped_box, was_clipped) for preprocessing."""
    if annotation.score <= 0:
        return "ignored_score", None, False
    if annotation.category not in CLASS_NAMES:
        return "ignored_category", None, False
    if annotation.width <= 0 or annotation.height <= 0:
        return "non_positive_boxes", None, False
    clipped = clip_xywh(annotation, image_width, image_height)
    if clipped is None:
        return "outside_image_boxes", None, False
    if clipped[2] < min_box_width or clipped[3] < min_box_height:
        return "too_small_boxes", None, False
    original = (annotation.x, annotation.y, annotation.width, annotation.height)
    was_clipped = any(abs(a - b) > 1e-9 for a, b in zip(original, clipped))
    return "keep", clipped, was_clipped


def _image_map(image_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in list_images(image_dir)}


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    """Copy or link an image without replacing an existing destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode != "none":
        raise ValueError("image_mode must be one of: hardlink, copy, symlink, none")


def iter_processed_records(
    processed_root: Path | str, split: str
):
    """Yield validated records from one model-neutral JSONL split."""
    annotation_path = Path(processed_root) / "annotations" / f"{split}.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing processed annotations for split {split!r}: {annotation_path}"
        )
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {annotation_path} at line {line_number}: {error}"
                ) from error
            required = {"image_id", "file_name", "width", "height", "objects"}
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"Record {line_number} in {annotation_path} is missing {sorted(missing)}"
                )
            yield record


def load_processed_manifest(processed_root: Path | str) -> dict:
    """Load and validate the model-neutral dataset manifest."""
    manifest_path = Path(processed_root) / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run preprocess_visdrone.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "visdrone-clean-jsonl" or manifest.get("version") != 1:
        raise ValueError(
            f"Unsupported processed dataset format in {manifest_path}: "
            f"{manifest.get('format')!r} version {manifest.get('version')!r}"
        )
    if manifest.get("bbox_format") != "xywh_absolute":
        raise ValueError("Processed bounding boxes must use xywh_absolute format")
    return manifest


def preprocess_split(
    source_split_dir: Path | str,
    output_root: Path | str,
    split: str,
    *,
    min_box_width: float = 1.0,
    min_box_height: float = 1.0,
    image_mode: str = "hardlink",
    dry_run: bool = False,
) -> PreprocessingStats:
    """Validate one raw split and write model-neutral image records as JSONL."""
    source = Path(source_split_dir)
    output = Path(output_root)
    image_dir = source / "images"
    annotation_dir = source / "annotations"
    images = _image_map(image_dir)
    annotation_files = (
        {path.stem: path for path in annotation_dir.glob("*.txt")}
        if annotation_dir.is_dir() else {}
    )

    stats = PreprocessingStats(
        split=split,
        images_found=len(images),
        annotations_found=len(annotation_files),
        missing_annotations=len(set(images) - set(annotation_files)),
        missing_images=len(set(annotation_files) - set(images)),
    )

    records: list[dict] = []
    for image_id, (stem, image_path) in enumerate(images.items(), start=1):
        annotation_path = annotation_files.get(stem)
        if annotation_path is None:
            continue
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError):
            stats.unreadable_images += 1
            continue

        annotations, malformed = read_annotation_file(annotation_path)
        stats.malformed_rows += malformed
        stats.rows_total += len(annotations) + malformed
        objects: list[dict] = []
        for annotation in annotations:
            status, clipped, was_clipped = evaluate_annotation(
                annotation,
                image_width,
                image_height,
                min_box_width,
                min_box_height,
            )
            if status != "keep":
                setattr(stats, status, getattr(stats, status) + 1)
                continue
            assert clipped is not None
            if was_clipped:
                stats.clipped_boxes += 1
            x, y, width, height = clipped
            objects.append(
                {
                    "category_id": annotation.category - 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "truncation": annotation.truncation,
                    "occlusion": annotation.occlusion,
                }
            )

        stats.objects_written += len(objects)
        record = {
            "image_id": image_id,
            "file_name": image_path.name,
            "width": image_width,
            "height": image_height,
            "objects": objects,
        }
        records.append(record)
        if not dry_run:
            destination = output / "images" / split / image_path.name
            materialize_image(image_path, destination, image_mode)
            stats.images_written += 1
            stats.records_written += 1

    if not dry_run:
        annotation_path = output / "annotations" / f"{split}.jsonl"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    return stats


def preprocess_dataset(
    data_root: Path | str,
    output_root: Path | str,
    splits: Sequence[str] = ("train", "val", "test-dev"),
    *,
    min_box_width: float = 1.0,
    min_box_height: float = 1.0,
    image_mode: str = "hardlink",
    dry_run: bool = False,
) -> list[PreprocessingStats]:
    """Validate raw splits and create one reusable, model-neutral dataset."""
    reports: list[PreprocessingStats] = []
    for split in splits:
        source = resolve_split_dir(data_root, split)
        reports.append(
            preprocess_split(
                source,
                output_root,
                split,
                min_box_width=min_box_width,
                min_box_height=min_box_height,
                image_mode=image_mode,
                dry_run=dry_run,
            )
        )
    if not dry_run:
        output = Path(output_root)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": "visdrone-clean-jsonl",
            "version": 1,
            "bbox_format": "xywh_absolute",
            "classes": {index - 1: name for index, name in CLASS_NAMES.items()},
            "splits": list(splits),
        }
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_path = output / "preprocessing_report.json"
        report_path.write_text(
            json.dumps([asdict(report) for report in reports], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return reports


def scan_dataset(
    data_root: Path | str,
    splits: Sequence[str] = ("train", "val", "test-dev"),
):
    """Return image-level, box-level, and split-level pandas DataFrames for EDA."""
    import pandas as pd

    image_rows: list[dict] = []
    box_rows: list[dict] = []
    split_rows: list[dict] = []

    for split in splits:
        source = resolve_split_dir(data_root, split)
        images = _image_map(source / "images")
        annotation_dir = source / "annotations"
        annotations = (
            {path.stem: path for path in annotation_dir.glob("*.txt")}
            if annotation_dir.is_dir() else {}
        )
        malformed_total = 0
        unreadable = 0

        for stem, image_path in images.items():
            annotation_path = annotations.get(stem)
            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                    image_mode = image.mode
            except (OSError, UnidentifiedImageError):
                image_width = image_height = 0
                image_mode = "unreadable"
                unreadable += 1

            parsed: list[Annotation] = []
            malformed = 0
            if annotation_path is not None:
                parsed, malformed = read_annotation_file(annotation_path)
            malformed_total += malformed
            kept_count = 0
            for annotation in parsed:
                status, clipped, was_clipped = evaluate_annotation(
                    annotation, image_width, image_height
                ) if image_width and image_height else ("unreadable_image", None, False)
                if status == "keep":
                    kept_count += 1
                clipped_width = clipped[2] if clipped else 0.0
                clipped_height = clipped[3] if clipped else 0.0
                box_rows.append(
                    {
                        "split": split,
                        "image_id": stem,
                        "x": annotation.x,
                        "y": annotation.y,
                        "box_width": annotation.width,
                        "box_height": annotation.height,
                        "score": annotation.score,
                        "category": annotation.category,
                        "class_name": CLASS_NAMES.get(annotation.category, "ignored/others"),
                        "truncation": annotation.truncation,
                        "occlusion": annotation.occlusion,
                        "image_width": image_width,
                        "image_height": image_height,
                        "status": status,
                        "was_clipped": was_clipped,
                        "clipped_width": clipped_width,
                        "clipped_height": clipped_height,
                        "area_px": clipped_width * clipped_height,
                        "relative_area": (
                            clipped_width * clipped_height / (image_width * image_height)
                            if image_width and image_height else 0.0
                        ),
                        "aspect_ratio": (
                            clipped_width / clipped_height if clipped_height else float("nan")
                        ),
                    }
                )

            image_rows.append(
                {
                    "split": split,
                    "image_id": stem,
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path) if annotation_path else None,
                    "width": image_width,
                    "height": image_height,
                    "mode": image_mode,
                    "file_size_kb": image_path.stat().st_size / 1024.0,
                    "raw_objects": len(parsed),
                    "trainable_objects": kept_count,
                    "malformed_rows": malformed,
                }
            )

        split_rows.append(
            {
                "split": split,
                "split_path": str(source),
                "images": len(images),
                "annotations": len(annotations),
                "missing_annotations": len(set(images) - set(annotations)),
                "missing_images": len(set(annotations) - set(images)),
                "unreadable_images": unreadable,
                "malformed_rows": malformed_total,
            }
        )

    return pd.DataFrame(image_rows), pd.DataFrame(box_rows), pd.DataFrame(split_rows)


def summarize_status(boxes) -> dict[str, int]:
    """Small serializable status summary, useful in scripts and notebooks."""
    return {str(key): int(value) for key, value in boxes["status"].value_counts().items()}
