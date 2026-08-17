"""Convert cleaned VisDrone JSONL records to an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import yaml

from visdrone_data import (
    CLASS_NAMES,
    iter_processed_records,
    load_processed_manifest,
    materialize_image,
)


@dataclass
class YoloConversionStats:
    split: str
    images_written: int = 0
    label_files_written: int = 0
    objects_written: int = 0


def xywh_to_yolo(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert absolute top-left xywh pixels to normalized YOLO center xywh."""
    x, y, width, height = box
    return (
        (x + width / 2.0) / image_width,
        (y + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    )


def convert_split_to_yolo(
    processed_root: Path | str,
    output_root: Path | str,
    split: str,
    *,
    image_mode: str = "hardlink",
) -> YoloConversionStats:
    """Convert one cleaned split to YOLO images/labels layout."""
    source_root = Path(processed_root)
    output = Path(output_root)
    stats = YoloConversionStats(split=split)

    for record in iter_processed_records(source_root, split):
        image_width = int(record["width"])
        image_height = int(record["height"])
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image size in split {split}: {record['file_name']}")

        lines: list[str] = []
        for obj in record["objects"]:
            category_id = int(obj["category_id"])
            if category_id not in range(len(CLASS_NAMES)):
                raise ValueError(
                    f"Invalid category_id {category_id} in {split}/{record['file_name']}"
                )
            box = tuple(float(value) for value in obj["bbox"])
            if len(box) != 4:
                raise ValueError(f"Invalid bbox in {split}/{record['file_name']}: {box}")
            yolo_box = xywh_to_yolo(box, image_width, image_height)
            if not all(0.0 <= value <= 1.0 for value in yolo_box):
                raise ValueError(
                    f"Out-of-range normalized bbox in {split}/{record['file_name']}: {yolo_box}"
                )
            values = " ".join(f"{value:.6f}" for value in yolo_box)
            lines.append(f"{category_id} {values}\n")

        label_path = output / "labels" / split / Path(record["file_name"]).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(lines), encoding="utf-8")

        source_image = source_root / "images" / split / record["file_name"]
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing processed image: {source_image}")
        destination = output / "images" / split / record["file_name"]
        materialize_image(source_image, destination, image_mode)

        stats.images_written += 1
        stats.label_files_written += 1
        stats.objects_written += len(lines)

    return stats


def write_yolo_yaml(output_root: Path | str, splits: Sequence[str]) -> Path:
    """Write an Ultralytics-compatible dataset YAML."""
    output = Path(output_root).resolve()
    split_set = set(splits)
    payload: dict = {
        "path": output.as_posix(),
        "names": {index - 1: name for index, name in CLASS_NAMES.items()},
    }
    if "train" in split_set:
        payload["train"] = "images/train"
    if "val" in split_set:
        payload["val"] = "images/val"
    if "test-dev" in split_set:
        payload["test"] = "images/test-dev"
    elif "test-challenge" in split_set:
        payload["test"] = "images/test-challenge"

    target = output / "visdrone_yolo.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def convert_dataset_to_yolo(
    processed_root: Path | str,
    output_root: Path | str,
    splits: Sequence[str] | None = None,
    *,
    image_mode: str = "hardlink",
) -> list[YoloConversionStats]:
    """Convert selected splits from the neutral dataset to YOLO."""
    manifest = load_processed_manifest(processed_root)
    selected = list(splits) if splits is not None else list(manifest["splits"])
    unknown = set(selected) - set(manifest["splits"])
    if unknown:
        raise ValueError(f"Splits are not present in processed data: {sorted(unknown)}")
    reports = [
        convert_split_to_yolo(
            processed_root, output_root, split, image_mode=image_mode
        )
        for split in selected
    ]
    write_yolo_yaml(output_root, selected)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert preprocessed VisDrone JSONL data to YOLO format."
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/processed/VisDrone")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/formatted/yolo/VisDrone")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test-dev", "test-challenge"],
        help="Defaults to all splits recorded by preprocessing.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How images are placed in the YOLO tree; hardlink falls back to copy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = convert_dataset_to_yolo(
        args.input_root,
        args.output_root,
        args.splits,
        image_mode=args.image_mode,
    )
    for report in reports:
        print(f"\n[{report.split}]")
        for key, value in asdict(report).items():
            if key != "split":
                print(f"  {key}: {value}")
    print(f"\nYOLO dataset written to: {args.output_root.resolve()}")
    print(f"Dataset YAML: {(args.output_root / 'visdrone_yolo.yaml').resolve()}")


if __name__ == "__main__":
    main()
