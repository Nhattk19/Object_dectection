"""Convert cleaned VisDrone JSONL records to COCO layout for DETR fine-tuning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from visdrone_data import (
    CLASS_NAMES,
    iter_processed_records,
    load_processed_manifest,
    materialize_image,
)


DETR_SPLIT_NAMES = {
    "train": "train",
    "val": "valid",
    "test-dev": "test",
    "test-challenge": "test-challenge",
}


@dataclass
class DetrConversionStats:
    source_split: str
    output_split: str
    images_written: int = 0
    annotations_written: int = 0


def convert_split_to_detr(
    processed_root: Path | str,
    output_root: Path | str,
    split: str,
    *,
    image_mode: str = "hardlink",
) -> DetrConversionStats:
    """Write one split in the COCO layout expected by the DETR notebook."""
    if split not in DETR_SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {split!r}")

    source_root = Path(processed_root)
    output_split = DETR_SPLIT_NAMES[split]
    destination_dir = Path(output_root) / output_split
    destination_dir.mkdir(parents=True, exist_ok=True)
    stats = DetrConversionStats(source_split=split, output_split=output_split)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    annotation_id = 1

    for record in iter_processed_records(source_root, split):
        image_id = int(record["image_id"])
        file_name = Path(record["file_name"]).name
        width = int(record["width"])
        height = int(record["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in split {split}: {file_name}")

        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for obj in record["objects"]:
            category_id = int(obj["category_id"])
            if category_id not in range(len(CLASS_NAMES)):
                raise ValueError(
                    f"Invalid category_id {category_id} in {split}/{file_name}"
                )
            bbox = [float(value) for value in obj["bbox"]]
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"Invalid bbox in {split}/{file_name}: {bbox}")
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": float(obj.get("area", bbox[2] * bbox[3])),
                    "iscrowd": 0,
                    "segmentation": [],
                    "truncation": int(obj.get("truncation", 0)),
                    "occlusion": int(obj.get("occlusion", 0)),
                }
            )
            annotation_id += 1

        source_image = source_root / "images" / split / file_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing processed image: {source_image}")
        materialize_image(source_image, destination_dir / file_name, image_mode)
        stats.images_written += 1

    categories = [
        {"id": index - 1, "name": name, "supercategory": "object"}
        for index, name in CLASS_NAMES.items()
    ]
    payload = {
        "info": {
            "description": "Cleaned VisDrone2019 detection dataset for DETR",
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    annotation_path = destination_dir / "_annotations.coco.json"
    annotation_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    stats.annotations_written = len(coco_annotations)
    return stats


def convert_dataset_to_detr(
    processed_root: Path | str,
    output_root: Path | str,
    splits: Sequence[str] | None = None,
    *,
    image_mode: str = "hardlink",
) -> list[DetrConversionStats]:
    """Convert selected neutral splits to per-split COCO datasets for DETR."""
    manifest = load_processed_manifest(processed_root)
    selected = list(splits) if splits is not None else list(manifest["splits"])
    unknown = set(selected) - set(manifest["splits"])
    if unknown:
        raise ValueError(f"Splits are not present in processed data: {sorted(unknown)}")

    reports = [
        convert_split_to_detr(
            processed_root, output_root, split, image_mode=image_mode
        )
        for split in selected
    ]

    output = Path(output_root)
    metadata = {
        "format": "coco-detection",
        "classes": {index - 1: name for index, name in CLASS_NAMES.items()},
        "split_mapping": {
            report.source_split: report.output_split for report in reports
        },
        "annotation_file": "_annotations.coco.json",
    }
    (output / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert preprocessed VisDrone JSONL data to per-split COCO layout "
            "for Hugging Face DETR."
        )
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/processed/VisDrone")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/formatted/detr/VisDrone")
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
        help="How images are placed in COCO split folders; hardlink falls back to copy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = convert_dataset_to_detr(
        args.input_root,
        args.output_root,
        args.splits,
        image_mode=args.image_mode,
    )
    for report in reports:
        print(f"\n[{report.source_split} -> {report.output_split}]")
        for key, value in asdict(report).items():
            if key not in {"source_split", "output_split"}:
                print(f"  {key}: {value}")
    print(f"\nDETR/COCO dataset written to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
