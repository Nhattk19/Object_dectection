"""Create a cleaned, model-neutral VisDrone dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from visdrone_data import preprocess_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw VisDrone data and write cleaned absolute-xywh JSONL records. "
            "Run a YOLO or DETR converter afterwards."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/VisDrone"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test-dev"],
        choices=["train", "val", "test-dev", "test-challenge"],
    )
    parser.add_argument("--min-box-width", type=float, default=1.0)
    parser.add_argument("--min-box-height", type=float, default=1.0)
    parser.add_argument(
        "--image-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How images are placed in the processed tree; hardlink falls back to copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all validation and filtering logic without writing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = preprocess_dataset(
        args.data_root,
        args.output_root,
        args.splits,
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
        image_mode=args.image_mode,
        dry_run=args.dry_run,
    )
    for report in reports:
        print(f"\n[{report.split}]")
        for key, value in asdict(report).items():
            if key != "split":
                print(f"  {key}: {value}")
    if args.dry_run:
        print("\nDry-run complete: no files were written.")
    else:
        print(f"\nModel-neutral dataset written to: {args.output_root.resolve()}")
        print("Next: run convert_visdrone_to_yolo.py or convert_visdrone_to_detr.py")


if __name__ == "__main__":
    main()
