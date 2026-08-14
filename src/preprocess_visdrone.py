"""Validate VisDrone and export both YOLO labels and COCO detection JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from visdrone_data import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate VisDrone and export YOLO plus COCO detection formats."
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
        choices=["hardlink", "copy", "symlink", "none"],
        default="hardlink",
        help="How images are placed in the processed tree; hardlink falls back to copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all validation/conversion logic without writing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = prepare_dataset(
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
        print(f"\nYOLO and COCO datasets written to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
