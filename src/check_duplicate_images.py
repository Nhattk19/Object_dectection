"""Find exact duplicate images across the VisDrone train/val/test splits.

Images are hashed from their decoded RGB pixels instead of their file bytes. As
a result, identical images are still detected when they have different names or
lossless file encodings. The script prints every duplicate group, writes a CSV
report, and saves a preview of a few groups.

Run from the project root::

    python src/check_duplicate_images.py

Optional example::

    python src/check_duplicate_images.py --max-preview-groups 10 --show
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import struct
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from visdrone_data import IMAGE_SUFFIXES, resolve_split_dir


@dataclass(frozen=True)
class ImageRecord:
    split: str
    path: Path
    digest: str
    width: int
    height: int


def hash_image(split_and_path: tuple[str, Path]) -> tuple[ImageRecord | None, str | None]:
    """Return a content hash for one image and a readable error if it fails."""
    split, path = split_and_path
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            hasher = hashlib.sha256()
            hasher.update(struct.pack(">II", width, height))
            hasher.update(image.tobytes())
        return ImageRecord(split, path.resolve(), hasher.hexdigest(), width, height), None
    except (OSError, UnidentifiedImageError) as error:
        return None, f"{path}: {error}"


def find_duplicate_groups(
    data_root: Path | str,
    splits: Sequence[str] = ("train", "val", "test-dev"),
    workers: int | None = None,
) -> tuple[list[list[ImageRecord]], list[str], dict[str, int]]:
    """Hash all selected splits together and return groups with at least 2 images."""
    candidates: list[tuple[str, Path]] = []
    split_counts: dict[str, int] = {}

    for split in splits:
        image_dir = resolve_split_dir(data_root, split) / "images"
        paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        split_counts[split] = len(paths)
        candidates.extend((split, path) for path in paths)
        print(f"[{split}] Tìm thấy {len(paths):,} ảnh tại {image_dir}")

    hash_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    errors: list[str] = []
    worker_count = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for record, error in executor.map(hash_image, candidates):
            if record is not None:
                hash_groups[record.digest].append(record)
            if error is not None:
                errors.append(error)

    duplicate_groups = [
        sorted(records, key=lambda item: (item.split, str(item.path)))
        for records in hash_groups.values()
        if len(records) > 1
    ]
    duplicate_groups.sort(key=lambda group: (-len(group), group[0].digest))
    return duplicate_groups, errors, split_counts


def is_cross_split(group: Sequence[ImageRecord]) -> bool:
    return len({record.split for record in group}) > 1


def print_report(
    groups: Sequence[Sequence[ImageRecord]],
    errors: Sequence[str],
    split_counts: dict[str, int],
) -> None:
    """Print all duplicate paths and a short summary."""
    total_images = sum(split_counts.values())
    duplicate_images = sum(len(group) for group in groups)
    cross_split_groups = sum(is_cross_split(group) for group in groups)

    print("\n" + "=" * 72)
    print(f"Đã kiểm tra: {total_images:,} ảnh")
    print(f"Số nhóm ảnh trùng: {len(groups):,}")
    print(f"Số ảnh nằm trong các nhóm trùng: {duplicate_images:,}")
    print(f"Số nhóm trùng giữa nhiều tập: {cross_split_groups:,}")

    if not groups:
        print("Kết luận: không phát hiện ảnh trùng trong các tập đã chọn.")
    else:
        print("\nDanh sách đầy đủ:")
        for group_id, group in enumerate(groups, start=1):
            group_type = "TRÙNG KHÁC TẬP" if is_cross_split(group) else "trùng trong cùng tập"
            print(f"\nNhóm {group_id} ({group_type}, {len(group)} ảnh):")
            for record in group:
                print(f"  [{record.split}] {record.path}")

    if errors:
        print(f"\nKhông đọc được {len(errors)} ảnh:")
        for error in errors:
            print(f"  {error}")


def write_csv_report(groups: Sequence[Sequence[ImageRecord]], output_path: Path) -> None:
    """Write one CSV row per image belonging to a duplicate group."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group_id",
                "cross_split",
                "split",
                "width",
                "height",
                "sha256_pixels",
                "image_path",
            ),
        )
        writer.writeheader()
        for group_id, group in enumerate(groups, start=1):
            for record in group:
                writer.writerow(
                    {
                        "group_id": group_id,
                        "cross_split": is_cross_split(group),
                        "split": record.split,
                        "width": record.width,
                        "height": record.height,
                        "sha256_pixels": record.digest,
                        "image_path": str(record.path),
                    }
                )


def save_preview(
    groups: Sequence[Sequence[ImageRecord]],
    output_path: Path,
    max_groups: int = 5,
    max_images_per_group: int = 4,
    show: bool = False,
) -> bool:
    """Save a contact sheet for the first few duplicate groups."""
    selected_groups = list(groups[:max_groups])
    if not selected_groups or max_groups == 0:
        return False

    import matplotlib.pyplot as plt

    columns = min(max_images_per_group, max(len(group) for group in selected_groups))
    figure, axes = plt.subplots(
        len(selected_groups),
        columns,
        figsize=(4.2 * columns, 3.4 * len(selected_groups)),
        squeeze=False,
    )

    for row, group in enumerate(selected_groups):
        for column in range(columns):
            axis = axes[row][column]
            axis.axis("off")
            if column >= min(len(group), max_images_per_group):
                continue
            record = group[column]
            with Image.open(record.path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            axis.imshow(image)
            axis.set_title(
                f"Nhóm {row + 1} | {record.split}\n{record.path.name}", fontsize=9
            )

    figure.suptitle("Một số nhóm ảnh trùng trong toàn bộ dataset", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm tra ảnh trùng chính xác giữa các tập VisDrone."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test-dev"],
        choices=["train", "val", "test-dev", "test-challenge"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/duplicate_check"),
        help="Thư mục lưu file CSV và ảnh xem trước.",
    )
    parser.add_argument("--max-preview-groups", type=int, default=5)
    parser.add_argument("--max-images-per-group", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mở cửa sổ matplotlib sau khi lưu ảnh xem trước.",
    )
    return parser


def main() -> None:
    # Keep Vietnamese terminal output readable on Windows installations whose
    # default console encoding is cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    if args.max_preview_groups < 0:
        raise SystemExit("--max-preview-groups phải >= 0")
    if args.max_images_per_group < 1:
        raise SystemExit("--max-images-per-group phải >= 1")
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers phải >= 1")

    groups, errors, split_counts = find_duplicate_groups(
        args.data_root, args.splits, args.workers
    )
    print_report(groups, errors, split_counts)

    csv_path = args.output_dir / "duplicate_images.csv"
    preview_path = args.output_dir / "duplicate_images_preview.png"
    write_csv_report(groups, csv_path)
    preview_created = save_preview(
        groups,
        preview_path,
        max_groups=args.max_preview_groups,
        max_images_per_group=args.max_images_per_group,
        show=args.show,
    )
    print(f"\nĐã lưu danh sách CSV: {csv_path.resolve()}")
    if preview_created:
        print(f"Đã lưu ảnh xem trước: {preview_path.resolve()}")
    else:
        print("Không tạo ảnh xem trước vì không có nhóm trùng để hiển thị.")


if __name__ == "__main__":
    main()
