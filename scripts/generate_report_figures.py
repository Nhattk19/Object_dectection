"""Generate publication-ready figures for the dataset and CNN report."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "VisDrone"

CLASS_NAMES = [
    "Car",
    "Pedestrian",
    "Motor",
    "People",
    "Van",
    "Truck",
    "Bicycle",
    "Bus",
    "Tricycle",
    "Awning-tricycle",
]
CLASS_COUNTS = np.array(
    [187004, 109187, 40378, 38560, 32702, 16284, 13069, 9117, 6387, 4377]
)
SCALE_SPLITS = ["Train", "Validation", "Test-dev"]
SCALE_PERCENTAGES = np.array(
    [
        [60.4667, 34.0019, 5.5314],
        [68.5647, 28.6798, 2.7555],
        [67.6626, 29.1284, 3.2090],
    ]
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def dataset_statistics_figure() -> None:
    fig, (ax_class, ax_scale) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    y = np.arange(len(CLASS_NAMES))
    colors = ["#2166ac" if idx < 2 else "#67a9cf" for idx in range(len(y))]
    ax_class.barh(y, CLASS_COUNTS / 1000, color=colors, edgecolor="none")
    ax_class.set_yticks(y, CLASS_NAMES)
    ax_class.invert_yaxis()
    ax_class.set_xlabel("Retained instances across labeled splits (thousands)")
    ax_class.set_title("(a) Strong class imbalance", loc="left")
    ax_class.grid(axis="x", color="#d9e1ea", linewidth=0.8)
    total = CLASS_COUNTS.sum()
    for idx, count in enumerate(CLASS_COUNTS):
        ax_class.text(
            count / 1000 + 1.6,
            idx,
            f"{100 * count / total:.1f}%",
            va="center",
            fontsize=8.5,
            color="#25364a",
        )
    ax_class.set_xlim(0, 210)

    scale_colors = ["#d73027", "#fdae61", "#4575b4"]
    scale_labels = ["Small (<32² px)", "Medium", "Large (≥96² px)"]
    left = np.zeros(len(SCALE_SPLITS))
    for values, label, color in zip(SCALE_PERCENTAGES.T, scale_labels, scale_colors):
        bars = ax_scale.barh(
            SCALE_SPLITS,
            values,
            left=left,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value, start in zip(bars, values, left):
            if value >= 5:
                ax_scale.text(
                    start + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if label != "Medium" else "#263238",
                    fontsize=9,
                    fontweight="bold",
                )
        left += values
    ax_scale.invert_yaxis()
    ax_scale.set_xlim(0, 100)
    ax_scale.set_xlabel("Share of retained bounding boxes (%)")
    ax_scale.set_title("(b) Small objects dominate every split", loc="left")
    ax_scale.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    ax_scale.grid(axis="x", color="#d9e1ea", linewidth=0.8)

    fig.suptitle("VisDrone dataset characteristics", fontsize=15, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.36, bottom=0.2)
    save_figure(fig, "dataset_statistics")


def eda_diagnostics_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    splits = ["Train", "Validation", "Test-dev"]
    mean_objects = [53.04, 70.73, 46.65]
    over_100 = [10.86, 19.53, 7.89]
    x = np.arange(3)
    bars = axes[0, 0].bar(x, mean_objects, color=["#2166ac", "#d95f02", "#67a9cf"])
    axes[0, 0].set_xticks(x, splits)
    axes[0, 0].set_ylabel("Mean retained objects per image")
    axes[0, 0].set_ylim(0, 82)
    axes[0, 0].set_title("(a) Validation is substantially denser", loc="left")
    axes[0, 0].grid(axis="y", color="#d9e1ea")
    for bar, mean, dense in zip(bars, mean_objects, over_100):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, mean + 1.2, f"{mean:.1f}\n({dense:.1f}% >100)", ha="center", fontsize=9)

    tiny_metrics = [38.91, 72.13, 69.19]
    labels = ["Either side\n<16 px", "Either side\n<32 px", "Area <0.1%\nof image"]
    bars = axes[0, 1].bar(labels, tiny_metrics, color=["#fdae61", "#d73027", "#c51b7d"])
    axes[0, 1].set_ylim(0, 80)
    axes[0, 1].set_ylabel("Share of retained boxes (%)")
    axes[0, 1].set_title("(b) Tiny-object indicators", loc="left")
    axes[0, 1].grid(axis="y", color="#d9e1ea")
    for bar, value in zip(bars, tiny_metrics):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.2f}%", ha="center", fontweight="bold", fontsize=9)

    shift_classes = ["Pedestrian", "People", "Bicycle", "Car", "Van", "Truck", "Tricycle", "Awning-tri.", "Bus", "Motor"]
    val_minus_train = [-0.30, 5.34, 0.27, -5.92, -2.18, -1.82, 1.29, 0.43, -1.08, 3.97]
    shift_colors = ["#d73027" if value < 0 else "#1a9850" for value in val_minus_train]
    axes[1, 0].barh(shift_classes, val_minus_train, color=shift_colors)
    axes[1, 0].axvline(0, color="#34495e", linewidth=0.9)
    axes[1, 0].set_xlabel("Validation minus train class share (percentage points)")
    axes[1, 0].set_title("(c) Class-composition shift", loc="left")
    axes[1, 0].grid(axis="x", color="#d9e1ea")
    axes[1, 0].invert_yaxis()

    input_sizes = [640, 960, 1280]
    under_8 = [46.63, 27.63, 16.65]
    bars = axes[1, 1].bar([str(size) for size in input_sizes], under_8, color=["#d73027", "#fdae61", "#4575b4"])
    axes[1, 1].set_xlabel("Simulated square letterbox size")
    axes[1, 1].set_ylabel("Boxes with short side <8 px (%)")
    axes[1, 1].set_title("(d) Resolution preserves small-object signal", loc="left")
    axes[1, 1].grid(axis="y", color="#d9e1ea")
    for bar, value in zip(bars, under_8):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("Key diagnostics from the VisDrone EDA notebook", fontsize=15, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.38, wspace=0.3)
    save_figure(fig, "eda_diagnostics")


def class_difficulty_figure() -> None:
    names = ["Pedestrian", "People", "Bicycle", "Car", "Van", "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"]
    tiny = np.array([88.88, 92.44, 81.38, 57.20, 52.56, 38.12, 53.97, 54.51, 30.89, 83.96])
    heavy = np.array([5.22, 10.90, 13.58, 10.31, 8.12, 8.66, 13.54, 13.05, 9.10, 15.92])
    truncated = np.array([1.92, 0.98, 2.42, 5.44, 5.53, 7.61, 3.38, 4.36, 9.63, 2.01])
    y = np.arange(len(names))

    fig, (ax_tiny, ax_visibility) = plt.subplots(1, 2, figsize=(13.2, 6.1))
    ax_tiny.barh(y, tiny, color="#d73027")
    ax_tiny.set_yticks(y, names)
    ax_tiny.invert_yaxis()
    ax_tiny.set_xlim(0, 100)
    ax_tiny.set_xlabel("Boxes with relative area <0.1% (%)")
    ax_tiny.set_title("(a) Tiny objects affect every class", loc="left")
    ax_tiny.grid(axis="x", color="#d9e1ea")
    for idx, value in enumerate(tiny):
        ax_tiny.text(value + 1.0, idx, f"{value:.1f}", va="center", fontsize=8.5)

    height = 0.36
    ax_visibility.barh(y - height / 2, heavy, height=height, label="Heavily occluded", color="#7b3294")
    ax_visibility.barh(y + height / 2, truncated, height=height, label="Truncated", color="#008837")
    ax_visibility.set_yticks(y, names)
    ax_visibility.invert_yaxis()
    ax_visibility.set_xlabel("Share of class boxes (%)")
    ax_visibility.set_title("(b) Visibility difficulty differs by class", loc="left")
    ax_visibility.grid(axis="x", color="#d9e1ea")
    ax_visibility.legend(frameon=False, loc="upper right")

    fig.suptitle("Per-class scale, occlusion, and truncation", fontsize=15, fontweight="bold", y=1.01)
    fig.subplots_adjust(wspace=0.34)
    save_figure(fig, "class_difficulty")


def density_and_bbox_geometry_figure() -> None:
    """Summarize object counts, box areas, and box aspect ratios."""

    split_colors = {"Train": "#2166ac", "Validation": "#d95f02", "Test-dev": "#67a9cf"}
    split_payloads = {}
    for label, split in zip(("Train", "Validation", "Test-dev"), ("train", "val", "test-dev")):
        path = DATA_ROOT / "annotations" / f"instances_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter(int(item["image_id"]) for item in payload["annotations"])
        per_image = np.array([counts.get(int(info["id"]), 0) for info in payload["images"]])
        widths = np.array([float(item["bbox"][2]) for item in payload["annotations"]])
        heights = np.array([float(item["bbox"][3]) for item in payload["annotations"]])
        split_payloads[label] = {
            "counts": per_image,
            "areas": widths * heights,
            "ratios": widths / heights,
        }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))

    density_bins = np.geomspace(1, 1000, 42)
    for label, values in split_payloads.items():
        counts = values["counts"]
        axes[0].hist(
            counts,
            bins=density_bins,
            histtype="step",
            linewidth=2,
            color=split_colors[label],
            label=f"{label}: mean {counts.mean():.1f}, median {np.median(counts):.0f}, max {counts.max():.0f}",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Retained objects per image (log scale)")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("(a) Object-density distribution", loc="left")
    axes[0].grid(color="#d9e1ea", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7.5)

    area_bins = np.geomspace(1, 5e5, 65)
    for label, values in split_payloads.items():
        axes[1].hist(
            values["areas"],
            bins=area_bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=split_colors[label],
            label=label,
        )
    axes[1].axvline(32**2, color="#d73027", linestyle="--", linewidth=1.2, label="32²: small/medium")
    axes[1].axvline(96**2, color="#4575b4", linestyle="--", linewidth=1.2, label="96²: medium/large")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Bounding-box area (px², log scale)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) Bounding-box area", loc="left")
    axes[1].grid(color="#d9e1ea", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=7.2)

    all_ratios = np.concatenate([values["ratios"] for values in split_payloads.values()])
    ratio_bins = np.geomspace(0.05, 13, 65)
    axes[2].hist(all_ratios, bins=ratio_bins, color="#7b3294", alpha=0.82)
    axes[2].axvline(1.0, color="#263238", linestyle="--", linewidth=1.2, label="square box")
    axes[2].axvline(np.median(all_ratios), color="#fdae61", linewidth=1.7, label=f"median = {np.median(all_ratios):.2f}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Bounding-box aspect ratio, width/height (log scale)")
    axes[2].set_ylabel("Number of boxes")
    axes[2].set_title("(c) Bounding-box shape", loc="left")
    axes[2].grid(color="#d9e1ea", linewidth=0.8)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Object density and bounding-box geometry", fontsize=15, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.32)
    save_figure(fig, "density_bbox_geometry")


def qualitative_eda_gallery() -> None:
    """Create a deterministic six-image audit gallery from the training split."""

    payload = json.loads((DATA_ROOT / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    info_by_stem = {Path(info["file_name"]).stem: info for info in payload["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    summaries = []
    for info in payload["images"]:
        annotations = annotations_by_image[int(info["id"])]
        relative_areas = [float(a["area"]) / (int(info["width"]) * int(info["height"])) for a in annotations]
        summaries.append(
            {
                "info": info,
                "count": len(annotations),
                "median_relative_area": float(np.median(relative_areas)) if relative_areas else 1.0,
            }
        )

    crowded = max(summaries, key=lambda item: item["count"])
    tiny_candidates = [item for item in summaries if 40 <= item["count"] <= 160]
    tiny = min(tiny_candidates, key=lambda item: item["median_relative_area"])

    raw_annotations = PROJECT_ROOT / "data/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations"
    heavy_scores = []
    for annotation_path in raw_annotations.glob("*.txt"):
        info = info_by_stem.get(annotation_path.stem)
        if info is None:
            continue
        heavy = 0
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) >= 8 and int(float(fields[4])) > 0 and 1 <= int(float(fields[5])) <= 10:
                heavy += int(float(fields[7])) == 2
        heavy_scores.append((heavy, info))
    heavy_info = max(heavy_scores, key=lambda item: item[0])[1]
    heavy = next(item for item in summaries if item["info"]["id"] == heavy_info["id"])

    illumination_pool = [item for item in summaries if 10 <= item["count"] <= 180][::20]
    illumination = []
    for item in illumination_pool:
        image_path = DATA_ROOT / "images" / item["info"]["file_name"]
        with Image.open(image_path) as image:
            sample = image.convert("L")
            sample.thumbnail((96, 96))
            illumination.append((float(np.asarray(sample, dtype=np.float32).mean()), item))
    dark = min(illumination, key=lambda item: item[0])[1]
    bright = max(illumination, key=lambda item: item[0])[1]
    viewpoint_candidates = [item for item in summaries if 10 <= item["count"] <= 80]
    viewpoint = max(viewpoint_candidates, key=lambda item: item["median_relative_area"])

    proposed = [
        ("Tiny objects", tiny),
        ("Crowded traffic", crowded),
        ("Heavy occlusion", heavy),
        ("Low illumination", dark),
        ("Bright illumination", bright),
        ("Scale/viewpoint variation", viewpoint),
    ]
    selected = []
    used_ids = set()
    fallback = sorted(summaries, key=lambda item: item["median_relative_area"])
    for title, item in proposed:
        if int(item["info"]["id"]) in used_ids:
            item = next(candidate for candidate in fallback if int(candidate["info"]["id"]) not in used_ids)
        used_ids.add(int(item["info"]["id"]))
        selected.append((title, item))

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.2))
    for ax, (title, item) in zip(axes.flat, selected):
        info = item["info"]
        image_path = DATA_ROOT / "images" / info["file_name"]
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        for annotation in annotations_by_image[int(info["id"])]:
            x, y, width, height = map(float, annotation["bbox"])
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ff3b30", linewidth=0.45, alpha=0.8))
        ax.set_title(f"{title}\n{Path(info['file_name']).stem} · {item['count']} objects", loc="left", fontsize=10)
        ax.axis("off")

    fig.suptitle("Representative VisDrone conditions used for qualitative audit", fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.18, wspace=0.08)
    save_figure(fig, "qualitative_eda_gallery")


def annotated_dataset_example() -> None:
    annotation_path = DATA_ROOT / "annotations" / "instances_train.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    candidates = []
    for image_info in payload["images"]:
        image_id = int(image_info["id"])
        count = len(annotations_by_image[image_id])
        image_path = DATA_ROOT / "images" / image_info["file_name"]
        if image_path.is_file() and 80 <= count <= 110:
            candidates.append((abs(count - 95), image_info, image_path))
    if not candidates:
        raise RuntimeError("No suitable annotated training image was found")
    _, image_info, image_path = min(candidates, key=lambda item: item[0])
    annotations = annotations_by_image[int(image_info["id"])]
    category_names = {
        int(category["id"]): category["name"] for category in payload["categories"]
    }
    category_counts = Counter(int(item["category_id"]) for item in annotations)

    image = Image.open(image_path).convert("RGB")
    fig, (ax_original, ax_boxes) = plt.subplots(1, 2, figsize=(14, 5.2))
    for axis in (ax_original, ax_boxes):
        axis.imshow(image)
        axis.axis("off")
    ax_original.set_title("(a) Original aerial image", loc="left")
    ax_boxes.set_title(f"(b) {len(annotations)} retained objects", loc="left")

    palette = plt.get_cmap("tab10")
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        category_id = int(annotation["category_id"])
        ax_boxes.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=palette(category_id - 1),
                linewidth=1.0,
                alpha=0.95,
            )
        )
    summary = ", ".join(
        f"{category_names[class_id]}: {count}"
        for class_id, count in category_counts.most_common(4)
    )
    fig.suptitle(
        "Small and crowded objects in VisDrone\n" + summary,
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.subplots_adjust(wspace=0.025)
    save_figure(fig, "visdrone_annotated_example")


def add_box(ax, x, y, width, height, title, subtitle, color):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#27405a",
        linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontweight="bold", fontsize=10)
    ax.text(x + width / 2, y + height * 0.31, subtitle, ha="center", va="center", fontsize=7.7, color="#34495e")
    return box


def add_arrow(ax, start, end, color="#506b82"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(15, 5.7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("white")

    add_box(ax, 0.25, 2.3, 1.35, 1.15, "Input image", "RGB, variable size", "#e8f1fa")
    add_box(ax, 2.0, 2.3, 1.55, 1.15, "ResNet-50", "CNN backbone", "#dceef2")
    add_box(ax, 4.05, 2.3, 1.55, 1.15, "FPN", "P2–P6 features", "#d8eadf")
    add_box(ax, 6.1, 3.65, 1.65, 1.15, "RPN", "objectness + boxes", "#fff0cf")
    add_box(ax, 8.2, 3.65, 1.65, 1.15, "Proposals", "top-N after NMS", "#ffe2bd")
    add_box(ax, 8.2, 1.05, 1.65, 1.15, "RoI Align", "proposal features", "#eadcf3")
    add_box(ax, 10.3, 1.05, 1.65, 1.15, "Detection head", "class + box regression", "#e3ddf5")
    add_box(ax, 12.4, 1.05, 2.0, 1.15, "Final detections", "class, score, refined box", "#f6dce5")

    add_arrow(ax, (1.6, 2.88), (2.0, 2.88))
    add_arrow(ax, (3.55, 2.88), (4.05, 2.88))
    add_arrow(ax, (5.6, 3.05), (6.1, 4.1))
    add_arrow(ax, (7.75, 4.23), (8.2, 4.23))
    add_arrow(ax, (9.03, 3.65), (9.03, 2.2))
    add_arrow(ax, (5.6, 2.65), (8.2, 1.64), color="#3f7d5a")
    add_arrow(ax, (9.85, 1.63), (10.3, 1.63))
    add_arrow(ax, (11.95, 1.63), (12.4, 1.63))

    ax.text(6.8, 5.15, "Stage 1: candidate generation", ha="center", fontsize=11, fontweight="bold", color="#8a5b00")
    ax.text(10.35, 0.45, "Stage 2: proposal classification and localization", ha="center", fontsize=11, fontweight="bold", color="#5d3b75")
    ax.text(7.0, 2.15, "multi-scale feature maps", ha="center", fontsize=8.5, color="#3f7d5a")
    ax.set_title("Faster R-CNN ResNet-50-FPN used in the CNN experiments", fontsize=15, fontweight="bold", pad=12)
    save_figure(fig, "faster_rcnn_architecture")


def experiment_progression_figure() -> None:
    runs = ["F0", "F1", "F2", "F3", "F4", "F5"]
    ap = np.array([23.972, 23.733, 24.508, 25.223, 25.893, 27.121])
    ap50 = np.array([42.603, 43.881, 45.513, 45.201, 44.652, 47.308])
    ap75 = np.array([23.507, 22.998, 23.028, 24.482, 25.958, 26.806])
    ar100 = np.array([40.604, 39.861, 43.029, 43.150, 44.407, 47.024])
    ar500 = np.array([40.618, 39.879, 43.272, 43.237, 44.459, 47.114])

    fig, (ax_ap, ax_ar) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    x = np.arange(len(runs))
    for values, label, color, marker in [
        (ap, "AP@[.50:.95]", "#2166ac", "o"),
        (ap50, "AP50", "#1b9e77", "s"),
        (ap75, "AP75", "#d95f02", "^"),
    ]:
        ax_ap.plot(x, values, label=label, color=color, marker=marker, linewidth=2.2, markersize=6)
    ax_ap.set_xticks(x, runs)
    ax_ap.set_ylabel("Average Precision (%)")
    ax_ap.set_title("(a) Detection precision", loc="left")
    ax_ap.grid(color="#d9e1ea", linewidth=0.8)
    ax_ap.legend(frameon=False, loc="best")
    ax_ap.axvspan(4.7, 5.3, color="#f4c95d", alpha=0.18)
    ax_ap.annotate("Best AP\n27.121%", (5, ap[-1]), xytext=(4.0, 33), arrowprops={"arrowstyle": "->", "color": "#34495e"}, fontsize=9, fontweight="bold")

    ax_ar.plot(x, ar100, label="AR@100", color="#7570b3", marker="o", linewidth=2.2)
    ax_ar.plot(x, ar500, label="AR@500", color="#e7298a", marker="s", linewidth=2.2)
    ax_ar.set_xticks(x, runs)
    ax_ar.set_ylabel("Average Recall (%)")
    ax_ar.set_title("(b) Recall in crowded scenes", loc="left")
    ax_ar.grid(color="#d9e1ea", linewidth=0.8)
    ax_ar.legend(frameon=False, loc="best")
    ax_ar.axvspan(4.7, 5.3, color="#f4c95d", alpha=0.18)
    ax_ar.annotate("F5 AR@500\n47.114%", (5, ar500[-1]), xytext=(3.8, 41.0), arrowprops={"arrowstyle": "->", "color": "#34495e"}, fontsize=9, fontweight="bold")

    fig.suptitle("Unified COCO-style CNN evaluation at maxDet=500", fontsize=15, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.3)
    save_figure(fig, "cnn_experiment_progression")


def f5_training_dynamics_figure() -> None:
    history_path = PROJECT_ROOT / "models" / "checkpoints" / "f5" / "history.csv"
    with history_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = np.array([int(row["epoch"]) for row in rows])
    loss = np.array([float(row["loss_total"]) for row in rows])
    ap = 100 * np.array([float(row["visdrone_ap"]) for row in rows])
    ap50 = 100 * np.array([float(row["visdrone_ap50"]) for row in rows])
    best_index = int(np.argmax(ap))

    fig, ax_ap = plt.subplots(figsize=(9.5, 5.0))
    ax_loss = ax_ap.twinx()
    ax_ap.plot(epochs, ap, color="#2166ac", linewidth=2.4, marker="o", markersize=3.5, label="VisDrone AP")
    ax_ap.plot(epochs, ap50, color="#1b9e77", linewidth=1.8, alpha=0.75, label="VisDrone AP50")
    ax_loss.plot(epochs, loss, color="#d95f02", linewidth=2.0, linestyle="--", label="Training loss")
    ax_ap.scatter(epochs[best_index], ap[best_index], s=90, color="#f4c95d", edgecolor="#6b4d00", zorder=5)
    ax_ap.annotate(
        f"best.pth: epoch {epochs[best_index]}\nAP = {ap[best_index]:.2f}%",
        (epochs[best_index], ap[best_index]),
        xytext=(1.2, 32.2),
        arrowprops={"arrowstyle": "->", "color": "#34495e"},
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#aeb6bf", "alpha": 0.96},
    )
    ax_ap.axvline(15, color="#7f8c8d", linewidth=1, linestyle=":")
    ax_ap.axvline(20, color="#7f8c8d", linewidth=1, linestyle=":")
    ax_ap.text(15.2, ax_ap.get_ylim()[0] + 0.6, "LR ×0.1", fontsize=8, color="#566573")
    ax_ap.text(20.2, ax_ap.get_ylim()[0] + 0.6, "LR ×0.1", fontsize=8, color="#566573")
    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("Validation metric (%)", color="#2166ac")
    ax_loss.set_ylabel("Training loss", color="#d95f02")
    ax_ap.grid(color="#d9e1ea", linewidth=0.8)
    handles_1, labels_1 = ax_ap.get_legend_handles_labels()
    handles_2, labels_2 = ax_loss.get_legend_handles_labels()
    ax_ap.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    ax_ap.set_title("F5 training dynamics: lower loss did not imply higher validation AP", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.24)
    save_figure(fig, "f5_training_dynamics")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    dataset_statistics_figure()
    eda_diagnostics_figure()
    class_difficulty_figure()
    density_and_bbox_geometry_figure()
    qualitative_eda_gallery()
    annotated_dataset_example()
    architecture_figure()
    experiment_progression_figure()
    f5_training_dynamics_figure()
    print(f"Generated report figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
