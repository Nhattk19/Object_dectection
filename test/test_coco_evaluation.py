import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("pycocotools")

from coco_evaluation import COCO_METRIC_NAMES, evaluate_coco_predictions


def make_coco(tmp_path, boxes):
    images = []
    annotations = []
    predictions = []
    for index, (width, height) in enumerate(boxes, start=1):
        images.append({"id": index, "file_name": f"val/{index}.jpg", "width": 512, "height": 512})
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [10, 10, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
        predictions.append(
            {
                "image_id": index,
                "category_id": 1,
                "bbox": [10, 10, width, height],
                "score": 0.99,
            }
        )
    payload = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    annotation_file = tmp_path / "instances_val.json"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    return annotation_file, predictions


def test_perfect_predictions_report_all_extended_metrics(tmp_path):
    annotation_file, predictions = make_coco(tmp_path, [(16, 16), (64, 64), (128, 128)])

    metrics = evaluate_coco_predictions(annotation_file, predictions)

    assert tuple(metrics) == COCO_METRIC_NAMES
    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_max_dets_500_increases_recall_for_crowded_image(tmp_path):
    images = [{"id": 1, "file_name": "val/crowded.jpg", "width": 1000, "height": 1000}]
    annotations = []
    predictions = []
    for index in range(150):
        x = 5 * (index % 20)
        y = 5 * (index // 20)
        annotations.append(
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        )
        predictions.append(
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, 2, 2],
                "score": 1.0 - index / 1000,
            }
        )
    payload = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    annotation_file = tmp_path / "instances_val.json"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")

    metrics = evaluate_coco_predictions(annotation_file, predictions)

    assert metrics["AR@100"] < metrics["AR@500"]
    assert metrics["AR@500"] == pytest.approx(1.0)
    assert metrics["AP (50:95)"] == pytest.approx(1.0)
