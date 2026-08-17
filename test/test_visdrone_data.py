from pathlib import Path
import json
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convert_visdrone_to_detr import convert_dataset_to_detr
from convert_visdrone_to_yolo import convert_dataset_to_yolo, xywh_to_yolo
from visdrone_data import Annotation, evaluate_annotation, preprocess_dataset


def test_valid_box_is_converted_to_normalized_yolo():
    annotation = Annotation(10, 20, 30, 40, 1, 4, 0, 0)
    status, clipped, was_clipped = evaluate_annotation(annotation, 100, 200)
    assert status == "keep"
    assert clipped == (10, 20, 30, 40)
    assert not was_clipped
    assert xywh_to_yolo(clipped, 100, 200) == (0.25, 0.2, 0.3, 0.2)


def test_box_is_clipped_to_image_bounds():
    annotation = Annotation(-10, 80, 30, 40, 1, 1, 0, 0)
    status, clipped, was_clipped = evaluate_annotation(annotation, 100, 100)
    assert status == "keep"
    assert clipped == (0.0, 80, 20.0, 20.0)
    assert was_clipped


def test_ignored_rows_do_not_become_yolo_labels():
    ignored_region = Annotation(0, 0, 20, 20, 0, 0, 0, 0)
    other_class = Annotation(0, 0, 20, 20, 1, 11, 0, 0)
    assert evaluate_annotation(ignored_region, 100, 100)[0] == "ignored_score"
    assert evaluate_annotation(other_class, 100, 100)[0] == "ignored_category"


def test_preprocess_then_convert_to_yolo_and_detr(tmp_path):
    source = tmp_path / "raw" / "VisDrone2019-DET-train"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    Image.new("RGB", (100, 200)).save(source / "images" / "sample.jpg")
    (source / "annotations" / "sample.txt").write_text(
        "10,20,30,40,1,4,0,0\n0,0,10,10,0,0,0,0\n",
        encoding="utf-8",
    )

    processed = tmp_path / "processed"
    reports = preprocess_dataset(
        tmp_path / "raw", processed, ["train"], image_mode="copy"
    )
    stats = reports[0]

    assert stats.rows_total == 2
    assert stats.objects_written == 1
    assert stats.ignored_score == 1
    assert stats.records_written == 1
    assert (processed / "images" / "train" / "sample.jpg").is_file()
    assert (processed / "dataset_manifest.json").is_file()

    record = json.loads(
        (processed / "annotations" / "train.jsonl").read_text(encoding="utf-8")
    )
    assert record == {
        "image_id": 1,
        "file_name": "sample.jpg",
        "width": 100,
        "height": 200,
        "objects": [
            {
                "category_id": 3,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "area": 1200.0,
                "truncation": 0,
                "occlusion": 0,
            }
        ],
    }

    yolo_output = tmp_path / "yolo"
    yolo_reports = convert_dataset_to_yolo(
        processed, yolo_output, image_mode="copy"
    )
    assert yolo_reports[0].objects_written == 1
    assert (yolo_output / "labels" / "train" / "sample.txt").read_text() == (
        "3 0.250000 0.200000 0.300000 0.200000\n"
    )
    assert (yolo_output / "images" / "train" / "sample.jpg").is_file()
    assert (yolo_output / "visdrone_yolo.yaml").is_file()

    detr_output = tmp_path / "detr"
    detr_reports = convert_dataset_to_detr(
        processed, detr_output, image_mode="copy"
    )
    assert detr_reports[0].output_split == "train"
    coco = json.loads(
        (detr_output / "train" / "_annotations.coco.json").read_text(
            encoding="utf-8"
        )
    )
    assert coco["images"] == [
        {"id": 1, "file_name": "sample.jpg", "width": 100, "height": 200}
    ]
    assert coco["annotations"][0]["category_id"] == 3
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert [category["id"] for category in coco["categories"]] == list(range(10))
    assert (detr_output / "train" / "sample.jpg").is_file()


def test_detr_maps_visdrone_split_names():
    from convert_visdrone_to_detr import DETR_SPLIT_NAMES

    assert DETR_SPLIT_NAMES["train"] == "train"
    assert DETR_SPLIT_NAMES["val"] == "valid"
    assert DETR_SPLIT_NAMES["test-dev"] == "test"
