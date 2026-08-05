from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visdrone_data import Annotation, convert_split, evaluate_annotation, xywh_to_yolo


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


def test_convert_split_writes_expected_yolo_label(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    Image.new("RGB", (100, 200)).save(source / "images" / "sample.jpg")
    (source / "annotations" / "sample.txt").write_text(
        "10,20,30,40,1,4,0,0\n0,0,10,10,0,0,0,0\n",
        encoding="utf-8",
    )

    output = tmp_path / "processed"
    stats = convert_split(source, output, "train", image_mode="copy")

    assert stats.rows_total == 2
    assert stats.rows_written == 1
    assert stats.ignored_score == 1
    assert (output / "images" / "train" / "sample.jpg").is_file()
    assert (output / "labels" / "train" / "sample.txt").read_text() == (
        "3 0.250000 0.200000 0.300000 0.200000\n"
    )
