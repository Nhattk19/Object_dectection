from pathlib import Path
import sys

from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visdrone_evaluation import evaluate_visdrone


def write_case(tmp_path, ground_truth, detections):
    annotation_root = tmp_path / "annotations"
    detection_root = tmp_path / "detections"
    image_root = tmp_path / "images"
    annotation_root.mkdir()
    detection_root.mkdir()
    image_root.mkdir()
    Image.new("RGB", (100, 100)).save(image_root / "sample.jpg")
    (annotation_root / "sample.txt").write_text(ground_truth, encoding="utf-8")
    (detection_root / "sample.txt").write_text(detections, encoding="utf-8")
    return annotation_root, detection_root, image_root


def test_perfect_detection_scores_one(tmp_path):
    roots = write_case(
        tmp_path,
        "10,10,20,20,1,4,0,0\n",
        "10,10,20,20,0.9,4,-1,-1\n",
    )
    metrics = evaluate_visdrone(*roots)
    assert metrics["visdrone_ap"] == pytest.approx(1.0)
    assert metrics["visdrone_ap50"] == pytest.approx(1.0)
    assert metrics["visdrone_ar500"] == pytest.approx(1.0)


def test_detection_inside_ignore_region_is_not_false_positive(tmp_path):
    roots = write_case(
        tmp_path,
        "1,1,40,40,0,0,0,0\n10,60,20,20,1,4,0,0\n",
        "5,5,10,10,0.99,4,-1,-1\n10,60,20,20,0.8,4,-1,-1\n",
    )
    metrics = evaluate_visdrone(*roots)
    assert metrics["visdrone_ap"] == pytest.approx(1.0)
