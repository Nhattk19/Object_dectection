import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from augmentation import AugmentationConfig, ControlledDetectionAugmenter
from faster_rcnn import (
    VisDroneCocoDataset,
    build_faster_rcnn,
    detection_collate_fn,
    load_checkpoint,
    predictions_to_coco,
    save_checkpoint,
)


def make_dataset(tmp_path):
    image_root = tmp_path / "images"
    (image_root / "train").mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(120, 130, 140)).save(
        image_root / "train" / "sample.jpg"
    )
    payload = {
        "images": [{"id": 7, "file_name": "train/sample.jpg", "width": 100, "height": 80}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 4, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
        "categories": [{"id": 4, "name": "car"}],
    }
    annotation_file = tmp_path / "instances_train.json"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    return annotation_file, image_root


def test_dataset_returns_torchvision_detection_contract(tmp_path):
    annotation_file, image_root = make_dataset(tmp_path)
    dataset = VisDroneCocoDataset(annotation_file, image_root)

    image, target = dataset[0]

    assert image.shape == (3, 80, 100)
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    torch.testing.assert_close(target["boxes"], torch.tensor([[10, 20, 40, 60]], dtype=torch.float32))
    assert target["labels"].tolist() == [4]
    assert target["image_id"].item() == 7
    assert target["area"].tolist() == [1200]
    assert dataset.get_height_and_width(0) == (80, 100)


def test_dataset_augmentation_keeps_target_fields_synchronized(tmp_path):
    annotation_file, image_root = make_dataset(tmp_path)
    config = AugmentationConfig(
        horizontal_flip_probability=1,
        vertical_flip_probability=0,
        affine_probability=0,
        photometric_probability=0,
        blur_probability=0,
    )
    dataset = VisDroneCocoDataset(
        annotation_file,
        image_root,
        augmentation=ControlledDetectionAugmenter(config, seed=42),
    )

    _, target = dataset[0]

    torch.testing.assert_close(target["boxes"], torch.tensor([[60, 20, 90, 60]], dtype=torch.float32))
    assert len(target["boxes"]) == len(target["labels"]) == len(target["area"])


def test_collate_supports_variable_image_sizes():
    batch = [
        (torch.zeros(3, 10, 20), {"image_id": torch.tensor(1)}),
        (torch.zeros(3, 30, 40), {"image_id": torch.tensor(2)}),
    ]
    images, targets = detection_collate_fn(batch)
    assert len(images) == len(targets) == 2
    assert images[0].shape != images[1].shape


def test_model_builder_replaces_head_and_small_anchors_without_download():
    model = build_faster_rcnn(
        pretrained=False,
        small_anchors=True,
        min_size=128,
        max_size=160,
        box_detections_per_img=50,
    )
    assert model.roi_heads.box_predictor.cls_score.out_features == 11
    assert model.rpn.anchor_generator.sizes == ((8,), (16,), (32,), (64,), (128,))
    assert model.roi_heads.detections_per_img == 50


def test_predictions_convert_xyxy_to_coco_xywh():
    outputs = [{
        "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
        "labels": torch.tensor([4]),
        "scores": torch.tensor([0.9]),
    }]
    targets = [{"image_id": torch.tensor(7)}]
    result = predictions_to_coco(outputs, targets)
    assert result[0]["image_id"] == 7
    assert result[0]["category_id"] == 4
    assert result[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert result[0]["score"] == pytest.approx(0.9)


def test_checkpoint_round_trip_includes_resume_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    target = tmp_path / "checkpoint.pth"
    history = [{"epoch": 1, "map": 0.25}]

    save_checkpoint(
        target,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        metrics={"map": 0.25},
        config={"experiment": "F0"},
        history=history,
        best_map=0.25,
    )
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    metadata = load_checkpoint(
        target,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert metadata["epoch"] == 1
    assert metadata["history"] == history
    assert metadata["best_map"] == pytest.approx(0.25)
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(expected, actual)
