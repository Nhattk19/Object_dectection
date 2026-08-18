"""Re-evaluate a trained Faster R-CNN checkpoint with VisDrone DET metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faster_rcnn import (  # noqa: E402
    VisDroneCocoDataset,
    build_faster_rcnn,
    detection_collate_fn,
)
from visdrone_evaluation import evaluate_visdrone  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional processed COCO root; raw validation images are used when omitted.",
    )
    parser.add_argument("--raw-val-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from torchvision.transforms.functional import pil_to_tensor

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a Kaggle GPU")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    experiment = str(config.get("experiment", args.checkpoint.parent.name)).upper()
    model_version = config.get(
        "model_version", "v2" if experiment in {"F4", "F5"} else "v1"
    )
    crowded = bool(
        config.get("crowded_proposals", experiment in {"F3", "F4", "F5"})
    )
    small_anchors = bool(config.get("small_anchors", experiment == "F2"))

    raw_val_root = args.raw_val_root.resolve()
    raw_annotations = raw_val_root / "annotations"
    raw_images = raw_val_root / "images"
    for required in (args.checkpoint, raw_annotations, raw_images):
        if not required.exists():
            raise FileNotFoundError(required)

    device = torch.device("cuda")
    model = build_faster_rcnn(
        pretrained=False,
        small_anchors=small_anchors,
        crowded_proposals=crowded,
        model_version=model_version,
        min_size=int(config.get("min_size", 800)),
        max_size=int(config.get("max_size", 1333)),
        box_detections_per_img=500,
        box_score_thresh=args.score_threshold,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    if args.data_root:
        data_root = args.data_root.resolve()
        val_json = data_root / "annotations" / "instances_val.json"
        image_root = data_root / "images"
        for required in (val_json, image_root):
            if not required.exists():
                raise FileNotFoundError(required)
        dataset = VisDroneCocoDataset(val_json, image_root)
    else:
        class RawVisDroneImageDataset(torch.utils.data.Dataset):
            def __init__(self, root: Path) -> None:
                self.paths = sorted(root.glob("*.jpg"))
                if not self.paths:
                    raise FileNotFoundError(f"No JPG images found in {root}")
                self.images = [
                    {"id": index, "file_name": path.name}
                    for index, path in enumerate(self.paths, start=1)
                ]

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, index: int):
                image = Image.open(self.paths[index]).convert("RGB")
                tensor = pil_to_tensor(image).to(dtype=torch.float32).div(255.0)
                return tensor, {"image_id": torch.tensor(index + 1, dtype=torch.int64)}

        dataset = RawVisDroneImageDataset(raw_images)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    image_names = {
        int(item["id"]): Path(item["file_name"]).stem for item in dataset.images
    }
    run_dir = args.output_dir.resolve() / experiment.lower()
    prediction_dir = run_dir / "visdrone_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=f"{experiment} inference"):
            output = model([image.to(device, non_blocking=True) for image in images])[0]
            image_id = int(targets[0]["image_id"].item())
            lines: list[str] = []
            for box, score, label in zip(
                output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()
            ):
                class_id = int(label)
                if not 1 <= class_id <= 10:
                    continue
                x1, y1, x2, y2 = map(float, box)
                lines.append(
                    f"{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},"
                    f"{float(score):.8f},{class_id},-1,-1\n"
                )
            (prediction_dir / f"{image_names[image_id]}.txt").write_text(
                "".join(lines), encoding="utf-8"
            )

    metrics = evaluate_visdrone(raw_annotations, prediction_dir, raw_images)
    report = {
        "experiment": experiment,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "model_version": model_version,
        "max_detections_per_image": 500,
        "score_threshold": args.score_threshold,
        **metrics,
    }
    (run_dir / "visdrone_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
