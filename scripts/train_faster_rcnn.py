"""Train and evaluate Faster R-CNN on processed VisDrone COCO annotations.

The same entry point is used for local smoke tests and full Kaggle GPU runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from augmentation import ControlledDetectionAugmenter
from faster_rcnn import (
    VisDroneCocoDataset,
    build_faster_rcnn,
    detection_collate_fn,
    evaluate_coco,
    load_checkpoint,
    save_checkpoint,
    seed_detection_worker,
    train_one_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "VisDrone",
        help="Processed root containing images/ and annotations/instances_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints",
    )
    parser.add_argument(
        "--experiment", choices=["F0", "F1", "F2", "F3", "F4", "F5"], default="F0"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a checkpoint path, or from last.pth when no path is supplied.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch with two train/validation batches and no weight download.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a full run without CUDA. Intended only for debugging.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset_contract(dataset) -> dict:
    import torch

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    image, target = dataset[0]
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Invalid image tensor: dtype={image.dtype}, shape={tuple(image.shape)}")
    if not bool((image >= 0).all() and (image <= 1).all()):
        raise ValueError("Image values must be in [0, 1]")
    boxes, labels = target["boxes"], target["labels"]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(labels):
        raise ValueError("boxes must be [N,4] and aligned with labels")
    if len(boxes) and not bool(((boxes[:, 2:] - boxes[:, :2]) > 0).all()):
        raise ValueError("All boxes must satisfy x2>x1 and y2>y1")
    if len(labels) and not bool((labels >= 1).all() and (labels <= 10).all()):
        raise ValueError("VisDrone labels must be in 1..10; 0 is background")
    return {
        "images": len(dataset),
        "sample_shape": list(image.shape),
        "sample_boxes": len(boxes),
        "sample_label_min": int(labels.min()) if len(labels) else None,
        "sample_label_max": int(labels.max()) if len(labels) else None,
    }


def write_history(history: list[dict], run_dir: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(history)
    frame.to_csv(run_dir / "history.csv", index=False)
    if frame.empty:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    frame.plot(x="epoch", y="loss_total", marker="o", ax=axes[0], title="Train loss")
    metric_columns = [name for name in ("map", "map_50", "map_small") if name in frame]
    if metric_columns:
        frame.plot(
            x="epoch", y=metric_columns, marker="o", ax=axes[1], title="Validation COCO"
        )
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    import torch

    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("epochs, batch-size, and eval-every must be positive")
    if not args.smoke_test and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "Full training requires CUDA. Use Kaggle GPU or pass --allow-cpu intentionally."
        )

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    data_root = args.data_root.expanduser().resolve()
    image_root = data_root / "images"
    train_json = data_root / "annotations" / "instances_train.json"
    val_json = data_root / "annotations" / "instances_val.json"
    for required in (image_root, train_json, val_json):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run src/preprocess_visdrone.py before training."
            )

    run_dir = args.output_dir.expanduser().resolve() / args.experiment.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_epochs = 1 if args.smoke_test else args.epochs
    batch_size = 1 if args.smoke_test else args.batch_size
    workers = 0 if args.smoke_test else args.workers
    min_size = 320 if args.smoke_test else args.min_size
    max_size = 533 if args.smoke_test else args.max_size
    max_batches = 2 if args.smoke_test else None
    pretrained = not args.smoke_test and not args.no_pretrained
    initialize_from_pretrained = pretrained and not args.resume
    accumulation_steps = (
        2
        if not args.smoke_test
        and args.experiment in {"F4", "F5"}
        and batch_size == 1
        else 1
    )

    config = {
        "experiment": args.experiment,
        "data_root": str(data_root),
        "output_dir": str(run_dir),
        "epochs": effective_epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "workers": workers,
        "seed": args.seed,
        "min_size": min_size,
        "max_size": max_size,
        "pretrained": pretrained,
        "initialize_from_pretrained_this_run": initialize_from_pretrained,
        "augmentation": args.experiment == "F1",
        "small_anchors": args.experiment == "F2",
        "crowded_proposals": args.experiment in {"F3", "F4", "F5"},
        "model_version": "v2" if args.experiment in {"F4", "F5"} else "v1",
        "device": str(device),
        "smoke_test": args.smoke_test,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    augmentation = ControlledDetectionAugmenter(seed=args.seed) if args.experiment == "F1" else None
    train_dataset = VisDroneCocoDataset(train_json, image_root, augmentation=augmentation)
    val_dataset = VisDroneCocoDataset(val_json, image_root)
    dataset_summary = {
        "train": validate_dataset_contract(train_dataset),
        "val": validate_dataset_contract(val_dataset),
    }
    print(json.dumps({"config": config, "dataset": dataset_summary}, indent=2))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_detection_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    model = build_faster_rcnn(
        pretrained=initialize_from_pretrained,
        small_anchors=args.experiment == "F2",
        crowded_proposals=args.experiment in {"F3", "F4", "F5"},
        model_version="v2" if args.experiment in {"F4", "F5"} else "v1",
        min_size=min_size,
        max_size=max_size,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 0.0025 if batch_size <= 2 else 0.005
    optimizer = torch.optim.SGD(
        parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    milestones = [max(1, int(effective_epochs * 0.6)), max(2, int(effective_epochs * 0.8))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:  # Compatibility with the minimum torch 2.1.
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    start_epoch = 0
    history: list[dict] = []
    best_map = -1.0
    if args.resume:
        resume_path = run_dir / "last.pth" if args.resume == "auto" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        metadata = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        previous_experiment = metadata["config"].get("experiment")
        if previous_experiment and previous_experiment != args.experiment:
            raise ValueError(
                f"Checkpoint experiment is {previous_experiment}, requested {args.experiment}"
            )
        start_epoch = metadata["epoch"]
        history = list(metadata["history"])
        best_map = metadata["best_map"]
        print(f"Resumed {resume_path} at epoch {start_epoch}; best mAP={best_map:.4f}")

    if start_epoch >= effective_epochs:
        print(f"Checkpoint already reached {start_epoch}/{effective_epochs} epochs; nothing to train.")
    for epoch in range(start_epoch, effective_epochs):
        print(f"Epoch {epoch + 1}/{effective_epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            max_batches=max_batches,
            accumulation_steps=accumulation_steps,
        )
        scheduler.step()
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == effective_epochs
        val_metrics = (
            evaluate_coco(model, val_loader, val_json, device, max_batches=max_batches)
            if should_evaluate
            else {}
        )
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        current_map = float(val_metrics.get("map", -1.0))
        best_map = max(best_map, current_map)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        checkpoint_args = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            metrics=val_metrics,
            config=config,
            history=history,
            best_map=best_map,
        )
        save_checkpoint(run_dir / "last.pth", **checkpoint_args)
        if current_map >= best_map and should_evaluate:
            save_checkpoint(run_dir / "best.pth", **checkpoint_args)
        write_history(history, run_dir)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    )
    summary = {
        "experiment": args.experiment,
        "completed_epochs": start_epoch if start_epoch >= effective_epochs else effective_epochs,
        "best_map": best_map,
        "peak_vram_gb": peak_vram_gb,
        "last_metrics": history[-1] if history else {},
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
