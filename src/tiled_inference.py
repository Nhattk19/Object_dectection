"""Tiled object-detection inference helpers.

Tiles overlap so small objects are less likely to be cut at a boundary. Model
outputs are mapped back to full-image coordinates and merged class-wise.
"""

from __future__ import annotations

from collections.abc import Iterator


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return deterministic tile starts that cover ``[0, length)`` fully."""

    if length <= 0:
        raise ValueError("length must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if length <= tile_size:
        return [0]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    last_start = length - tile_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_tile_coordinates(
    image_height: int,
    image_width: int,
    tile_size: int,
    overlap: float,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(x1, y1, x2, y2)`` tiles covering an image."""

    y_starts = tile_starts(image_height, tile_size, overlap)
    x_starts = tile_starts(image_width, tile_size, overlap)
    for y1 in y_starts:
        for x1 in x_starts:
            yield (
                x1,
                y1,
                min(image_width, x1 + tile_size),
                min(image_height, y1 + tile_size),
            )


def tiled_predict(
    model,
    image,
    device,
    *,
    tile_size: int = 640,
    overlap: float = 0.20,
    tile_batch_size: int = 1,
    merge_nms_iou: float = 0.50,
    max_detections: int = 500,
    pre_nms_topk: int = 10_000,
):
    """Run tiled inference and return one merged Torchvision-style output."""

    import torch
    from torchvision.ops import batched_nms

    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive")
    if not 0.0 < merge_nms_iou <= 1.0:
        raise ValueError("merge_nms_iou must satisfy 0 < value <= 1")
    if max_detections <= 0 or pre_nms_topk <= 0:
        raise ValueError("detection limits must be positive")

    height, width = map(int, image.shape[-2:])
    coordinates = list(iter_tile_coordinates(height, width, tile_size, overlap))
    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for batch_start in range(0, len(coordinates), tile_batch_size):
        batch_coordinates = coordinates[batch_start : batch_start + tile_batch_size]
        tiles = [
            image[:, y1:y2, x1:x2].to(device, non_blocking=True)
            for x1, y1, x2, y2 in batch_coordinates
        ]
        outputs = model(tiles)
        for output, (x1, y1, _, _) in zip(outputs, batch_coordinates):
            boxes = output["boxes"].detach().cpu().clone()
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            merged_boxes.append(boxes)
            merged_scores.append(output["scores"].detach().cpu())
            merged_labels.append(output["labels"].detach().cpu())

    if not merged_boxes:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }

    boxes = torch.cat(merged_boxes)
    scores = torch.cat(merged_scores)
    labels = torch.cat(merged_labels)
    if len(scores) > pre_nms_topk:
        top = scores.topk(pre_nms_topk).indices
        boxes, scores, labels = boxes[top], scores[top], labels[top]

    keep = batched_nms(boxes, scores, labels, merge_nms_iou)
    keep = keep[:max_detections]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
