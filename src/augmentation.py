"""Bounding-box-safe online augmentation for VisDrone detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.2
    affine_probability: float = 0.6
    max_rotation_degrees: float = 10.0
    max_translate_fraction: float = 0.05
    scale_min: float = 0.9
    scale_max: float = 1.1
    photometric_probability: float = 0.8
    brightness_min: float = 0.85
    brightness_max: float = 1.15
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    blur_probability: float = 0.1
    blur_radius_max: float = 1.0
    min_box_size: float = 2.0
    min_visibility: float = 0.30
    fill_color: tuple[int, int, int] = (114, 114, 114)


def _boxes_array(boxes) -> np.ndarray:
    result = np.asarray(boxes, dtype=np.float32)
    return result.reshape(-1, 4).copy()


def filter_and_clip_boxes(
    boxes,
    width: int,
    height: int,
    *,
    original_areas: Optional[np.ndarray] = None,
    min_size: float = 2.0,
    min_visibility: float = 0.30,
):
    clipped = _boxes_array(boxes)
    if not len(clipped):
        return clipped, np.zeros(0, dtype=bool)
    if original_areas is None:
        original_areas = np.maximum(0, clipped[:, 2] - clipped[:, 0]) * np.maximum(
            0, clipped[:, 3] - clipped[:, 1]
        )
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    widths = clipped[:, 2] - clipped[:, 0]
    heights = clipped[:, 3] - clipped[:, 1]
    areas = np.maximum(0, widths) * np.maximum(0, heights)
    visibility = areas / np.maximum(np.asarray(original_areas), 1e-9)
    keep = (
        np.isfinite(clipped).all(axis=1)
        & (widths >= min_size)
        & (heights >= min_size)
        & (visibility >= min_visibility)
    )
    return clipped[keep], keep


def horizontal_flip(image: Image.Image, boxes):
    result = _boxes_array(boxes)
    if len(result):
        x1, x2 = result[:, 0].copy(), result[:, 2].copy()
        result[:, 0], result[:, 2] = image.width - x2, image.width - x1
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), result


def vertical_flip(image: Image.Image, boxes):
    result = _boxes_array(boxes)
    if len(result):
        y1, y2 = result[:, 1].copy(), result[:, 3].copy()
        result[:, 1], result[:, 3] = image.height - y2, image.height - y1
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), result


def affine_transform(
    image: Image.Image,
    boxes,
    *,
    angle_degrees: float,
    scale: float,
    translate_x: float,
    translate_y: float,
    fill_color=(114, 114, 114),
):
    radians = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(radians) * scale, np.sin(radians) * scale
    cx, cy = image.width / 2, image.height / 2
    forward = (
        np.array([[1, 0, cx + translate_x], [0, 1, cy + translate_y], [0, 0, 1]])
        @ np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    )
    inverse = np.linalg.inv(forward)
    output = image.transform(
        image.size,
        Image.Transform.AFFINE,
        tuple(inverse[:2].reshape(-1)),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill_color,
    )
    source = _boxes_array(boxes)
    if not len(source):
        return output, source
    corners = np.stack(
        [source[:, [0, 1]], source[:, [2, 1]], source[:, [2, 3]], source[:, [0, 3]]],
        axis=1,
    )
    homogeneous = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], axis=2)
    transformed = homogeneous @ forward.T
    result = np.concatenate(
        [transformed[:, :, :2].min(axis=1), transformed[:, :, :2].max(axis=1)], axis=1
    )
    return output, result.astype(np.float32)


class ControlledDetectionAugmenter:
    def __init__(self, config: Optional[AugmentationConfig] = None, *, seed=None):
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image, boxes, labels):
        image = image.convert("RGB")
        boxes = _boxes_array(boxes)
        labels = np.asarray(labels, dtype=np.int64).copy()
        initial_count = len(boxes)
        applied = []
        if self.rng.random() < self.config.horizontal_flip_probability:
            image, boxes = horizontal_flip(image, boxes)
            applied.append("horizontal_flip")
        if self.rng.random() < self.config.vertical_flip_probability:
            image, boxes = vertical_flip(image, boxes)
            applied.append("vertical_flip")
        if self.rng.random() < self.config.affine_probability:
            scale = float(self.rng.uniform(self.config.scale_min, self.config.scale_max))
            original_areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
                0, boxes[:, 3] - boxes[:, 1]
            )
            image, boxes = affine_transform(
                image,
                boxes,
                angle_degrees=float(self.rng.uniform(-self.config.max_rotation_degrees, self.config.max_rotation_degrees)),
                scale=scale,
                translate_x=float(self.rng.uniform(-self.config.max_translate_fraction, self.config.max_translate_fraction) * image.width),
                translate_y=float(self.rng.uniform(-self.config.max_translate_fraction, self.config.max_translate_fraction) * image.height),
                fill_color=self.config.fill_color,
            )
            boxes, keep = filter_and_clip_boxes(
                boxes,
                image.width,
                image.height,
                original_areas=original_areas * scale**2,
                min_size=self.config.min_box_size,
                min_visibility=self.config.min_visibility,
            )
            labels = labels[keep]
            applied.append("affine")
        if self.rng.random() < self.config.photometric_probability:
            image = ImageEnhance.Brightness(image).enhance(
                float(self.rng.uniform(self.config.brightness_min, self.config.brightness_max))
            )
            image = ImageEnhance.Contrast(image).enhance(
                float(self.rng.uniform(self.config.contrast_min, self.config.contrast_max))
            )
            applied.append("brightness_contrast")
        if self.rng.random() < self.config.blur_probability:
            image = image.filter(
                ImageFilter.GaussianBlur(float(self.rng.uniform(0.1, self.config.blur_radius_max)))
            )
            applied.append("gaussian_blur")
        audit = {
            "applied": applied,
            "initial_boxes": initial_count,
            "retained_boxes": len(boxes),
            "dropped_boxes": initial_count - len(boxes),
        }
        return image, boxes, labels, audit


def paste_bbox_patch(
    target_image: Image.Image,
    target_boxes,
    target_labels,
    donor_image: Image.Image,
    donor_box: Sequence[float],
    donor_label: int,
    *,
    destination_xy: tuple[int, int],
    max_intersection_over_area: float = 0.10,
    feather_radius: float = 1.0,
):
    """Experimental rectangle copy-paste retained for the augmentation demo."""

    boxes = _boxes_array(target_boxes)
    labels = np.asarray(target_labels, dtype=np.int64)
    x1, y1, x2, y2 = np.rint(donor_box).astype(int)
    patch = donor_image.convert("RGB").crop((x1, y1, x2, y2))
    px1, py1 = destination_xy
    candidate = np.array([px1, py1, px1 + patch.width, py1 + patch.height], dtype=np.float32)
    if candidate[0] < 0 or candidate[1] < 0 or candidate[2] > target_image.width or candidate[3] > target_image.height:
        raise ValueError("pasted patch must fit completely inside target_image")
    if len(boxes):
        ix1, iy1 = np.maximum(candidate[0], boxes[:, 0]), np.maximum(candidate[1], boxes[:, 1])
        ix2, iy2 = np.minimum(candidate[2], boxes[:, 2]), np.minimum(candidate[3], boxes[:, 3])
        intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        ioa = intersection / max(1e-9, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
        maximum_ioa = float(ioa.max(initial=0))
    else:
        maximum_ioa = 0.0
    if maximum_ioa > max_intersection_over_area:
        raise ValueError("candidate overlaps existing objects")
    mask = Image.new("L", patch.size, 255).filter(ImageFilter.GaussianBlur(feather_radius))
    output = target_image.convert("RGB").copy()
    output.paste(patch, destination_xy, mask)
    return (
        output,
        np.concatenate([boxes, candidate[None]], axis=0),
        np.concatenate([labels, np.array([donor_label], dtype=np.int64)]),
        {
            "operation": "experimental_bbox_copy_paste",
            "destination_box": candidate.tolist(),
            "maximum_intersection_over_area": maximum_ioa,
            "warning": "rectangular context copied because instance masks are unavailable",
        },
    )
