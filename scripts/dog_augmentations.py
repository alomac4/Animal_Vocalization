"""On-the-fly single-dog blur/truncation; no source images or labels are modified."""

from __future__ import annotations

from copy import deepcopy
import random

import cv2
import numpy as np

if __package__:
    from .project_paths import configure_ultralytics
else:
    from project_paths import configure_ultralytics

configure_ultralytics()
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model


POLICY = {
    "version": 1,
    "scope": "train only, source images containing exactly one dog box, before mosaic",
    "motion_blur_probability": 0.20,
    "motion_blur_kernel_sizes": [3, 5, 7],
    "motion_blur_directions": ["horizontal", "vertical", "diagonal", "anti_diagonal"],
    "edge_crop_probability": 0.20,
    "trim_fraction_of_box_dimension": [0.10, 0.25],
    "minimum_retained_box_area_fraction": 0.70,
    "minimum_retained_image_dimension_fraction": 0.50,
    "minimum_visible_box_dimension_pixels": 16,
    "original_files_modified": False,
    "existing_yolo_augmentations_preserved": True,
}


def motion_blur(image: np.ndarray, size: int, direction: str) -> np.ndarray:
    if size not in (3, 5, 7):
        raise ValueError("Use a mild 3, 5 or 7 pixel blur kernel")
    kernel = np.zeros((size, size), dtype=np.float32)
    if direction == "horizontal":
        kernel[size // 2, :] = 1
    elif direction == "vertical":
        kernel[:, size // 2] = 1
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1)
    elif direction == "anti_diagonal":
        kernel = np.fliplr(np.eye(size, dtype=np.float32))
    else:
        raise ValueError("Unknown blur direction")
    return cv2.filter2D(image, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT_101)


def crop_window(shape: tuple, box: np.ndarray, side: str, fraction: float) -> tuple | None:
    """Cut through one dog-box edge while preserving useful image/box dimensions."""
    if not .10 <= fraction <= .25:
        raise ValueError("Crop must trim 10..25% of a box dimension")
    height, width = shape[:2]
    x1, y1, x2, y2 = map(float, box)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    left, top, right, bottom = 0, 0, width, height
    if side == "left":
        left = round(x1 + fraction * bw)
    elif side == "right":
        right = round(x2 - fraction * bw)
    elif side == "top":
        top = round(y1 + fraction * bh)
    elif side == "bottom":
        bottom = round(y2 - fraction * bh)
    else:
        raise ValueError("Unknown crop side")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    visible_w = min(x2, right) - max(x1, left)
    visible_h = min(y2, bottom) - max(y1, top)
    if (right - left < width * .5 or bottom - top < height * .5
            or min(visible_w, visible_h) < 16 or bw <= 0 or bh <= 0
            or visible_w * visible_h / (bw * bh) < .70):
        return None
    return left, top, right, bottom


def edge_crop(label: dict, side: str, fraction: float) -> dict:
    """Clip and renormalize the single box along with the image; safely skip invalid crops."""
    if len(label["cls"]) != 1 or len(label["instances"]) != 1:
        return label
    image = label["img"]
    height, width = image.shape[:2]
    instances = deepcopy(label["instances"])
    instances.convert_bbox("xyxy")
    instances.denormalize(width, height)
    window = crop_window(image.shape, instances.bboxes[0], side, fraction)
    if window is None:
        return label
    left, top, right, bottom = window
    cropped = image[top:bottom, left:right].copy()
    new_height, new_width = cropped.shape[:2]
    instances.add_padding(-left, -top)
    instances.clip(new_width, new_height)
    instances.normalize(new_width, new_height)
    instances.convert_bbox("xywh")
    result = dict(label)
    result.update(img=cropped, instances=instances, ori_shape=(new_height, new_width),
                  resized_shape=(new_height, new_width), ratio_pad=(1.0, 1.0))
    result.pop("rect_shape", None)
    return result


def augment_single_dog(label: dict, rng=random, blur_probability: float = .20,
                       crop_probability: float = .20) -> dict:
    if not (0 <= blur_probability <= 1 and 0 <= crop_probability <= 1):
        raise ValueError("Augmentation probabilities must be in [0, 1]")
    if len(label["cls"]) != 1 or len(label["instances"]) != 1:
        return label
    result = dict(label)
    if rng.random() < crop_probability:
        sides = ["left", "right", "top", "bottom"]
        rng.shuffle(sides)
        fraction = rng.uniform(.10, .25)
        for side in sides:
            candidate = edge_crop(result, side, fraction)
            if candidate is not result:
                result = candidate
                break
    if rng.random() < blur_probability:
        result["img"] = motion_blur(result["img"], rng.choice([3, 5, 7]),
                                    rng.choice(["horizontal", "vertical", "diagonal", "anti_diagonal"]))
    return result


class SingleDogDataset(YOLODataset):
    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        return augment_single_dog(label) if self.augment else label


class SingleDogTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return SingleDogDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=True,
            hyp=self.args, rect=self.args.rect, cache=self.args.cache or None,
            single_cls=self.args.single_cls or False, stride=stride, pad=0.0,
            prefix=colorstr("train: "), task=self.args.task, classes=self.args.classes,
            data=self.data, fraction=self.args.fraction,
        )