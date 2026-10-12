from copy import deepcopy
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dog_augmentations import (
    SingleDogDataset, SingleDogTrainer, augment_single_dog, crop_window, edge_crop, motion_blur,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.instance import Instances


def sample(count=1, normalized=True):
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[40:160, 60:240] = 255
    boxes = np.array([[60, 40, 240, 160]] * count, dtype=np.float32).reshape(count, 4)
    if normalized:
        boxes /= np.array([300, 200, 300, 200], dtype=np.float32)
    return dict(img=image, cls=np.zeros((count, 1), dtype=np.float32),
                instances=Instances(boxes, bbox_format="xyxy", normalized=normalized),
                ori_shape=(200, 300), resized_shape=(200, 300), ratio_pad=(1., 1.))


class DogAugmentationTests(unittest.TestCase):
    def test_blur_preserves_shape_type_and_input(self):
        label = sample()
        image = label["img"]
        original = image.copy()
        for direction in ("horizontal", "vertical", "diagonal", "anti_diagonal"):
            for size in (3, 5, 7):
                blurred = motion_blur(image, size, direction)
                self.assertEqual(blurred.shape, image.shape)
                self.assertEqual(blurred.dtype, image.dtype)
                self.assertFalse(np.array_equal(blurred, image))
        np.testing.assert_array_equal(image, original)

    def test_blur_preserves_constant_image(self):
        image = np.full((30, 40, 3), 127, dtype=np.uint8)
        np.testing.assert_array_equal(motion_blur(image, 5, "horizontal"), image)

    def test_crop_all_edges_clips_and_normalizes_box(self):
        expected_windows = {
            "left": (96, 0, 300, 200), "right": (0, 0, 204, 200),
            "top": (0, 64, 300, 200), "bottom": (0, 0, 300, 136),
        }
        for normalized in (True, False):
            for side, (left, top, right, bottom) in expected_windows.items():
                with self.subTest(side=side, normalized=normalized):
                    label = sample(normalized=normalized)
                    before = deepcopy(label)
                    cropped = edge_crop(label, side, .20)
                    self.assertIsNot(cropped, label)
                    self.assertEqual(cropped["img"].shape[:2], (bottom-top, right-left))
                    self.assertTrue(cropped["instances"].normalized)
                    instances = deepcopy(cropped["instances"])
                    instances.convert_bbox("xyxy")
                    instances.denormalize(right-left, bottom-top)
                    expected = [max(60-left, 0), max(40-top, 0),
                                min(240-left, right-left), min(160-top, bottom-top)]
                    np.testing.assert_allclose(instances.bboxes[0], expected, atol=1e-4)
                    self.assertAlmostEqual(instances.bbox_areas[0] / (180*120), .80, places=5)
                    np.testing.assert_array_equal(label["img"], before["img"])
                    np.testing.assert_array_equal(label["instances"].bboxes, before["instances"].bboxes)
                    self.assertFalse(np.shares_memory(cropped["img"], label["img"]))

    def test_xywh_source_format(self):
        label = sample()
        label["instances"].convert_bbox("xywh")
        result = edge_crop(label, "left", .20)
        result["instances"].convert_bbox("xyxy")
        self.assertAlmostEqual(result["instances"].bboxes[0, 0], 0)

    def test_negative_and_multi_dog_unchanged(self):
        for count in (0, 2, 3):
            label = sample(count)
            self.assertIs(augment_single_dog(label, blur_probability=1, crop_probability=1), label)
            self.assertIs(edge_crop(label, "left", .20), label)

    def test_small_or_invalid_crops_skip_safely(self):
        self.assertIsNone(crop_window((200, 300), np.array([10, 10, 20, 20]), "left", .25))
        self.assertIsNone(crop_window((200, 300), np.array([230, 40, 290, 160]), "left", .20))
        for box in ([50, 50, 50, 60], [60, 60, 50, 50], [np.nan, 20, 100, 100]):
            self.assertIsNone(crop_window((200, 300), np.array(box), "left", .20))

    def test_zero_probabilities_leave_data_unchanged(self):
        label = sample()
        result = augment_single_dog(label, blur_probability=0, crop_probability=0)
        self.assertIs(result["img"], label["img"])
        self.assertIs(result["instances"], label["instances"])

    def test_seeded_combined_augmentation_reproducible(self):
        first = augment_single_dog(sample(), random.Random(42), 1, 1)
        second = augment_single_dog(sample(), random.Random(42), 1, 1)
        np.testing.assert_array_equal(first["img"], second["img"])
        np.testing.assert_array_equal(first["instances"].bboxes, second["instances"].bboxes)

    def test_invalid_options_rejected(self):
        for value in (-.1, 1.1):
            with self.assertRaises(ValueError):
                augment_single_dog(sample(), blur_probability=value)
            with self.assertRaises(ValueError):
                augment_single_dog(sample(), crop_probability=value)
        with self.assertRaises(ValueError):
            motion_blur(sample()["img"], 9, "horizontal")
        with self.assertRaises(ValueError):
            motion_blur(sample()["img"], 3, "unknown")
        with self.assertRaises(ValueError):
            crop_window((200, 300), np.array([60, 40, 240, 160]), "left", .5)

    def test_dataset_augmentation_only_for_training(self):
        dataset = object.__new__(SingleDogDataset)
        label = sample()
        with patch.object(YOLODataset, "get_image_and_label", return_value=label), \
                patch("scripts.dog_augmentations.augment_single_dog", return_value=label) as augment:
            dataset.augment = False
            self.assertIs(dataset.get_image_and_label(0), label)
            augment.assert_not_called()
            dataset.augment = True
            dataset.get_image_and_label(0)
            augment.assert_called_once_with(label)

    def test_validation_uses_unmodified_standard_dataset(self):
        trainer = object.__new__(SingleDogTrainer)
        with patch.object(DetectionTrainer, "build_dataset", return_value="standard") as build:
            self.assertEqual(trainer.build_dataset("val.txt", "val", 4), "standard")
            build.assert_called_once_with("val.txt", "val", 4)

    def test_random_crops_keep_boxes_in_bounds(self):
        rng = random.Random(8)
        for _ in range(100):
            result = augment_single_dog(sample(), rng, 1, 1)
            instances = deepcopy(result["instances"])
            instances.convert_bbox("xyxy")
            self.assertTrue(np.isfinite(instances.bboxes).all())
            self.assertTrue((instances.bboxes >= 0).all())
            self.assertTrue((instances.bboxes <= 1).all())
            self.assertTrue((instances.bbox_areas > 0).all())


if __name__ == "__main__":
    unittest.main()