"""Dependency-free crop bounds, tile coverage, and pipeline routing checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.super_resolution import output_size, tile_regions
from backend.tests.test_breed_pipeline import Frame, Model, result
from backend.worker import infer_dog_regions


class SuperResolutionTests(unittest.TestCase):
    def test_aspect_ratio_bounded_enlargement_and_bypass(self):
        self.assertEqual(output_size(100, 50, 960, 4), (400, 200))
        self.assertEqual(output_size(600, 300, 960, 4), (960, 480))
        self.assertEqual(output_size(1200, 600, 960, 4), (1200, 600))
        for dims in [(0, 30, 960, 4), (30, 0, 960, 4), (30, 30, 960, 5)]:
            with self.assertRaises(ValueError):
                output_size(*dims)

    def test_tiles_cover_every_pixel_once_and_bound_context(self):
        for width, height in [(2, 2), (193, 197), (959, 511)]:
            area = 0
            for (x1, y1, x2, y2), (px1, py1, px2, py2) in tile_regions(width, height):
                self.assertTrue(0 <= px1 <= x1 < x2 <= px2 <= width)
                self.assertTrue(0 <= py1 <= y1 < y2 <= py2 <= height)
                self.assertLessEqual(px2 - px1, 264)
                self.assertLessEqual(py2 - py1, 264)
                area += (x2 - x1) * (y2 - y1)
            self.assertEqual(area, width * height)
        with self.assertRaises(ValueError):
            list(tile_regions(100, 100, pad=1))

    def test_neural_crops_only_enter_breed_model_original_appearance_is_retained(self):
        args = SimpleNamespace(device="cuda:0", imgsz=1536, breed_conf=.25,
                               breed_imgsz=960, breed_batch=4, crop_upscale=4, half=True)
        frame = Frame()
        restored = Frame(240, 200)
        upscaler = SimpleNamespace(enhance=Mock(return_value=restored))
        dog = Model([result([[10, 20, 60, 80]], [.9], [0])])
        breed = Model([result([[0, 0, 200, 240]], [.8], [0])])
        with patch("backend.worker.crop_appearance", return_value=(1.0, 0.0)) as appearance, \
                patch("backend.worker.upscale_dog_crop") as lanczos:
            regions = infer_dog_regions(frame, dog, breed, args, [0], upscaler=upscaler)
        self.assertIs(dog.calls[0][0], frame)
        self.assertIs(breed.calls[0][0][0], restored)
        original_crop = upscaler.enhance.call_args.args[0]
        self.assertEqual(original_crop.region, (10, 20, 60, 80))
        self.assertIs(appearance.call_args.args[0], original_crop)
        self.assertEqual(regions[0]["box"], original_crop.region)
        self.assertEqual(regions[0]["breed"]["name"], "beagle")
        lanczos.assert_not_called()

    def test_no_dogs_does_not_invoke_upscaler(self):
        args = SimpleNamespace(device="cuda:0", imgsz=1536, half=True)
        upscaler = SimpleNamespace(enhance=Mock())
        self.assertEqual(infer_dog_regions(Frame(), Model([result()]), Model([]), args, [0],
                                          upscaler=upscaler), [])
        upscaler.enhance.assert_not_called()

    def test_crop_batch_upscaling_preserves_order_and_original_tracking_appearance(self):
        args = SimpleNamespace(device="cuda:0", imgsz=1536, breed_imgsz=960,
                               breed_conf=.25, breed_batch=4, crop_upscale=4, half=True)
        restored = [Frame(200, 200), Frame(240, 240)]
        upscaler = SimpleNamespace(enhance_many=Mock(return_value=restored), enhance=Mock())
        dog = Model([result([[0, 0, 50, 50], [60, 60, 120, 120]], [.9, .8], [0, 0])])
        breed = Model([result([[0, 0, 200, 200]], [.8], [0]),
                       result([[0, 0, 240, 240]], [.9], [1])])
        with patch("backend.worker.crop_appearance", side_effect=[(1.0, 0.0), (0.0, 1.0)]):
            regions = infer_dog_regions(Frame(), dog, breed, args, [0], upscaler=upscaler)
        upscaler.enhance_many.assert_called_once()
        upscaler.enhance.assert_not_called()
        self.assertEqual([crop.region for crop in upscaler.enhance_many.call_args.args[0]],
                         [(0, 0, 50, 50), (60, 60, 120, 100)])
        self.assertEqual(breed.calls[0][0], restored)
        self.assertEqual([item["appearance"] for item in regions], [(1.0, 0.0), (0.0, 1.0)])
        self.assertEqual([item["breed"]["name"] for item in regions], ["beagle", "pug"])

    def test_failed_neural_enhancement_does_not_silently_fall_back(self):
        args = SimpleNamespace(device="cuda:0", imgsz=1536, breed_imgsz=960,
                               breed_batch=4, crop_upscale=4, half=True)
        upscaler = SimpleNamespace(enhance=Mock(side_effect=RuntimeError("CUDA out of memory")))
        with patch("backend.worker.crop_appearance", return_value=(1.0,)), \
                self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            infer_dog_regions(Frame(), Model([result([[0, 0, 50, 50]], [.9], [0])]),
                              Model([]), args, [0], upscaler=upscaler)


if __name__ == "__main__":
    unittest.main()