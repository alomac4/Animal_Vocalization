"""Stitching/transfer regression tests; run in the existing Torch environment."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(AVAILABLE, "Run with project .venv for Torch/OpenCV assembly tests")
class UpscalerAssemblyTests(unittest.TestCase):
    def make_upscaler(self):
        import torch
        from backend.super_resolution import NeuralCropUpscaler

        class NearestModel(torch.nn.Module):
            def forward(self, tensor):
                return torch.nn.functional.interpolate(tensor, scale_factor=4, mode="nearest")

        # CPU here isolates indexing/channel conversion from CUDA/model loading.
        upscaler = object.__new__(NeuralCropUpscaler)
        upscaler.device, upscaler.half, upscaler.tile, upscaler.batch = "cpu", False, 64, 4
        upscaler.model = NearestModel()
        upscaler.neural_crops = upscaler.skipped_crops = upscaler.tiles = 0
        upscaler.model_calls = upscaler.host_transfers = upscaler.host_pixels = 0
        return upscaler

    def test_tiled_assembly_keeps_bgr_edges_order_and_one_transfer_per_crop(self):
        import numpy as np
        upscaler = self.make_upscaler()
        rng = np.random.default_rng(1)
        crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                 for h, w in [(97, 145), (83, 67), (2, 2)]]
        outputs = upscaler.enhance_many(crops, 960, 4)
        for crop, output in zip(crops, outputs):
            np.testing.assert_array_equal(output, crop.repeat(4, axis=0).repeat(4, axis=1))
        self.assertEqual(upscaler.host_transfers, len(crops))
        self.assertEqual(upscaler.host_pixels, sum(h * w for h, w, _ in (out.shape for out in outputs)))
        self.assertGreater(upscaler.tiles, len(crops))

    def test_area_resize_and_bypass_are_unchanged(self):
        import cv2
        import numpy as np
        upscaler = self.make_upscaler()
        crop = np.arange(90 * 120 * 3, dtype=np.uint8).reshape(90, 120, 3)
        bypass = np.zeros((180, 240, 3), dtype=np.uint8)
        outputs = upscaler.enhance_many([crop, bypass], target=240)
        reference = cv2.resize(crop.repeat(4, axis=0).repeat(4, axis=1), (240, 180),
                               interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(outputs[0], reference)
        self.assertIs(outputs[1], bypass)
        self.assertEqual(upscaler.host_transfers, 1)
        self.assertEqual(upscaler.skipped_crops, 1)

    def test_empty_input_and_bounds(self):
        import numpy as np
        upscaler = self.make_upscaler()
        self.assertEqual(upscaler.enhance_many([]), [])
        self.assertEqual(upscaler.host_transfers, 0)
        for crops, target in [([np.zeros((10, 10, 3), dtype=np.uint8)] * 9, 960),
                              ([np.zeros((10, 10, 3), dtype=np.uint8)], 2049),
                              ([np.zeros((10, 10), dtype=np.uint8)], 960)]:
            with self.assertRaises(ValueError):
                upscaler.enhance_many(crops, target)


if __name__ == "__main__":
    unittest.main()