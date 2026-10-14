"""Tiled CUDA Real-ESRGAN general-x4v3 crop restoration; no BasicSR dependency.

SRVGG architecture adapted from Xintao Wang's BSD-3-Clause Real-ESRGAN.
Upstream revision a4abfb2979a7bbff3f69f58f58ae324608821e27.
See third_party/Real-ESRGAN-LICENSE.txt and assets/upscaler-manifest.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "assets" / "realesr-general-x4v3.pth"


def output_size(width, height, target, max_scale):
    if min(width, height, target) < 2 or not 1 <= max_scale <= 4:
        raise ValueError("Use valid crop dimensions and a 1–4 enlargement limit")
    scale = min(max_scale, max(1.0, target / max(width, height)))
    return max(2, round(width * scale)), max(2, round(height * scale))


def tile_regions(width, height, tile=192, pad=36):
    """Core and contextual input rectangles; pad covers all 34 convolutions."""
    if min(width, height, tile) < 1 or pad < 34:
        raise ValueError("Tile size must be positive and context at least 34 pixels")
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            core = (x, y, min(x + tile, width), min(y + tile, height))
            context = (max(0, x - pad), max(0, y - pad),
                       min(width, core[2] + pad), min(height, core[3] + pad))
            yield core, context


def build_model():
    """Official general-x4v3 layout: 64 features, 32 body convs, PReLU, x4."""
    from torch import nn
    from torch.nn import functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)])
            for _ in range(32):
                self.body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
            self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(4)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + F.interpolate(x, scale_factor=4, mode="nearest")

    return SRVGGNetCompact()


class NeuralCropUpscaler:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cuda:0", half=True, tile=192, batch=4):
        import torch
        self.weights = Path(weights)
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("Real-ESRGAN requires CUDA; CPU fallback is disabled")
        if not self.weights.is_file():
            raise RuntimeError("Upscaler weights missing. Run backend/install_upscaler.py")
        manifest_path = self.weights.with_name("upscaler-manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != "realesr-general-x4v3" or hashlib.sha256(
                self.weights.read_bytes()).hexdigest() != manifest.get("sha256"):
            raise RuntimeError("Upscaler manifest/checksum mismatch; reinstall the official weights")
        if not 1 <= batch <= 8:
            raise ValueError("Upscaler tile batch must be 1–8")
        self.device, self.half, self.tile, self.batch = device, half, tile, batch
        # The official release is a state-dict checkpoint; never execute pickle globals.
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        self.model = build_model()
        self.model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model.eval().requires_grad_(False).to(device)
        if half:
            self.model.half()
        self.neural_crops = self.skipped_crops = self.tiles = 0
        self.model_calls = 0
        self.host_transfers = self.host_pixels = 0

    def enhance(self, crop, target=960, max_scale=4.0):
        return self.enhance_many([crop], target, max_scale)[0]

    def enhance_many(self, crops, target=960, max_scale=4.0):
        """Batch tiles and stitch on-device; transfer only one image per enhanced crop.

        Quantize each core exactly as before, but never copy discarded halo pixels
        to the host. Final area resizing remains OpenCV to preserve pixel output.
        """
        import cv2
        import numpy as np
        import torch

        if len(crops) > 8 or target > 2048:
            raise ValueError("Enhance at most eight crops with a target up to 2048 pixels")
        outputs, sizes = [], []
        # Group exact context shapes; no padding that would change edge predictions.
        groups = {}
        for crop_index, crop in enumerate(crops):
            if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
                raise ValueError("Upscaler expects a uint8 BGR dog crop")
            height, width = crop.shape[:2]
            size = output_size(width, height, target, max_scale)
            sizes.append(size)
            if size == (width, height):
                outputs.append(crop)
                self.skipped_crops += 1
                continue
            if max(width, height) > 2048:
                raise ValueError("Crop super-resolution input exceeds the bounded 2048-pixel limit")
            outputs.append(torch.empty((height * 4, width * 4, 3), dtype=torch.uint8,
                                       device=self.device))
            for core, context in tile_regions(width, height, self.tile):
                px1, py1, px2, py2 = context
                groups.setdefault((py2 - py1, px2 - px1), []).append((crop_index, core, context))
            self.neural_crops += 1
        with torch.inference_mode():
            for regions in groups.values():
                for offset in range(0, len(regions), self.batch):
                    chunk = regions[offset:offset + self.batch]
                    rgb = np.stack([crops[crop_index][py1:py2, px1:px2, ::-1]
                                    for crop_index, _, (px1, py1, px2, py2) in chunk])
                    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(
                        device=self.device, dtype=torch.float16 if self.half else torch.float32).div_(255)
                    output = self.model(tensor)
                    for index, (crop_index, (x1, y1, x2, y2), (px1, py1, _, _)) in enumerate(chunk):
                        # Convert/quantize only the useful tile core. RGB -> BGR
                        # is also on-device; no .cpu() synchronization in this loop.
                        core = output[index, :, (y1 - py1) * 4:(y2 - py1) * 4,
                                      (x1 - px1) * 4:(x2 - px1) * 4]
                        pixels = core.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                        outputs[crop_index][y1 * 4:y2 * 4, x1 * 4:x2 * 4].copy_(
                            pixels.flip(0).permute(1, 2, 0))
                        del core, pixels
                    self.tiles += len(chunk)
                    self.model_calls += 1
                    del tensor, output
        for index, (crop, restored, size) in enumerate(zip(crops, outputs, sizes)):
            if restored is crop:
                continue
            self.host_pixels += restored.shape[0] * restored.shape[1]
            restored = restored.cpu().numpy()
            self.host_transfers += 1
            outputs[index] = (cv2.resize(restored, size, interpolation=cv2.INTER_AREA)
                              if size != (restored.shape[1], restored.shape[0]) else restored)
        return outputs