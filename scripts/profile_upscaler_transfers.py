"""Compare the original host tile assembly with GPU assembly on identical data."""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from backend.settings import Settings
from backend.super_resolution import NeuralCropUpscaler, output_size, tile_regions
from backend.worker import process


class HostTileUpscaler(NeuralCropUpscaler):
    """Frozen pre-change implementation, used only as a measurement reference."""
    def enhance_many(self, crops, target=960, max_scale=4.0):
        outputs, sizes, groups = [], [], {}
        for crop_index, crop in enumerate(crops):
            height, width = crop.shape[:2]
            size = output_size(width, height, target, max_scale)
            sizes.append(size)
            if size == (width, height):
                outputs.append(crop)
                self.skipped_crops += 1
                continue
            outputs.append(np.empty((height * 4, width * 4, 3), dtype=np.uint8))
            for core, context in tile_regions(width, height, self.tile):
                px1, py1, px2, py2 = context
                groups.setdefault((py2 - py1, px2 - px1), []).append((crop_index, core, context))
            self.neural_crops += 1
        with torch.inference_mode():
            for regions in groups.values():
                for offset in range(0, len(regions), self.batch):
                    chunk = regions[offset:offset + self.batch]
                    rgb = np.stack([crops[i][py1:py2, px1:px2, ::-1]
                                    for i, _, (px1, py1, px2, py2) in chunk])
                    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(
                        device=self.device, dtype=torch.float16 if self.half else torch.float32).div_(255)
                    output = self.model(tensor)
                    pixels = output.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                    pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
                    self.host_transfers += 1
                    self.host_pixels += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                    for index, (i, (x1, y1, x2, y2), (px1, py1, _, _)) in enumerate(chunk):
                        outputs[i][y1 * 4:y2 * 4, x1 * 4:x2 * 4] = pixels[index,
                            (y1 - py1) * 4:(y2 - py1) * 4, (x1 - px1) * 4:(x2 - px1) * 4, ::-1]
                    self.tiles += len(chunk)
                    self.model_calls += 1
                    del tensor, output, pixels
        for index, (crop, restored, size) in enumerate(zip(crops, outputs, sizes)):
            if restored is not crop and size != (restored.shape[1], restored.shape[0]):
                outputs[index] = cv2.resize(restored, size, interpolation=cv2.INTER_AREA)
        return outputs


def benchmark_crops(config):
    gpu = NeuralCropUpscaler(config.upscaler_weights, config.device, config.half,
                            config.upscaler_tile, config.upscaler_batch)
    host = object.__new__(HostTileUpscaler)
    host.__dict__ = gpu.__dict__.copy()  # Share identical weights, not two model allocations.
    rng = np.random.default_rng(42)
    cases = [("small", [(65, 97)]), ("medium", [(215, 315)]),
             ("four_equal", [(217, 247)] * 4), ("four_varied", [(180, 260), (217, 247), (350, 520), (480, 640)]),
             ("large", [(600, 900)]), ("bypass", [(600, 1200)])]
    records = []
    for name, shapes in cases:
        crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in shapes]
        reference = host.enhance_many(crops)
        actual = gpu.enhance_many(crops)
        differences = [int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
                       for a, b in zip(reference, actual)]
        assert max(differences) == 0, (name, differences)
        times = {"host": [], "gpu": []}
        metrics = {}
        for trial in range(7):
            for variant, upscaler in [("host", host), ("gpu", gpu)][::1 if trial % 2 == 0 else -1]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                transfers, pixels = upscaler.host_transfers, upscaler.host_pixels
                start = time.perf_counter()
                upscaler.enhance_many(crops)
                torch.cuda.synchronize()
                times[variant].append((time.perf_counter() - start) * 1000)
                metrics[variant] = {"host_transfers": upscaler.host_transfers - transfers,
                                    "host_bytes": (upscaler.host_pixels - pixels) * 3,
                                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2)}
        record = {"case": name, "shapes": shapes, "max_pixel_difference": max(differences),
                  "median_ms": {key: round(statistics.median(value), 3) for key, value in times.items()},
                  "trials_ms": times, "metrics": metrics}
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def benchmark_video(config):
    ffmpeg = subprocess.check_output([str(ROOT / 'backend/.venv/Scripts/python.exe'), '-c',
        'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True).strip()
    folders = [next((ROOT / 'Benchmarks/Dog_recog_model').rglob(name)) for name in
               ('frames_easy_gsd_gate', 'frames_medium_backyard_pack')]
    reports = {"host": [], "gpu": []}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'profile.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 540))
        if not writer.isOpened():
            raise RuntimeError('Cannot create temporary comparison video')
        try:
            for folder in folders:
                paths = sorted(folder.rglob('f*.jpg'))
                for index in range(6):
                    image = cv2.resize(cv2.imread(str(paths[round(index * (len(paths) - 1) / 5)])), (960, 540))
                    for _ in range(10):
                        writer.write(image)
        finally:
            writer.release()
        for trial in range(2):
            for name, factory in [("host", HostTileUpscaler), ("gpu", NeuralCropUpscaler)][::1 if trial == 0 else -1]:
                args = SimpleNamespace(input=source, output=root / 'result.mp4', weights=config.weights,
                    breed_weights=config.breed_weights, device=config.device, imgsz=config.imgsz,
                    breed_imgsz=config.breed_imgsz, half=config.half, breed_conf=config.breed_conf,
                    crop_upscale=config.crop_upscale, breed_batch=config.breed_batch, dog_batch=config.dog_batch,
                    upscaler_weights=config.upscaler_weights, upscaler_tile=config.upscaler_tile,
                    upscaler_batch=config.upscaler_batch, dog_conf=config.dog_conf, track_high_conf=config.track_high_conf,
                    track_new_conf=config.track_new_conf, track_min_hits=config.track_min_hits,
                    track_buffer_seconds=config.track_buffer_seconds, track_link_seconds=config.track_link_seconds,
                    max_duration=config.max_duration_seconds, ffmpeg=ffmpeg)
                events = []
                with patch('backend.worker.emit', side_effect=lambda **event: events.append(event)), \
                        patch('backend.super_resolution.NeuralCropUpscaler', factory):
                    process(args)
                performance = next(event['performance'] for event in events if 'performance' in event)
                assert performance['frames'] == 120
                reports[name].append(performance)
                print(json.dumps({"variant": name, "trial": trial, "performance": performance}), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    config = Settings.load()
    report = {"gpu": torch.cuda.get_device_name(0), "crop_cases": benchmark_crops(config),
              "notes": "Seven warmed crop trials, interleaved order, identical weights/images. Exact pixel equality. Synthetic video is 120 frames from gate/backyard stills; not user's clip. No accuracy claim."}
    if args.video:
        report['video_trials'] = benchmark_video(config)
    output = ROOT / 'backend/runtime/upscaler-transfer-profile.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    main()