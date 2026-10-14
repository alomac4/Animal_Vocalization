"""Same-input CUDA throughput/VRAM and cross-crop SR regression measurements."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.settings import Settings
from backend.super_resolution import NeuralCropUpscaler
from backend.worker import infer_frame_batch


class SerialUpscaler:
    def __init__(self, inner):
        self.inner = inner

    def enhance(self, *args):
        return self.inner.enhance(*args)


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    settings = Settings.load()
    sr = NeuralCropUpscaler(settings.upscaler_weights, settings.device, True, settings.upscaler_tile, 8)
    rng = np.random.default_rng(7)
    crops = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in
             [(65, 97, 3), (65, 97, 3), (71, 86, 3)]]
    sr.tile = 64
    serial = [sr.enhance(crop) for crop in crops]
    calls_before = sr.model_calls
    combined = sr.enhance_many(crops)
    batch_calls = sr.model_calls - calls_before
    differences = [int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(serial, combined)]
    assert max(differences) <= 8, differences
    bypass = np.zeros((600, 1200, 3), dtype=np.uint8)
    mixed = sr.enhance_many([crops[0], bypass])
    assert mixed[1] is bypass and mixed[0].shape == serial[0].shape
    sr.tile = settings.upscaler_tile
    paths = []
    for folder in sorted((ROOT / 'Benchmarks/Dog_recog_model').rglob('frames')):
        images = sorted(folder.glob('*.jpg'))
        if images:
            paths.extend(images[round(i * (len(images) - 1) / 11)] for i in range(12))
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError('Benchmark images missing/unreadable')
    reports = []
    for name, dog_batch, breed_batch, tile_batch, serial_sr in [
        ('previous_serial_frames_crop4_tile4', 1, 4, 4, True),
        ('frame1_crop4_tile8', 1, 4, 8, False),
        ('frame1_crop8_tile8', 1, 8, 8, False),
        ('frame2_crop8_tile8', 2, 8, 8, False),
        ('frame4_crop8_tile8', 4, 8, 8, False),
        ('frame8_crop8_tile8', 8, 8, 8, False),
    ]:
        dog, breed = YOLO(str(settings.weights)), YOLO(str(settings.breed_weights))
        classes = [key for key, value in dog.names.items() if value.lower() == 'dog']
        args = SimpleNamespace(device=settings.device, imgsz=1536, breed_imgsz=960, breed_conf=.25,
                               dog_conf=.25, breed_batch=breed_batch, half=True, crop_upscale=4)
        sr.batch = tile_batch
        upscaler = SerialUpscaler(sr) if serial_sr else sr
        for _ in range(2):
            infer_frame_batch(frames[:dog_batch], dog, breed, args, classes, upscaler=upscaler)
        torch.cuda.synchronize()
        trials = []
        for _ in range(5):
            torch.cuda.reset_peak_memory_stats()
            counts, labels = [], []
            timings = dict(dog_seconds=0.0, upscale_seconds=0.0, breed_seconds=0.0)
            calls_before = sr.model_calls
            start = time.perf_counter()
            for offset in range(0, len(frames), dog_batch):
                regions = infer_frame_batch(frames[offset:offset + dog_batch], dog, breed, args, classes,
                                            timings, upscaler)
                counts.extend(len(items) for items in regions)
                labels.extend([[region['breed']['name'] if region['breed'] else None for region in items]
                               for items in regions])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            trials.append(dict(seconds=seconds, fps=len(frames) / seconds, dog_boxes=sum(counts),
                               counts_per_frame=counts, labels_per_frame=labels, timings=timings,
                               sr_model_calls=sr.model_calls - calls_before,
                               peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                               peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2))
        report = dict(name=name, dog_batch=dog_batch, breed_batch=breed_batch, upscaler_batch=tile_batch,
                      frames_per_trial=len(frames), median_fps=float(np.median([t['fps'] for t in trials])),
                      trials=trials)
        reports.append(report)
        print(json.dumps({key: value for key, value in report.items() if key != 'trials'}), flush=True)
        del dog, breed
        torch.cuda.empty_cache()
    report = dict(gpu=torch.cuda.get_device_name(0), sr_batch_max_pixel_differences=differences,
                  sr_test_batch_model_calls=batch_calls, variants=reports, images=[str(path) for path in paths],
                  notes='24 sampled still images, five measured trials after warmup. No video decode/render/network/model loading. Different batches can change near-threshold FP16 outputs. Not an accuracy or exact-video runtime estimate.')
    destination = ROOT / 'backend/runtime/batched-pipeline-profile.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {destination}', flush=True)


if __name__ == '__main__':
    main()