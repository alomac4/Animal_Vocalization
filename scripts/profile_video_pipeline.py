"""Compare crop pipeline throughput on the same local benchmark frames (not accuracy)."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO
from backend.worker import infer_dog_regions


def main():
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    config = json.loads((ROOT / "backend/server.local.json").read_text(encoding="utf-8"))
    groups = sorted((ROOT / "Benchmarks/Dog_recog_model").rglob("frames"))
    paths = []
    for folder in groups:
        images = sorted(folder.glob("*.jpg"))
        if images:
            paths += [images[round(i * (len(images) - 1) / 11)] for i in range(12)]
    if not paths:
        raise RuntimeError("No local benchmark frames available")
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("Unreadable benchmark frame")
    variants = [
        ("960_serial_fp32", dict(imgsz=960, breed_batch=1, crop_upscale=1.0, half=False)),
        ("1536_upscale_serial_fp32", dict(imgsz=1536, breed_batch=1, crop_upscale=4.0, half=False)),
        ("1536_upscale_batch4_fp16", dict(imgsz=1536, breed_batch=4, crop_upscale=4.0, half=True)),
    ]
    reports = []
    for name, options in variants:
        dog = YOLO(config["weights"])
        breed = YOLO(config["breed_weights"])
        args = SimpleNamespace(device="cuda:0", breed_imgsz=960, breed_conf=.25, **options)
        classes = [key for key, value in dog.names.items() if value.lower() == "dog"]
        # Exclude initialization and warmup from this per-frame throughput measurement.
        for frame in frames[:2]:
            infer_dog_regions(frame, dog, breed, args, classes)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = {"dog_seconds": 0.0, "upscale_seconds": 0.0, "breed_seconds": 0.0}
        counts, labels = [], 0
        start = time.perf_counter()
        for frame in frames * 5:
            regions = infer_dog_regions(frame, dog, breed, args, classes, timings)
            counts.append(len(regions))
            labels += sum(region["breed"] is not None for region in regions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        reports.append(dict(name=name, settings=options, frames=len(frames) * 5, dog_boxes=sum(counts),
                            breed_labels=labels, counts_per_frame=counts, seconds=elapsed,
                            fps=len(frames) * 5 / elapsed, timings=timings,
                            dog_input_size=dog.predictor.imgsz, breed_input_size=breed.predictor.imgsz,
                            actual_fp16=bool(dog.predictor.model.fp16 and breed.predictor.model.fp16),
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2))
        print(json.dumps(reports[-1]), flush=True)
        del dog, breed
        torch.cuda.empty_cache()
    report = {"gpu": torch.cuda.get_device_name(0), "images": [str(path) for path in paths],
              "notes": "24 sampled stills repeated five times; excludes decode/encode/upload/loading. Same square input preprocessing for all variants. No accuracy claim or full-video estimate. Aggregate throughput, not independent accuracy samples.",
              "variants": reports}
    destination = ROOT / "backend/runtime/pipeline-profile.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()