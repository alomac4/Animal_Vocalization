"""Local CUDA restoration checks, same-crop timings and a visual contact sheet."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.settings import Settings
from backend.super_resolution import NeuralCropUpscaler
from backend.worker import clip_dog_box, upscale_dog_crop, select_breed


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    config = Settings.load()
    output = ROOT / "outputs" / "benchmarks" / "neural_crop_upscaling"
    output.mkdir(parents=True, exist_ok=True)
    upscaler = NeuralCropUpscaler(device=config.device, half=config.half)
    # Verify halo-tiled and full-image inference agree on the same random input.
    rng = np.random.default_rng(42)
    crop = rng.integers(0, 256, (65, 97, 3), dtype=np.uint8)
    upscaler.tile = 64
    tiled = upscaler.enhance(crop, 960, 4)
    tensor = torch.from_numpy(np.ascontiguousarray(crop[:, :, ::-1])).permute(2, 0, 1).unsqueeze(0).to(
        device=config.device, dtype=torch.float16 if config.half else torch.float32).div_(255)
    with torch.inference_mode():
        full = upscaler.model(tensor).float().clamp(0, 1).mul(255).round().to(torch.uint8)
    full = full[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    difference = np.abs(tiled.astype(float) - full.astype(float))
    assert difference.mean() < .25 and difference.max() <= 8, (difference.mean(), difference.max())
    upscaler.tile = config.upscaler_tile
    big = np.zeros((600, 1200, 3), dtype=np.uint8)
    assert upscaler.enhance(big) is big
    dog = YOLO(str(config.weights))
    breed = YOLO(str(config.breed_weights))
    rows, records = [], []
    folders = sorted((ROOT / "Benchmarks" / "Dog_recog_model").glob("**/frames"))
    for folder in folders:
        paths = sorted(folder.glob("*.jpg"))
        for path in paths[::max(1, len(paths) // 12)]:
            frame = cv2.imread(str(path))
            prediction = dog.predict(frame, imgsz=config.imgsz, rect=False, device=config.device,
                                     conf=.6, quantize=16, verbose=False)[0]
            if prediction.boxes is None or not len(prediction.boxes):
                continue
            bounds = clip_dog_box(prediction.boxes.xyxy[0].tolist(), frame.shape[1], frame.shape[0])
            if not bounds:
                continue
            x1, y1, x2, y2 = bounds
            original = frame[y1:y2, x1:x2].copy()
            # Real detected crop, not intentionally degraded imagery.
            if max(original.shape[:2]) >= 960:
                continue
            restored = upscaler.enhance(original)
            lanczos = upscale_dog_crop(original, 960, 4)
            measured = {}
            for name, enhance in [("lanczos", lambda: upscale_dog_crop(original, 960, 4)),
                                  ("neural", lambda: upscaler.enhance(original))]:
                enhance()  # Warm-up; model loading is excluded.
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    enhance()
                torch.cuda.synchronize()
                measured[name + "_ms"] = round((time.perf_counter() - start) * 1000 / 3, 3)
            labels = []
            for image in [lanczos, restored]:
                result = breed.predict(image, imgsz=960, rect=False, device=config.device,
                                       conf=config.breed_conf, quantize=16, verbose=False)[0]
                label = select_breed(result, image.shape[1], image.shape[0], config.breed_conf)
                labels.append(label)
            index = len(records) + 1
            cv2.imwrite(str(output / f"crop_{index}_original.png"), original)
            cv2.imwrite(str(output / f"crop_{index}_lanczos.png"), lanczos)
            cv2.imwrite(str(output / f"crop_{index}_neural.png"), restored)
            panels = []
            for heading, image, label in [("Original (display enlarged)", original, None),
                                           ("Lanczos", lanczos, labels[0]),
                                           ("Real-ESRGAN", restored, labels[1])]:
                panel = np.full((370, 340, 3), (236, 248, 255), dtype=np.uint8)
                cv2.putText(panel, heading, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 50), 1)
                scale = min(320 / image.shape[1], 300 / image.shape[0])
                preview = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                             max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
                panel[40:40 + preview.shape[0], 10:10 + preview.shape[1]] = preview
                if heading != "Original (display enlarged)":
                    text = label["name"] if label else "breed unknown"
                    cv2.putText(panel, text[:42], (10, 358), cv2.FONT_HERSHEY_SIMPLEX, .4, (30, 30, 50), 1)
                panels.append(panel)
            rows.append(np.concatenate(panels, axis=1))
            records.append({"source": str(path), "box": bounds, "input_shape": list(original.shape),
                            "output_shape": list(restored.shape), **measured,
                            "lanczos_breed": labels[0], "neural_breed": labels[1]})
            if len(records) >= 4:
                break
        if len(records) >= 4:
            break
    assert len(records) >= 2, "Insufficient real dog crops for validation"
    cv2.imwrite(str(output / "comparison.jpg"), np.concatenate(rows, axis=0))
    report = {"gpu": torch.cuda.get_device_name(0), "tile_mean_pixel_difference": float(difference.mean()),
              "tile_max_pixel_difference": float(difference.max()), "crops": records,
              "notes": "Same real crops. Warmup/loading excluded; 3 repeats each. No ground-truth breed labels or accuracy claim."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Visual comparison: {output / 'comparison.jpg'}")


if __name__ == "__main__":
    main()