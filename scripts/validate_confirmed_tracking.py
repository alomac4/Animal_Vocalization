"""Local CUDA tracker diagnostics on benchmark sequences, not identity accuracy."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO
from backend.tracking import DogTracker, apply_track_breeds
from backend.worker import clip_dog_box, crop_appearance


def main():
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    config = json.loads((ROOT / "backend/server.local.json").read_text(encoding="utf-8"))
    model = YOLO(config["weights"])
    classes = [key for key, name in model.names.items() if name.lower() == "dog"]
    reports = []
    for folder in sorted((ROOT / "Benchmarks/Dog_recog_model").rglob("frames")):
        paths = sorted(folder.glob("*.jpg"))
        if not paths:
            continue
        # Sampling metadata does not supply source FPS; assume 30 frames/second
        # solely for this tracker diagnostic. Never claim a unique-dog metric.
        tracker = DogTracker(max_gap=90)
        frames, raw_boxes = [], 0
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Unreadable frame: {path}")
            result = model.predict(frame, imgsz=1536, rect=False, conf=.25, iou=.5, agnostic_nms=True,
                                   classes=classes, device="cuda:0", quantize=16, verbose=False, save=False)[0]
            detections = []
            if result.boxes is not None:
                for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                    bounds = clip_dog_box(box, frame.shape[1], frame.shape[0])
                    if bounds is None:
                        continue
                    x1, y1, x2, y2 = bounds
                    detections.append(dict(box=bounds, confidence=score, breed=None,
                                           appearance=crop_appearance(frame[y1:y2, x1:x2])))
            raw_boxes += len(detections)
            frames.append(tracker.update(detections, index))
        winners = tracker.final_breeds()
        report = dict(sequence=folder.parent.name, frames=len(paths), raw_detector_boxes=raw_boxes,
                      candidate_tracks=tracker.next_id - 1, confirmed_tracks=len(winners),
                      discarded_candidates=tracker.next_id - 1 - len(winners),
                      rendered_boxes=sum(len(apply_track_breeds(items, winners)) for items in frames))
        reports.append(report)
        print(json.dumps(report), flush=True)
    if not reports:
        raise RuntimeError("No benchmark frame sequences found")
    destination = ROOT / "backend/runtime/confirmed-tracking-benchmark.json"
    destination.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), sequences=reports,
        notes="Same 300 local sampled frames; 30fps assumed for association, source timing unavailable. Dog-only inference, no breed labels. Track counts are diagnostics, NOT ground-truth identity/detection accuracy. Original user clip unavailable."), indent=2), encoding="utf-8")
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()