"""Dog-only tracking diagnostic on a retained rendered result; not the original clip."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO

from backend.settings import Settings
from backend.tracking import DogTracker
from backend.worker import clip_dog_box, crop_appearance, dog_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    settings = Settings.load()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError('Cannot open retained result for diagnostic')
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = DogTracker(max_gap=round(fps * 3), reconnect_gap=round(fps * 15),
                         confirmation_window=max(3, round(fps)), tentative_gap=max(1, round(fps * .5)))
    model = YOLO(str(settings.weights))
    classes = [key for key, name in model.names.items() if name.lower() == 'dog']
    frames = boxes = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, imgsz=1536, rect=False, conf=.25, iou=.5,
                                   agnostic_nms=True, classes=classes, device=settings.device,
                                   quantize=16, verbose=False, save=False)[0]
            detections = []
            if result.boxes is not None:
                for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                    bounds = clip_dog_box(box, frame.shape[1], frame.shape[0])
                    if bounds is None:
                        continue
                    x1, y1, x2, y2 = bounds
                    detections.append(dict(box=bounds, confidence=score, breed=None,
                                           appearance=crop_appearance(frame[y1:y2, x1:x2])))
            tracker.update(detections, frames)
            frames += 1
            boxes += len(detections)
            if frames % 500 == 0:
                print(json.dumps(dict(frames=frames, candidates=tracker.next_id - 1,
                                      recoveries=tracker.recovered_tracks)), flush=True)
    finally:
        cap.release()
    from backend.tracking import appearance_distance
    tracks = {**tracker.completed_tracks, **{key: value for key, value in tracker.active.items() if value.confirmed}}
    opportunities = []
    for before in tracks.values():
        for after in tracks.values():
            gap = after.first_frame - before.last_frame
            if 0 < gap <= fps * 15:
                opportunities.append(dict(before=before.id, after=after.id, gap=gap,
                    appearance=before.appearance_cost(after.first_appearance),
                    cost=tracker.reconnection_cost(before, after.first_box, after.first_appearance,
                                                  after.first_frame, round(fps * 15))))
    unfiltered = tracker.final_consensus(round(fps * 15))
    minimum = min(frames, max(math.ceil(fps * 3), math.ceil(frames * .03)))
    winners = tracker.final_consensus(round(fps * 15), min_observations=minimum, min_relative_presence=.05)
    report = dict(frames=frames, fps=fps, detector_boxes=boxes,
                  candidate_tracks=tracker.next_id - 1, online_recoveries=tracker.recovered_tracks,
                  unfiltered_identities=len({entry['identity_id'] for entry in unfiltered.values()}),
                  final_identities=len({entry['identity_id'] for entry in winners.values()}),
                  presence_rejected_identities=tracker.presence_rejected_identities,
                  minimum_observed_frames=minimum, dogs=dog_statistics(winners),
                  association_opportunities=opportunities,
                  tracklets=[dict(id=t.id,first=t.first_frame,last=t.last_frame,observations=t.observations,
                                  first_box=t.first_box,last_box=t.box,gallery_size=len(t.appearance_gallery)) for t in tracks.values()],
                  notes='Dog-only re-inference on the retained annotated/downscaled MP4, not original input. Burned-in overlays and re-encoding may change detections. No breed/SR re-run. Counts do not prove five identities or identity accuracy. No input copied or video retained by this script.')
    output = ROOT / 'backend/runtime/identity-recovery-video.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()