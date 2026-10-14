"""Same original video/detections, current tracker vs installed sequential trackers."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO

from backend.settings import Settings
from backend.tracking import DogTracker, apply_track_breeds
from backend.temporal_tracking import TemporalDogTracker
from backend.worker import clip_dog_box, crop_appearance, dog_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reuse-detections', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['current','ocsort','botsort','fasttrack','ocsort_flow'],
                        default=['current','ocsort','botsort','fasttrack'])
    parser.add_argument('--output-name', default='original_video_tracking')
    args = parser.parse_args()
    settings = Settings.load()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if Path(args.output_name).name != args.output_name:
        raise ValueError('Output name must be a single folder name')
    output = ROOT / 'outputs/benchmarks' / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    source = cv2.VideoCapture(str(args.input))
    if not source.isOpened():
        raise RuntimeError('Cannot decode original test video')
    fps = source.get(cv2.CAP_PROP_FPS)
    total = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    cache = output / 'detections.jsonl'
    if args.reuse_detections:
        cache = ROOT / 'outputs/benchmarks/original_video_tracking/detections.jsonl'
    if not args.reuse_detections:
        model = YOLO(str(settings.weights))
        classes = [key for key, name in model.names.items() if name.lower() == 'dog']
        with cache.open('w', encoding='utf-8') as spool:
            number = 0
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                result = model.predict(frame, imgsz=1536, rect=False, conf=.25, iou=.5,
                                       classes=classes, agnostic_nms=True, device=settings.device,
                                       quantize=16, verbose=False, save=False)[0]
                items = []
                if result.boxes is not None:
                    for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                        bounds = clip_dog_box(box, frame.shape[1], frame.shape[0])
                        if bounds:
                            x1, y1, x2, y2 = bounds
                            items.append(dict(box=bounds, confidence=score, breed=None,
                                              appearance=crop_appearance(frame[y1:y2, x1:x2])))
                spool.write(json.dumps(dict(frame=number, detections=items)) + '\n')
                number += 1
                if number % 500 == 0:
                    print(f'Detection cache: {number}/{total}', flush=True)
        assert number == total
        del model
        torch.cuda.empty_cache()
    source.release()
    summaries = []
    # Run trackers separately: upstream global ID counters reset at construction.
    for name in args.variants:
        tracker = (DogTracker(max_gap=round(fps * 3), reconnect_gap=round(fps * 15),
                              confirmation_window=round(fps), tentative_gap=round(fps * .5))
                   if name == 'current' else TemporalDogTracker(fps, name))
        cap = cv2.VideoCapture(str(args.input))
        records = output / f'{name}-observations.jsonl'
        start = time.perf_counter()
        count = 0
        with cache.open(encoding='utf-8') as detections, records.open('w', encoding='utf-8') as spool:
            for line in detections:
                record = json.loads(line)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Input/cache length mismatch')
                items = (tracker.update(record['detections'], count) if name == 'current' else
                         tracker.update(record['detections'], count, frame))
                spool.write(json.dumps(dict(frame=count, detections=[dict(box=item['box'],track_id=item['track_id'])
                                                                    for item in items])) + '\n')
                count += 1
        cap.release()
        before = tracker.final_consensus(round(fps * 15))
        minimum = min(count, max(math.ceil(fps * 3), math.ceil(count * .03)))
        winners = tracker.final_consensus(round(fps * 15), min_observations=minimum, min_relative_presence=.05)
        summary = dict(tracker=name, frames=count, association_decode_seconds=round(time.perf_counter()-start,3),
                       candidates=tracker.next_id-1, confirmed_tracklets=len(before),
                       unfiltered_identities=len({v['identity_id'] for v in before.values()}),
                       final_identities=len({v['identity_id'] for v in winners.values()}),
                       rejected=tracker.presence_rejected_identities, dogs=dog_statistics(winners),
                       flow_estimates=tracker.gap_motion.estimates if getattr(tracker,'gap_motion',None) else 0,
                       flow_prediction_updates=getattr(getattr(tracker,'backend',None),'prediction_updates',0))
        summaries.append(summary)
        (output / f'{name}-winners.json').write_text(json.dumps(winners), encoding='utf-8')
        print(json.dumps(summary), flush=True)
        # Visual diagnostic, every original frame, dog-only IDs. No neural SR/breed.
        cap = cv2.VideoCapture(str(args.input))
        writer = cv2.VideoWriter(str(output/f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,720))
        if not writer.isOpened():
            raise RuntimeError('Cannot encode tracking preview')
        with records.open(encoding='utf-8') as spool:
            for line in spool:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Render/input mismatch')
                for item in apply_track_breeds(json.loads(line)['detections'], winners):
                    x1,y1,x2,y2=item['box']
                    color=((item['track_id']*83)%200+40,(item['track_id']*53)%200+40,(item['track_id']*31)%200+40)
                    cv2.rectangle(frame,(x1,y1),(x2-1,y2-1),color,2)
                    cv2.putText(frame,f"Dog #{item['track_id']}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                writer.write(frame)
        writer.release()
        cap.release()
    report=dict(source=str(args.input),fps=fps,frames=total,variants=summaries,
                notes='Original video, identical cached dog detections, every frame. Dog-only visual tracker experiment, no breed/SR, no accuracy score; fewer IDs can still conceal swaps/merges. MP4 previews have no audio. User reports five dogs, not imposed as a count constraint.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Results: {output}',flush=True)


if __name__ == '__main__':
    main()