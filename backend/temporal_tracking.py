"""Adapter for installed Ultralytics sequential trackers; breed-independent IDs.

Only real detector observations are rendered; Kalman predictions are used solely
for association. Clip consensus/presence rules remain in the shared accumulator.
"""
from __future__ import annotations

if __package__:
    from .tracking import DogTracker, Track
else:
    from tracking import DogTracker, Track


class TemporalDogTracker(DogTracker):
    def __init__(self, fps, algorithm="botsort", min_hits=3, high_conf=.5, new_conf=.6,
                 low_conf=.25, buffer_seconds=15):
        import ultralytics
        from pathlib import Path
        from types import SimpleNamespace
        from ultralytics.utils import YAML
        from ultralytics.trackers import BOTSORT, OCSORT, FASTTracker

        factories = {"botsort": BOTSORT, "ocsort": OCSORT, "fasttrack": FASTTracker}
        self.gap_motion = None
        if algorithm == "ocsort_flow":
            if __package__:
                from .gap_motion import GapMotion, flow_ocsort_class
            else:
                from gap_motion import GapMotion, flow_ocsort_class
            factories[algorithm] = flow_ocsort_class()
            self.gap_motion = GapMotion(max(1, round(fps * .75)))
        if algorithm not in factories:
            raise ValueError("Unsupported sequential tracker")
        config_name = "ocsort" if algorithm == "ocsort_flow" else algorithm
        options = YAML.load(Path(ultralytics.__file__).parent / "cfg/trackers" / f"{config_name}.yaml")
        options.update(track_high_thresh=high_conf, track_low_thresh=low_conf, new_track_thresh=new_conf,
                       track_buffer=max(1, round(fps * buffer_seconds)), match_thresh=.85, fuse_score=False)
        if algorithm in ("ocsort", "ocsort_flow"):
            options.update(use_byte=True)
        if algorithm == "botsort":
            options.update(with_reid=False, gmc_method="sparseOptFlow")
        super().__init__(min_hits=min_hits, high_conf=high_conf, new_conf=new_conf, low_conf=low_conf,
                         confirmation_window=max(min_hits, round(fps)))
        self.backend = factories[algorithm](SimpleNamespace(**options))
        self.algorithm = algorithm
        self.max_simultaneous = 0

    def update(self, detections, frame_number, frame=None):
        import numpy as np
        from ultralytics.engine.results import Boxes

        if frame_number != self.last_frame + 1:
            raise ValueError("Sequential tracking requires every frame in order, including empty frames")
        self.last_frame = frame_number
        data = np.asarray([[*item["box"], item["confidence"], 0] for item in detections], dtype=np.float32).reshape(-1, 6)
        results = Boxes(data, frame.shape[:2] if frame is not None else (1, 1))
        if self.gap_motion is not None:
            self.backend.flow_hints = self.gap_motion.advance(frame, frame_number)
        tracked = self.backend.update(results, frame)
        accepted = []
        seen_ids, seen_indices = set(), set()
        for row in tracked:
            track_id, index = int(row[4]), int(row[-1])
            if not 0 <= index < len(detections) or track_id in seen_ids or index in seen_indices:
                raise RuntimeError("Sequential tracker returned invalid/duplicate ownership")
            seen_ids.add(track_id)
            seen_indices.add(index)
            item = dict(detections[index])
            # Use original, unsmoothed detector box/crop, not predicted tracker bounds.
            track = self.active.get(track_id)
            if track is None:
                track = Track(track_id, tuple(item["box"]), frame_number)
                self.active[track_id] = track
            track.observe(item, frame_number)
            track.high_hits += int(item["confidence"] >= self.high_conf)
            track.high_frames = [number for number in track.high_frames if frame_number - number < self.confirmation_window]
            if item["confidence"] >= self.high_conf:
                track.high_frames.append(frame_number)
            track.confirmed = track.confirmed or len(track.high_frames) >= self.min_hits
            item["track_id"] = track_id
            accepted.append(item)
            if self.gap_motion is not None and track.confirmed:
                self.gap_motion.anchor(track_id, item["box"], frame_number)
        self.next_id = len(self.active) + 1  # Candidate count, not highest backend ID.
        self.max_simultaneous = max(self.max_simultaneous, len(accepted))
        return accepted