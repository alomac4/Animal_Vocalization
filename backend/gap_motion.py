"""Bounded, forward/backward-checked visual motion for detector gaps.

Estimates are association hints, not observations, votes, or rendered dog boxes.
No learned model or external weights. All points belong to original image crops.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowState:
    box: tuple
    anchor_box: tuple
    points: object
    last_detection: int


class GapMotion:
    def __init__(self, max_gap_frames, max_side=640):
        if max_gap_frames < 1 or max_side < 64:
            raise ValueError("Invalid visual gap-motion bounds")
        self.max_gap_frames, self.max_side = max_gap_frames, max_side
        self.previous = self.current = None
        self.scale = 1.0
        self.states = {}
        self.frame_number = -1
        self.hints = {}
        self.estimates = self.rejections = self.scene_cuts = 0

    def advance(self, frame, frame_number):
        import cv2
        import numpy as np

        if frame_number != self.frame_number + 1:
            raise ValueError("Visual motion needs every frame in order")
        self.frame_number = frame_number
        self.hints = {}
        if frame is None:
            self.states.clear()
            self.previous = self.current = None
            return self.hints
        scale = min(1.0, self.max_side / max(frame.shape[:2]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1:
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        self.current = gray
        if self.previous is None or self.previous.shape != gray.shape or scale != self.scale:
            self.states.clear()
            self.previous, self.scale = gray, scale
            return self.hints
        # A large global visual jump invalidates local points; do not carry boxes
        # through a hard cut. This conservative gate can also reject fast motion.
        if np.mean(cv2.absdiff(self.previous, gray)) > 45:
            self.states.clear()
            self.scene_cuts += 1
        valid = []
        for key, state in list(self.states.items()):
            if frame_number - state.last_detection > self.max_gap_frames:
                del self.states[key]
                self.rejections += 1
            elif state.points is not None and len(state.points) >= 6:
                valid.append((key, state))
        if valid:
            original = np.concatenate([state.points for _, state in valid], axis=0)
            options = dict(winSize=(21, 21), maxLevel=3,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
            forward, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, original, None, **options)
            if forward is None:
                self.rejections += len(valid)
                self.states.clear()
            else:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None, **options)
                if backward is None:
                    self.rejections += len(valid)
                    self.states.clear()
                else:
                    usable = (status.ravel() == 1) & (back_status.ravel() == 1)
                    usable &= np.isfinite(forward).all(axis=(1, 2)) & np.isfinite(backward).all(axis=(1, 2))
                    usable &= np.linalg.norm(original[:, 0] - backward[:, 0], axis=1) <= 1.0
                    usable &= error.ravel() < 25
                    offset = 0
                    for key, state in valid:
                        count = len(state.points)
                        keep = usable[offset:offset + count]
                        old = original[offset:offset + count, 0][keep]
                        new = forward[offset:offset + count, 0][keep]
                        offset += count
                        predicted = self._transform_box(state, old, new, gray.shape)
                        if predicted is None:
                            self.states.pop(key, None)
                            self.rejections += 1
                            continue
                        box, points = predicted
                        state.box, state.points = box, points
                        # Original-frame coordinates, never the downscaled flow grid.
                        self.hints[key] = tuple(value / scale for value in box)
                        self.estimates += 1
        self.previous, self.scale = gray, scale
        return self.hints

    @staticmethod
    def _transform_box(state, old, new, shape):
        import cv2
        import numpy as np

        if len(old) < 6 or len(old) < len(state.points) * .5:
            return None
        x1, y1, x2, y2 = state.box
        width, height = x2 - x1, y2 - y1
        # Reject a tiny clump of background points driving an entire dog box.
        extent = np.ptp(old, axis=0)
        if extent[0] < width * .15 or extent[1] < height * .15:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                        ransacReprojThreshold=1.5, maxIters=100)
        if transform is None or not np.isfinite(transform).all() or inliers is None:
            return None
        keep = inliers.ravel().astype(bool)
        if keep.sum() < 6 or keep.mean() < .6:
            return None
        scale = float(np.hypot(transform[0, 0], transform[1, 0]))
        rotation = abs(float(np.arctan2(transform[1, 0], transform[0, 0])))
        if not .8 <= scale <= 1.25 or rotation > .25:
            return None
        center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, 1])
        moved = transform @ center
        if np.linalg.norm(moved - center[:2]) > .5 * np.hypot(width, height):
            return None
        w, h = width * scale, height * scale
        anchor_area = (state.anchor_box[2] - state.anchor_box[0]) * (state.anchor_box[3] - state.anchor_box[1])
        if not .5 <= w * h / max(1, anchor_area) <= 2:
            return None
        box = (max(0., moved[0] - w / 2), max(0., moved[1] - h / 2),
               min(float(shape[1]), moved[0] + w / 2), min(float(shape[0]), moved[1] + h / 2))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return None
        points = new[keep]
        inside = ((points[:, 0] >= box[0]) & (points[:, 0] < box[2]) &
                  (points[:, 1] >= box[1]) & (points[:, 1] < box[3]))
        if inside.sum() < 6:
            return None
        return tuple(float(value) for value in box), points[inside].reshape(-1, 1, 2).astype(np.float32)

    def anchor(self, track_id, box, frame_number):
        import cv2
        import numpy as np

        if self.current is None:
            return
        scaled = tuple(value * self.scale for value in box)
        x1, y1, x2, y2 = scaled
        inset_x, inset_y = .18 * (x2 - x1), .18 * (y2 - y1)
        mask = np.zeros(self.current.shape, dtype=np.uint8)
        a, b = max(0, round(x1 + inset_x)), max(0, round(y1 + inset_y))
        c, d = min(mask.shape[1], round(x2 - inset_x)), min(mask.shape[0], round(y2 - inset_y))
        mask[b:d, a:c] = 255
        points = cv2.goodFeaturesToTrack(self.current, maxCorners=40, qualityLevel=.02,
                                         minDistance=4, mask=mask, blockSize=5)
        if points is None or len(points) < 6:
            self.states.pop(track_id, None)
            return
        self.states[track_id] = FlowState(scaled, scaled, points, frame_number)


def flow_ocsort_class():
    """Installed tracker hook: adjust prediction only, never call a measurement update."""
    import numpy as np
    from ultralytics.trackers.oc_sort import OCSORT

    class FlowOCSORT(OCSORT):
        flow_hints = None
        prediction_updates = 0

        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            for track in pool:
                box = (self.flow_hints or {}).get(track.track_id)
                if box is None or track.mean is None:
                    continue
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue
                track.mean[:4] = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, width / height, height])
                self.prediction_updates += 1

    return FlowOCSORT