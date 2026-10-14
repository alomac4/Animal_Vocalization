"""Confirmed high/low-score tracking, bounded motion and optional color matching.

No learned re-identification, camera-motion compensation, or identity guarantees.
Only actual detector boxes are returned; missing frames never get invented boxes.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


def box_iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def minimum_assignment(costs):
    """Rectangular Hungarian assignment; rows <= columns, finite costs only."""
    rows = len(costs)
    if not rows:
        return []
    columns = len(costs[0])
    if columns < rows or any(len(row) != columns for row in costs):
        raise ValueError("Assignment requires rectangular costs with rows <= columns")
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    owners, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        owners[0] = row
        column = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            current_row = owners[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate], previous[candidate] = reduced, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[owners[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if owners[column] == 0:
                break
        while column:
            preceding = previous[column]
            owners[column] = owners[preceding]
            column = preceding
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if owners[column]:
            assignment[owners[column] - 1] = column - 1
    return assignment


@dataclass
class Track:
    id: int
    box: tuple
    last_frame: int
    velocity: tuple = (0.0, 0.0, 0.0, 0.0)
    observations: int = 0
    votes: Counter = field(default_factory=Counter)
    scores: dict = field(default_factory=dict)
    appearance: tuple | None = None
    high_hits: int = 0
    high_frames: list = field(default_factory=list)
    confirmed: bool = False
    first_frame: int = -1
    first_box: tuple = ()
    first_appearance: tuple | None = None
    appearance_gallery: list = field(default_factory=list)

    def __post_init__(self):
        self.first_frame = self.last_frame
        self.first_box = self.box

    def predict(self, frame_number):
        # Never extrapolate box corners indefinitely: jitter can invert/shrink
        # boxes and a lost dog's old velocity becomes stale very quickly.
        gap = min(3, frame_number - self.last_frame)
        dx, dy = self.velocity[0] * gap, self.velocity[1] * gap
        return (self.box[0] + dx, self.box[1] + dy, self.box[2] + dx, self.box[3] + dy)

    def observe(self, detection, frame_number):
        box = tuple(detection["box"])
        gap = frame_number - self.last_frame
        if gap > 0:
            old_x, old_y = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            limit = max(box[2] - box[0], box[3] - box[1]) * .5
            dx, dy = max(-limit, min(limit, (x - old_x) / gap)), max(-limit, min(limit, (y - old_y) / gap))
            weight = 1.0 if self.observations == 1 else .5
            self.velocity = (weight * dx + (1 - weight) * self.velocity[0],
                             weight * dy + (1 - weight) * self.velocity[1], 0.0, 0.0)
        self.box, self.last_frame = box, frame_number
        self.observations += 1
        appearance = detection.get("appearance")
        if appearance:
            descriptor = tuple(appearance)
            if not self.appearance_gallery or min(appearance_distance(descriptor, item)
                                                  for item in self.appearance_gallery) > .08:
                self.appearance_gallery.append(descriptor)
                if len(self.appearance_gallery) > 12:
                    self.appearance_gallery.pop(1)  # Keep the first view plus recent pose views.
            if self.appearance is None:
                self.appearance = tuple(appearance)
                self.first_appearance = tuple(appearance)
            else:
                self.appearance = tuple(.8 * old + .2 * new for old, new in zip(self.appearance, appearance))
        breed = detection.get("breed")
        if not breed:
            return
        name = str(breed.get("name", "")).strip()
        score = breed.get("confidence")
        if name.casefold() in ("", "dog", "unknown", "breed unknown"):
            return
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            return
        self.votes[name] += 1
        self.scores[name] = self.scores.get(name, 0.0) + score

    def winner(self):
        if not self.votes:
            return None
        name = min(self.votes, key=lambda label: (-self.votes[label], -self.scores[label], label))
        return {"name": name, "votes": self.votes[name], "known_votes": sum(self.votes.values()),
                "confidence": self.scores[name] / self.votes[name]}

    def appearance_cost(self, descriptor):
        references = self.appearance_gallery or [self.appearance]
        distances = [appearance_distance(descriptor, item) for item in references]
        return min((value for value in distances if value is not None), default=None)


def appearance_distance(first, second):
    """Hellinger distance between normalized histograms, not neural ReID."""
    if not first or not second or len(first) != len(second):
        return None
    coefficient = sum(math.sqrt(max(0, a * b)) for a, b in zip(first, second))
    return math.sqrt(max(0, 1 - min(1, coefficient)))


class DogTracker:
    """High-first association, low-score recovery, and confirmed output tracks."""
    def __init__(self, max_gap=90, minimum_iou=0.15, new_conf=.6, high_conf=.5, min_hits=3,
                 low_conf=.25, confirmation_window=30, tentative_gap=15, reconnect_gap=0):
        if (max_gap < 1 or not 0 < minimum_iou <= 1 or not 0 < low_conf <= high_conf <= new_conf <= 1
                or min_hits < 1 or confirmation_window < min_hits or tentative_gap < 1 or reconnect_gap < 0):
            raise ValueError("Invalid tracking gap or IoU threshold")
        self.max_gap, self.minimum_iou = max_gap, minimum_iou
        self.new_conf, self.high_conf, self.min_hits = new_conf, high_conf, min_hits
        self.low_conf = low_conf
        self.confirmation_window, self.tentative_gap = confirmation_window, tentative_gap
        self.reconnect_gap = reconnect_gap
        self.recovered_tracks = 0
        self.presence_rejected_identities = self.presence_rejected_tracklets = 0
        self.active = {}
        self.completed_breeds = {}
        self.completed_tracks = {}
        self.next_id = 1
        self.last_frame = -1

    def association_cost(self, detection, track, frame_number):
        box, predicted = detection["box"], track.predict(frame_number)
        w, h = box[2] - box[0], box[3] - box[1]
        tw, th = track.box[2] - track.box[0], track.box[3] - track.box[1]
        area_ratio = w * h / max(1, tw * th)
        if not .25 <= area_ratio <= 4:
            return 2.0
        overlap = max(box_iou(box, predicted), .9 * box_iou(box, track.box))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        distances = [math.hypot(cx - (candidate[0] + candidate[2]) / 2,
                                cy - (candidate[1] + candidate[3]) / 2)
                     / max(1, math.hypot(tw, th)) for candidate in (predicted, track.box)]
        distance = min(distances)
        appearance = track.appearance_cost(detection.get("appearance"))
        # Appearance can rule out a nearby differently colored dog. Position is
        # always gated, so similarly colored objects cannot reconnect anywhere.
        if appearance is not None and appearance > .8 and overlap < .3:
            return 2.0
        gap = frame_number - track.last_frame
        fallback_limit = .75 if gap <= 3 else .45
        if gap > 3 and appearance is not None and appearance <= .3:
            fallback_limit = 1.0
        if overlap < self.minimum_iou and distance > fallback_limit:
            return 2.0
        geometry = .65 * (1 - overlap) + .35 * min(1, distance)
        return .75 * geometry + .25 * appearance if appearance is not None else geometry

    def match(self, detections, indices, tracks, frame_number):
        if not indices or not tracks:
            return {}
        costs = [[self.association_cost(detections[index], track, frame_number) for track in tracks]
                 + [1.0] * len(indices) for index in indices]
        return {indices[row]: tracks[column] for row, column in enumerate(minimum_assignment(costs))
                if column < len(tracks) and costs[row][column] < 1.0}

    def update(self, detections, frame_number):
        if frame_number <= self.last_frame:
            raise ValueError("Tracker requires strictly increasing frame numbers")
        self.last_frame = frame_number
        for track_id, track in list(self.active.items()):
            if frame_number - track.last_frame > (self.max_gap if track.confirmed else self.tentative_gap):
                if track.confirmed:
                    self.completed_breeds[track_id] = track.winner()
                    self.completed_tracks[track_id] = track
                del self.active[track_id]
        tracks = list(self.active.values())
        high = [i for i, detection in enumerate(detections) if detection["confidence"] >= self.high_conf]
        low = [i for i, detection in enumerate(detections)
               if self.low_conf <= detection["confidence"] < self.high_conf]
        matches = self.match(detections, high, [track for track in tracks if track.confirmed], frame_number)
        matches.update(self.match(detections, [i for i in high if i not in matches],
                                  [track for track in tracks if not track.confirmed], frame_number))
        used = {track.id for track in matches.values()}
        matches.update(self.match(detections, low, [track for track in tracks
                                                  if track.confirmed and track.id not in used], frame_number))
        # Before creating an ID, attempt one-to-one recovery from dormant confirmed
        # identities. Breed is never consulted. Ambiguous matches remain separate.
        remaining = [i for i in high if i not in matches and detections[i]["confidence"] >= self.new_conf]
        dormant = [track for track in self.completed_tracks.values()
                   if 0 < frame_number - track.last_frame <= self.reconnect_gap]
        recovery_costs = {}
        for index in remaining:
            for track in dormant:
                cost = self.reconnection_cost(track, detections[index]["box"],
                                              detections[index].get("appearance"), frame_number, self.reconnect_gap)
                if cost is not None:
                    recovery_costs[index, track.id] = cost
        for index, track_id in unambiguous_pairs(recovery_costs):
            track = self.completed_tracks.pop(track_id)
            self.completed_breeds.pop(track_id, None)
            self.active[track_id] = track
            matches[index] = track
            self.recovered_tracks += 1
        accepted = []
        # Matched boxes first so an unmatched duplicate cannot create a second
        # track simply because the detector changed its output order.
        order = sorted(range(len(detections)), key=lambda i: (i not in matches, -detections[i]["confidence"]))
        for index in order:
            detection = detections[index]
            track = matches.get(index)
            if track is None:
                if detection["confidence"] < self.new_conf:
                    continue
                # Suppress a nested duplicate instead of spawning a second ID.
                # Do not suppress merely touching/partially overlapping dogs.
                if any(box_iou(detection["box"], item["box"]) > .75 for item in accepted):
                    continue
                track = Track(self.next_id, tuple(detection["box"]), frame_number)
                self.active[track.id] = track
                self.next_id += 1
            track.observe(detection, frame_number)
            track.high_hits += int(detection["confidence"] >= self.high_conf)
            if not track.confirmed:
                track.high_frames = [frame for frame in track.high_frames
                                     if frame_number - frame < self.confirmation_window]
                if detection["confidence"] >= self.high_conf:
                    track.high_frames.append(frame_number)
                track.confirmed = len(track.high_frames) >= self.min_hits
            detection["track_id"] = track.id
            accepted.append(detection)
        # Preserve detector order for the temporary observation spool.
        accepted_ids = {id(item) for item in accepted}
        return [item for item in detections if id(item) in accepted_ids]

    def final_breeds(self):
        return {**self.completed_breeds,
                **{track_id: track.winner() for track_id, track in self.active.items() if track.confirmed}}

    @staticmethod
    def reconnection_cost(before, box, appearance, frame_number, max_gap):
        gap = frame_number - before.last_frame
        if not 0 < gap <= max_gap:
            return None
        distance = before.appearance_cost(appearance)
        if distance is None or distance > .3:
            return None
        a, b = before.box, box
        ratio = (b[2] - b[0]) * (b[3] - b[1]) / max(1, (a[2] - a[0]) * (a[3] - a[1]))
        diagonal = max(1, math.hypot(a[2] - a[0], a[3] - a[1]), math.hypot(b[2] - b[0], b[3] - b[1]))
        movement = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                              (a[1] + a[3] - b[1] - b[3]) / 2) / diagonal
        if not .25 <= ratio <= 4 or movement > 2:
            return None
        return .55 * distance + .25 * min(1, movement / 2) + .1 * abs(math.log(ratio)) / math.log(4) + .1 * gap / max_gap

    def final_consensus(self, max_link_gap, min_observations=0, min_relative_presence=0):
        """Pool votes across mutually-best unambiguous appearance/motion links.

        No breed-based matching. This is conservative tracklet linking, not ReID.
        Overlapping lifetimes never link, nor do missing appearance descriptors.
        """
        tracks = {**self.completed_tracks,
                  **{key: value for key, value in self.active.items() if value.confirmed}}
        if min_observations < 0 or not 0 <= min_relative_presence <= 1:
            raise ValueError("Invalid identity presence filter")
        costs = {}
        for before in tracks.values():
            for after in tracks.values():
                cost = self.reconnection_cost(before, after.first_box, after.first_appearance,
                                              after.first_frame, max_link_gap)
                if cost is not None:
                    costs[before.id, after.id] = cost
        parents = {key: key for key in tracks}

        def root(key):
            while parents[key] != key:
                key = parents[key]
            return key

        for before, after in sorted(unambiguous_pairs(costs), key=lambda pair: costs[pair]):
            first, second = root(before), root(after)
            # A transitive chain must never join dogs visible simultaneously.
            left = [track for key, track in tracks.items() if root(key) == first]
            right = [track for key, track in tracks.items() if root(key) == second]
            if first == second or any(max(a.first_frame, b.first_frame) <= min(a.last_frame, b.last_frame)
                                      for a in left for b in right):
                continue
            parents[max(first, second)] = min(first, second)
        groups = {}
        for key, track in tracks.items():
            groups.setdefault(root(key), []).append(track)
        winners = {}
        largest_presence = max((sum(track.observations for track in members) for members in groups.values()), default=0)
        required = max(min_observations, math.ceil(largest_presence * min_relative_presence))
        self.presence_rejected_identities = self.presence_rejected_tracklets = 0
        for identity, members in groups.items():
            observed_frames = sum(track.observations for track in members)
            if observed_frames < required:
                self.presence_rejected_identities += 1
                self.presence_rejected_tracklets += len(members)
                continue
            votes, scores = Counter(), {}
            for track in members:
                votes.update(track.votes)
                for name, score in track.scores.items():
                    scores[name] = scores.get(name, 0) + score
            winner = None
            if votes:
                name = min(votes, key=lambda label: (-votes[label], -scores[label], label))
                winner = {"name": name, "votes": votes[name], "known_votes": sum(votes.values()),
                          "confidence": scores[name] / votes[name]}
            for track in members:
                winners[track.id] = {"identity_id": identity, "breed": winner,
                                     "observed_frames": observed_frames}
        return winners


def unambiguous_pairs(costs, margin=.04):
    """Mutual best matches, with separation from each side's runner-up."""
    rows, columns = {}, {}
    for (first, second), cost in costs.items():
        rows.setdefault(first, []).append((cost, second))
        columns.setdefault(second, []).append((cost, first))
    pairs = []
    for first, choices in rows.items():
        choices.sort()
        cost, second = choices[0]
        reverse = sorted(columns[second])
        if reverse[0][1] != first:
            continue
        if len(choices) > 1 and choices[1][0] - cost < margin:
            continue
        if len(reverse) > 1 and reverse[1][0] - cost < margin:
            continue
        pairs.append((first, second))
    return pairs


def apply_track_breeds(detections, winners):
    """Replace every observation's label, including earlier unknown observations."""
    identities = {key: value["identity_id"] if value and "identity_id" in value else key
                  for key, value in winners.items()}
    display_ids = {key: index + 1 for index, key in enumerate(sorted(set(identities.values())))}
    return [{**detection, "source_track_id": detection["track_id"], "track_id": display_ids[identities[detection["track_id"]]],
             "breed": (winners[detection["track_id"]]["breed"]
                       if winners[detection["track_id"]] and "identity_id" in winners[detection["track_id"]]
                       else winners[detection["track_id"]]), "track_consensus": True}
            for detection in detections if detection.get("track_id") in winners]