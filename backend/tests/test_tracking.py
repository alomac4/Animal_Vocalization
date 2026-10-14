"""No GPU dependencies: association, majority voting and retrospective labels."""
import itertools
import random
import unittest

from backend.tracking import DogTracker, apply_track_breeds, box_iou, minimum_assignment
from backend.worker import detection_label


def detection(box=(0, 0, 20, 20), name=None, score=0.8):
    return {"box": box, "confidence": .9,
            "breed": {"name": name, "confidence": score} if name is not None else None}


class TrackingTests(unittest.TestCase):
    def test_iou_and_rectangular_assignment(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0)
        self.assertEqual(minimum_assignment([[.2, .3, 1, 1], [.21, .9, 1, 1]]), [1, 0])
        self.assertEqual(minimum_assignment([]), [])

    def test_assignment_matches_exhaustive_global_minimum(self):
        rng = random.Random(7)
        for rows in range(1, 5):
            for _ in range(20):
                columns = rows + 2
                costs = [[rng.random() for _ in range(columns)] for _ in range(rows)]
                actual = minimum_assignment(costs)
                expected = min(sum(costs[i][j] for i, j in enumerate(option))
                               for option in itertools.permutations(range(columns), rows))
                self.assertEqual(len(set(actual)), rows)
                self.assertAlmostEqual(sum(costs[i][j] for i, j in enumerate(actual)), expected)

    def test_detection_order_changes_do_not_swap_separate_dogs(self):
        tracker = DogTracker(min_hits=1)
        first = tracker.update([detection(name="beagle"), detection((80, 0, 100, 20), "pug")], 0)
        second = tracker.update([detection((81, 0, 101, 20), "pug"), detection((1, 0, 21, 20), "beagle")], 1)
        self.assertEqual([item["track_id"] for item in first], [1, 2])
        self.assertEqual([item["track_id"] for item in second], [2, 1])
        self.assertEqual(tracker.final_breeds()[1]["name"], "beagle")
        self.assertEqual(tracker.final_breeds()[2]["name"], "pug")

    def test_motion_keeps_ids_when_dogs_cross_with_reordered_detections(self):
        tracker = DogTracker(min_hits=1)
        tracker.update([detection((0, 0, 20, 20)), detection((60, 0, 80, 20))], 0)
        tracker.update([detection((10, 0, 30, 20)), detection((50, 0, 70, 20))], 1)
        tracker.update([detection((20, 0, 40, 20)), detection((40, 0, 60, 20))], 2)
        tracker.update([detection((30, 0, 50, 20)), detection((30, 0, 50, 20))], 3)
        result = tracker.update([detection((20, 0, 40, 20)), detection((40, 0, 60, 20))], 4)
        self.assertEqual([item["track_id"] for item in result], [2, 1])

    def test_short_miss_keeps_id_but_expired_track_gets_new_id(self):
        tracker = DogTracker(max_gap=2, min_hits=1)
        tracker.update([detection(name="pug")], 0)
        self.assertEqual(tracker.update([], 1), [])
        self.assertEqual(tracker.update([detection()], 2)[0]["track_id"], 1)
        tracker.update([], 3)
        tracker.update([], 4)
        self.assertEqual(tracker.update([detection()], 5)[0]["track_id"], 2)
        self.assertEqual(tracker.final_breeds()[1]["name"], "pug")
        self.assertIsNone(tracker.final_breeds()[2])

    def test_majority_ignores_unknown_and_updates_earliest_observation(self):
        tracker = DogTracker(min_hits=1)
        early = tracker.update([detection()], 0)
        for frame, name in enumerate(["pug", "beagle", "beagle", "unknown", "breed unknown", None], 1):
            tracker.update([detection(name=name)], frame)
        winner = tracker.final_breeds()[1]
        self.assertEqual(winner["name"], "beagle")
        self.assertEqual((winner["votes"], winner["known_votes"]), (2, 3))
        rendered = apply_track_breeds(early, tracker.final_breeds())[0]
        self.assertEqual(detection_label(rendered), "Dog #1")
        self.assertEqual(rendered["breed"]["name"], "beagle")
        self.assertIsNone(early[0]["breed"])
        self.assertEqual(rendered["box"], early[0]["box"])

    def test_vote_count_beats_confidence_and_ties_are_deterministic(self):
        tracker = DogTracker(min_hits=1)
        for frame, (name, score) in enumerate([("pug", .99), ("beagle", .3), ("beagle", .3)]):
            tracker.update([detection(name=name, score=score)], frame)
        self.assertEqual(tracker.final_breeds()[1]["name"], "beagle")
        tracker.update([detection(name="pug", score=.99)], 3)
        self.assertEqual(tracker.final_breeds()[1]["name"], "pug")
        other = DogTracker(min_hits=1)
        other.update([detection(name="pug")], 0)
        other.update([detection(name="beagle")], 1)
        self.assertEqual(other.final_breeds()[1]["name"], "beagle")

    def test_invalid_votes_and_all_unknown_tracks_stay_unknown(self):
        tracker = DogTracker(min_hits=1)
        for frame, item in enumerate([detection(name="dog"), detection(name=""),
                                     detection(name="pug", score=float("nan")), detection()]):
            tracker.update([item], frame)
        winners = tracker.final_breeds()
        self.assertIsNone(winners[1])
        rendered = apply_track_breeds([item], winners)[0]
        self.assertEqual(detection_label(rendered), "Dog #1")
        self.assertIsNone(rendered["breed"])

    def test_unmatched_detection_never_steals_a_far_track(self):
        tracker = DogTracker(min_hits=1)
        tracker.update([detection()], 0)
        result = tracker.update([detection((100, 100, 120, 120)), detection()], 1)
        self.assertEqual([item["track_id"] for item in result], [2, 1])

    def test_frame_order_and_tracker_reset(self):
        tracker = DogTracker(min_hits=1)
        tracker.update([detection()], 0)
        with self.assertRaises(ValueError):
            tracker.update([], 0)
        self.assertEqual(DogTracker().update([detection()], 0)[0]["track_id"], 1)


if __name__ == "__main__":
    unittest.main()