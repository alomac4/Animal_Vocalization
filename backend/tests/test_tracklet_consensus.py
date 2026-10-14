import unittest

from backend.tracking import DogTracker, apply_track_breeds
from backend.worker import detection_label


def detection(name=None, box=(0, 0, 20, 20), appearance=(1.0, 0.0)):
    return {"box": box, "appearance": appearance, "confidence": .9,
            "breed": {"name": name, "confidence": .8} if name else None}


class TrackletConsensusTests(unittest.TestCase):
    def test_split_dog_pools_votes_and_relabels_earliest_unknown(self):
        tracker = DogTracker(max_gap=1, min_hits=1)
        early = tracker.update([detection()], 0)
        tracker.update([detection("pug")], 1)
        tracker.update([], 2)
        later = tracker.update([detection("beagle")], 3)
        tracker.update([detection("beagle")], 4)
        winners = tracker.final_consensus(6)
        self.assertEqual(set(winners), {1, 2})
        self.assertEqual(winners[1]["identity_id"], winners[2]["identity_id"])
        self.assertEqual(winners[1]["breed"]["name"], "beagle")
        labels = [detection_label(item) for item in apply_track_breeds(early + later, winners)]
        self.assertEqual(labels, ["Dog #1"] * 2)

    def test_overlapping_dogs_stay_separate_even_with_same_appearance_and_breed(self):
        tracker = DogTracker(min_hits=1)
        tracker.update([detection("pug"), detection("pug", (40, 0, 60, 20))], 0)
        winners = tracker.final_consensus(10)
        self.assertEqual(len({entry["identity_id"] for entry in winners.values()}), 2)

    def test_no_link_for_wrong_appearance_missing_appearance_or_disabled_linking(self):
        for appearance, gap in [((0.0, 1.0), 6), (None, 6), ((1.0, 0.0), 0)]:
            tracker = DogTracker(max_gap=1, min_hits=1)
            tracker.update([detection("pug")], 0)
            tracker.update([], 2)
            tracker.update([detection("beagle", appearance=appearance)], 3)
            winners = tracker.final_consensus(gap)
            self.assertEqual(len({entry["identity_id"] for entry in winners.values()}), 2)

    def test_ambiguous_successors_do_not_merge(self):
        tracker = DogTracker(max_gap=1, min_hits=1)
        tracker.update([detection("pug")], 0)
        tracker.update([], 2)
        tracker.update([detection("beagle", (-10, 0, 10, 20)),
                        detection("pug", (10, 0, 30, 20))], 3)
        winners = tracker.final_consensus(6)
        self.assertEqual(len({entry["identity_id"] for entry in winners.values()}), 3)

    def test_all_unknown_linked_dog_stays_unknown(self):
        tracker = DogTracker(max_gap=1, min_hits=1)
        first = tracker.update([detection("unknown")], 0)
        tracker.update([], 2)
        second = tracker.update([detection()], 3)
        winners = tracker.final_consensus(6)
        self.assertEqual([detection_label(item) for item in apply_track_breeds(first + second, winners)],
                          ["Dog #1"] * 2)