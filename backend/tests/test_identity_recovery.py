"""Changing breed must never alter IDs; false-positive duration is a heuristic."""
import copy
import unittest

from backend.tracking import DogTracker, apply_track_breeds, unambiguous_pairs
from backend.tests.test_confirmed_tracking import dog


class IdentityRecoveryTests(unittest.TestCase):
    def test_breed_predictions_cannot_change_identity_assignment(self):
        fixed, changing = DogTracker(max_gap=3, reconnect_gap=60), DogTracker(max_gap=3, reconnect_gap=60)
        for frame in range(300):
            items = [] if frame % 50 >= 40 else [dog((frame % 3, 0, frame % 3 + 40, 40), appearance=(1., 0.))]
            altered = copy.deepcopy(items)
            for item in altered:
                item["breed"] = {"name": ["pug", "beagle", "unknown"][frame % 3], "confidence": .8}
            first, second = fixed.update(items, frame), changing.update(altered, frame)
            self.assertEqual([item["track_id"] for item in first], [item["track_id"] for item in second])
        self.assertEqual(fixed.next_id, 2)
        self.assertEqual(changing.next_id, 2)
        self.assertGreater(changing.recovered_tracks, 0)

    def test_five_dogs_keep_ids_through_breed_flips_and_four_second_gaps(self):
        tracker = DogTracker(max_gap=90, reconnect_gap=450)
        expected = {}
        for frame in range(1000):
            items = []
            for index in range(5):
                if (frame + index * 30) % 250 >= 130:
                    continue
                appearance = tuple(float(i == index) for i in range(5))
                item = dog((index * 150, 0, index * 150 + 40, 40), appearance=appearance,
                           breed=["beagle", "pug", "unknown"][frame % 3])
                item["test_identity"] = index
                items.append(item)
            for item in tracker.update(list(reversed(items)) if frame % 2 else items, frame):
                index = item["test_identity"]
                expected.setdefault(index, item["track_id"])
                self.assertEqual(expected[index], item["track_id"])
        winners = tracker.final_consensus(450)
        self.assertEqual(len({entry["identity_id"] for entry in winners.values()}), 5)
        self.assertEqual(tracker.next_id, 6)

    def test_duration_filter_runs_after_linking_and_drops_100_frame_noise(self):
        tracker = DogTracker(min_hits=1)
        initial = tracker.update([dog((i * 100, 0, i * 100 + 40, 40)) for i in range(6)], 0)
        for index, track in tracker.active.items():
            track.observations = 3000 if index <= 5 else 100
        winners = tracker.final_consensus(450, min_observations=90, min_relative_presence=.05)
        self.assertEqual(len(winners), 5)
        self.assertEqual(len(apply_track_breeds(initial, winners)), 5)
        self.assertEqual(tracker.presence_rejected_identities, 1)

    def test_short_fragments_are_pooled_before_presence_filter(self):
        tracker = DogTracker(max_gap=1, min_hits=1)
        early = tracker.update([dog(appearance=(1., 0.), breed="pug")], 0)
        tracker.update([], 2)
        late = tracker.update([dog(appearance=(1., 0.), breed="beagle")], 3)
        tracker.completed_tracks[1].observations = 60
        tracker.active[2].observations = 60
        winners = tracker.final_consensus(10, min_observations=90)
        self.assertEqual(len(winners), 2)
        self.assertEqual(winners[1]["observed_frames"], 120)
        self.assertEqual([item["track_id"] for item in apply_track_breeds(early + late, winners)], [1, 1])

    def test_overlapping_lifetimes_cannot_merge_transitively(self):
        tracker = DogTracker(min_hits=1, max_gap=1)
        tracker.update([dog(appearance=(1., 0.))], 0)
        tracker.update([], 2)
        tracker.update([dog(appearance=(1., 0.)), dog((30, 0, 70, 40), appearance=(1., 0.))], 3)
        winners = tracker.final_consensus(20)
        self.assertNotEqual(winners[2]["identity_id"], winners[3]["identity_id"])

    def test_far_or_ambiguous_reappearance_is_not_forced_to_existing_dog(self):
        tracker = DogTracker(max_gap=1, min_hits=1, reconnect_gap=30)
        tracker.update([dog(appearance=(1., 0.))], 0)
        tracker.update([], 2)
        new = tracker.update([dog((1000, 0, 1040, 40), appearance=(1., 0.))], 3)
        self.assertEqual(new[0]["track_id"], 2)
        self.assertEqual(unambiguous_pairs({(1, 2): .1, (1, 3): .11}), [])
        self.assertEqual(unambiguous_pairs({(1, 2): .1, (1, 3): .3}), [(1, 2)])

    def test_invalid_presence_parameters_rejected(self):
        tracker = DogTracker()
        for options in ({"min_observations": -1}, {"min_relative_presence": 2}):
            with self.assertRaises(ValueError):
                tracker.final_consensus(10, **options)

    def test_different_genuine_durations_are_not_equalized(self):
        tracker = DogTracker(min_hits=1)
        tracker.update([dog((0, 0, 40, 40)), dog((100, 0, 140, 40))], 0)
        tracker.active[1].observations = 3000
        tracker.active[2].observations = 300
        winners = tracker.final_consensus(10, min_observations=142, min_relative_presence=.05)
        self.assertEqual(len(winners), 2)
        self.assertEqual(winners[2]["observed_frames"], 300)


if __name__ == "__main__":
    unittest.main()