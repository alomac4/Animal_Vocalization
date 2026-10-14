"""Regression tests for brief false positives, ID churn, and weak-score recovery."""
import unittest

from backend.tracking import DogTracker, apply_track_breeds


def dog(box=(0, 0, 40, 40), confidence=.9, appearance=None, breed=None):
    return dict(box=box, confidence=confidence, appearance=appearance,
                breed={"name": breed, "confidence": .8} if breed else None)


class ConfirmedTrackingTests(unittest.TestCase):
    def test_three_strong_hits_confirm_and_relabel_earliest_box(self):
        tracker = DogTracker()
        first = tracker.update([dog(breed="pug")], 0)
        self.assertEqual(tracker.final_breeds(), {})
        tracker.update([dog(breed="beagle")], 1)
        self.assertEqual(tracker.final_breeds(), {})
        tracker.update([dog(breed="beagle")], 2)
        rendered = apply_track_breeds(first, tracker.final_breeds())
        self.assertEqual(rendered[0]["track_id"], 1)
        self.assertEqual(rendered[0]["breed"]["name"], "beagle")

    def test_weak_and_one_frame_false_positives_do_not_render_or_inflate_display_id(self):
        tracker = DogTracker()
        weak = tracker.update([dog(confidence=.4)], 0)
        self.assertEqual(weak, [])
        brief = tracker.update([dog()], 1)
        tracker.update([], 17)
        real = tracker.update([dog((100, 100, 140, 140))], 18)
        tracker.update([dog((100, 100, 140, 140))], 19)
        tracker.update([dog((100, 100, 140, 140))], 20)
        self.assertEqual(apply_track_breeds(brief, tracker.final_breeds()), [])
        self.assertEqual(apply_track_breeds(real, tracker.final_breeds())[0]["track_id"], 1)

    def test_tentative_track_cannot_confirm_from_weak_or_sparse_hits(self):
        tracker = DogTracker(confirmation_window=5, tentative_gap=10)
        tracker.update([dog()], 0)
        self.assertEqual(tracker.update([dog(confidence=.3)], 1), [])
        tracker.update([dog()], 4)
        tracker.update([dog()], 8)
        self.assertEqual(tracker.final_breeds(), {})

    def test_confirmed_dog_recovers_using_low_scores_then_two_second_gap(self):
        tracker = DogTracker(max_gap=90)
        for frame in range(3):
            tracker.update([dog(appearance=(1., 0.))], frame)
        weak = tracker.update([dog(confidence=.3, appearance=(1., 0.))], 3)
        self.assertEqual(weak[0]["track_id"], 1)
        tracker.update([], 40)
        recovered = tracker.update([dog((28, 0, 68, 40), appearance=(1., 0.))], 63)
        self.assertEqual(recovered[0]["track_id"], 1)
        self.assertEqual(len(tracker.final_breeds()), 1)

    def test_color_separates_close_dogs_without_using_breed(self):
        tracker = DogTracker()
        for frame in range(3):
            tracker.update([dog((0, 0, 40, 40), appearance=(1., 0.), breed="pug"),
                            dog((40, 0, 80, 40), appearance=(0., 1.), breed="pug")], frame)
        result = tracker.update([dog((20, 0, 60, 40), appearance=(0., 1.), breed="beagle"),
                                 dog((20, 0, 60, 40), appearance=(1., 0.), breed="beagle")], 3)
        self.assertEqual([item["track_id"] for item in result], [2, 1])

    def test_duplicate_order_does_not_spawn_second_track(self):
        tracker = DogTracker()
        for frame in range(3):
            tracker.update([dog()], frame)
        result = tracker.update([dog((1, 1, 41, 41), confidence=.65), dog()], 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["track_id"], 1)
        self.assertEqual(tracker.next_id, 2)

    def test_far_similar_color_cannot_reconnect(self):
        tracker = DogTracker()
        for frame in range(3):
            tracker.update([dog(appearance=(1., 0.))], frame)
        distant = tracker.update([dog((1000, 1000, 1040, 1040), appearance=(1., 0.))], 10)
        self.assertEqual(distant[0]["track_id"], 2)

    def test_bounded_motion_never_inverts_box_after_gap(self):
        tracker = DogTracker()
        tracker.update([dog()], 0)
        tracker.update([dog((10, 0, 50, 40))], 1)
        predicted = tracker.active[1].predict(80)
        self.assertEqual(predicted[2] - predicted[0], 40)
        self.assertEqual(predicted[3] - predicted[1], 40)
        self.assertLessEqual(predicted[0], 70)

    def test_long_jittering_clip_keeps_one_id_and_filters_brief_noise(self):
        tracker = DogTracker(max_gap=90)
        for frame in range(4725):
            observations = []
            # Two-second missed-detection bursts do not erase the identity.
            if frame % 300 < 240:
                offset = frame % 7
                observations.append(dog((offset, 0, 40 + offset, 40), appearance=(1., 0.),
                                        confidence=.9 if frame % 10 < 8 else .3))
            if frame % 29 == 0:
                # Scattered false positives must not inflate rendered IDs.
                observations.append(dog((1000 + frame, 1000, 1040 + frame, 1040), confidence=.7))
            tracker.update(observations, frame)
        self.assertEqual(len(tracker.final_breeds()), 1)
        self.assertGreater(tracker.next_id, 100)

    def test_invalid_threshold_order_is_rejected(self):
        for options in (dict(new_conf=.3), dict(low_conf=.8), dict(confirmation_window=2),
                        dict(tentative_gap=0)):
            with self.assertRaises(ValueError):
                DogTracker(**options)


if __name__ == "__main__":
    unittest.main()