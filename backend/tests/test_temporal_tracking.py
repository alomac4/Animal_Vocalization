"""Installed sequential trackers: breed independence, ownership and reset."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('ultralytics', 'numpy', 'cv2', 'lap'))


@unittest.skipUnless(AVAILABLE, 'Run in project .venv with Ultralytics/lap')
class TemporalTrackingTests(unittest.TestCase):
    def test_changing_breed_keeps_id_and_original_box(self):
        import numpy as np
        from backend.temporal_tracking import TemporalDogTracker
        for algorithm in ('botsort', 'ocsort', 'fasttrack'):
            tracker = TemporalDogTracker(30, algorithm)
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            ids = []
            for number in range(20):
                item = dict(box=(10+number, 10, 50+number, 60), confidence=.9,
                            appearance=(1., 0.), breed=dict(name=['pug', 'beagle'][number % 2], confidence=.8))
                result = tracker.update([item], number, frame)
                for observation in result:
                    self.assertEqual(observation['box'], item['box'])
                    ids.append(observation['track_id'])
            self.assertEqual(len(set(ids)), 1, algorithm)
            winners = tracker.final_consensus(0)
            self.assertEqual(len(winners), 1)
            self.assertIsNotNone(next(iter(winners.values()))['breed'])

    def test_empty_frames_and_video_local_reset(self):
        import numpy as np
        from backend.temporal_tracking import TemporalDogTracker
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        first_ids = []
        for _ in range(2):
            tracker = TemporalDogTracker(30, 'ocsort')
            for number in range(5):
                result = tracker.update([dict(box=(10,10,50,60),confidence=.9,breed=None)], number, frame)
                if result and number == 0:
                    first_ids.append(result[0]['track_id'])
            self.assertEqual(tracker.update([],5,frame), [])
            with self.assertRaises(ValueError):
                tracker.update([],7,frame)
        self.assertEqual(first_ids, [1,1])

    def test_different_dogs_never_share_id_in_same_frame(self):
        import numpy as np
        from backend.temporal_tracking import TemporalDogTracker
        tracker=TemporalDogTracker(30,'ocsort')
        frame=np.zeros((120,200,3),dtype=np.uint8)
        for number in range(20):
            items=[dict(box=(x,10,x+40,60),confidence=.9,breed=None) for x in (10,110)]
            result=tracker.update(items[::1 if number%2 else -1],number,frame)
            self.assertEqual(len({item['track_id'] for item in result}), len(result))
        self.assertEqual(len(tracker.final_consensus(0)),2)


if __name__ == '__main__':
    unittest.main()