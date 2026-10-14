"""Exercise two-pass rendering without CUDA, OpenCV, downloaded weights or FFmpeg."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.worker import process, dog_statistics


class Frame:
    shape = (100, 120, 3)

    def tobytes(self):
        return b"frame"


class Capture:
    def __init__(self, length=4):
        self.length = length
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        return 5 if property_id == 1 else self.length

    def read(self):
        self.index += 1
        return (True, Frame()) if self.index <= self.length else (False, None)

    def release(self):
        self.released = True


def region(name=None):
    return {"box": (10, 10, 60, 80), "confidence": .9,
            "breed": {"name": name, "confidence": .8} if name else None}


class TrackRenderingTests(unittest.TestCase):
    def exercise(self, observations, render_length=None, track_buffer_seconds=3, presence_seconds=0, distance=False):
        length = len(observations)
        render_length = length if render_length is None else render_length
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(input=root / "input.video", output=root / "result.mp4",
                                   weights=root / "dog.pt", breed_weights=root / "breed.pt",
                                   device="cuda:0", imgsz=1536, breed_imgsz=960, half=True,
                                    crop_upscale=4.0, breed_batch=4, breed_conf=.25, dog_batch=4,
                                     max_duration=300, ffmpeg="ffmpeg", track_buffer_seconds=track_buffer_seconds, distance_enabled=distance,
                                     track_link_seconds=6, track_presence_seconds=presence_seconds, track_presence_ratio=0, track_clip_presence_ratio=0,
                                     track_reconnect_seconds=6)
            first, second = Capture(length), Capture(render_length)
            cv2 = SimpleNamespace(CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2, INTER_AREA=3,
                                  FONT_HERSHEY_SIMPLEX=4, LINE_AA=5, setNumThreads=Mock(),
                                  VideoCapture=Mock(side_effect=[first, second]),
                                  resize=Mock(side_effect=lambda frame, *_args, **_kwargs: frame),
                                  rectangle=Mock(), getTextSize=Mock(return_value=((90, 10), 0)), putText=Mock())
            cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                   get_device_name=lambda _: "Test GPU")
            torch = SimpleNamespace(set_num_threads=Mock(), cuda=cuda)
            model_factory = Mock(side_effect=[SimpleNamespace(task="detect", names={0: "dog"}),
                                             SimpleNamespace(task="detect", names={0: "beagle", 1: "pug"})])
            encoder = SimpleNamespace(stdin=io.BytesIO(), returncode=0, wait=Mock(),
                                      poll=lambda: 0, kill=Mock())
            events = []
            mux = Mock(side_effect=lambda *a, **kw: args.output.write_bytes(b"mp4"))
            with patch.dict("sys.modules", cv2=cv2, torch=torch,
                            ultralytics=SimpleNamespace(YOLO=model_factory)), \
                    patch("backend.worker.infer_frame_batch", side_effect=[
                        observations[offset:offset + 4] for offset in range(0, length, 4)]) as infer, \
                    patch("backend.worker.subprocess.Popen", return_value=encoder), \
                    patch("backend.worker.subprocess.run", mux), \
                    patch("backend.worker.emit", side_effect=lambda **event: events.append(event)), \
                    patch("backend.depth_distance.DepthProClient") as depth_factory:
                if render_length < length:
                    with self.assertRaisesRegex(ValueError, "decode ended"):
                        process(args)
                else:
                    process(args)
                depth_factory.assert_not_called()
            self.assertTrue(first.released and second.released)
            self.assertFalse((root / "track-observations.jsonl").exists())
            self.assertFalse((root / "silent.mp4").exists())
            self.assertEqual(infer.call_count, (length + 3) // 4)  # No inference while rendering.
            return events, cv2.putText.call_args_list, mux

    def test_all_frames_use_final_vote_including_first_unknown(self):
        events, draw_calls, mux = self.exercise([[region()], [region("pug")],
                                                [region("beagle")], [region("beagle")]])
        labels = [call.args[1] for call in draw_calls]
        self.assertEqual(labels, ["Dog #1"] * 8)  # Stroke + foreground.
        statistics = next(event["dog_statistics"] for event in events if "dog_statistics" in event)
        self.assertEqual(statistics, [{"dog_id": 1, "name": "Dog #1", "breed": "beagle", "votes": 2, "known_votes": 3, "observed_frames": 4}])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["tracked_dog_count"], 1)
        self.assertEqual(final["consensus_labeled_count"], 4)
        self.assertEqual(final["frames"], 4)
        mux.assert_called_once()
        statuses = [event["status"] for event in events if "status" in event]
        self.assertFalse("detecting" in statuses[statuses.index("rendering"):])

    def test_empty_video_detections_still_render_every_frame(self):
        events, draw_calls, mux = self.exercise([[], [], [], []])
        self.assertEqual(draw_calls, [])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["frames"], 4)
        self.assertEqual(final["tracked_dog_count"], 0)
        self.assertEqual(final["consensus_labeled_count"], 0)
        self.assertEqual(next(event["dog_statistics"] for event in events if "dog_statistics" in event), [])
        mux.assert_called_once()

    def test_linked_fragments_draw_one_majority_label_in_all_batches(self):
        def dog(name):
            return {**region(name), "appearance": (1.0, 0.0)}

        observations = [[dog(None)], [dog("pug")], [dog("pug")]] + [[] for _ in range(6)]
        observations += [[dog("beagle")] for _ in range(4)]
        events, draw_calls, _ = self.exercise(observations, track_buffer_seconds=1)
        self.assertEqual([call.args[1] for call in draw_calls], ["Dog #1"] * 14)
        statistics = next(event["dog_statistics"] for event in events if "dog_statistics" in event)
        self.assertEqual(statistics, [{"dog_id": 1, "name": "Dog #1", "breed": "beagle", "votes": 4, "known_votes": 6, "observed_frames": 7}])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["confirmed_tracklet_count"], 1)
        self.assertEqual(final["recovered_track_count"], 1)
        self.assertEqual(final["tracked_dog_count"], 1)
        self.assertEqual(final["consensus_labeled_count"], 7)

    def test_unconfirmed_boxes_are_not_drawn_including_earlier_frames(self):
        events, draw_calls, mux = self.exercise([[region("pug")], [], [], []])
        self.assertEqual(draw_calls, [])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["tracked_dog_count"], 0)
        self.assertEqual(final["discarded_candidate_track_count"], 1)
        mux.assert_called_once()

    def test_low_score_detection_without_track_renders_without_error(self):
        weak = region()
        weak["confidence"] = .3
        events, draw_calls, _ = self.exercise([[weak], [], [], []])
        self.assertEqual(draw_calls, [])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["candidate_track_count"], 0)

    def test_short_second_decode_fails_and_removes_spool(self):
        _, _, mux = self.exercise([[region()]] * 4, render_length=2)
        mux.assert_not_called()

    def test_statistics_ids_match_sorted_render_ids_and_deduplicate_linked_tracks(self):
        breed = {"name": "pug", "votes": 3, "known_votes": 4}
        winners = {9: {"identity_id": 9, "breed": None},
                   8: {"identity_id": 2, "breed": breed}, 2: {"identity_id": 2, "breed": breed}}
        self.assertEqual(dog_statistics(winners), [
            {"dog_id": 1, "name": "Dog #1", "breed": "pug", "votes": 3, "known_votes": 4, "observed_frames": 0},
            {"dog_id": 2, "name": "Dog #2", "breed": None, "votes": 0, "known_votes": 0, "observed_frames": 0}])

    def test_presence_rejection_removes_all_early_boxes_and_statistics(self):
        observations = [[region("beagle")]] * 3 + [[] for _ in range(17)]
        events, draw_calls, _ = self.exercise(observations, presence_seconds=3)
        self.assertEqual(draw_calls, [])
        self.assertEqual(next(event["dog_statistics"] for event in events if "dog_statistics" in event), [])
        final = next(event["performance"] for event in events if "performance" in event)
        self.assertEqual(final["presence_rejected_identities"], 1)
        self.assertEqual(final["presence_minimum_frames"], 15)

    def test_distance_overlay_and_notice_are_not_breed_or_tracking_inputs(self):
        observations=[[region('beagle')]]*4
        events,draw_calls,_=self.exercise(observations,distance=True)
        labels=[call.args[1] for call in draw_calls]
        dog_labels=[label for label in labels if label.startswith('Dog #')]
        self.assertEqual(len(dog_labels),8)
        self.assertTrue(all('est.' in label and ' m / ' in label and ' ft' in label for label in dog_labels))
        self.assertTrue(any('UNCALIBRATED' in label for label in labels))
        self.assertTrue(all('beagle' not in label for label in labels))
        final=next(event['performance'] for event in events if 'performance' in event)
        self.assertEqual(final['tracked_dog_count'],1)
        self.assertEqual(final['distance_estimated_boxes'],4)
        metadata=next(event['distance_estimation'] for event in events if 'distance_estimation' in event)
        self.assertTrue(metadata['approximate'])

    def test_depth_disabled_keeps_dog_only_labels_and_zero_depth_work(self):
        events,draw_calls,_=self.exercise([[region('beagle')]]*4)
        self.assertEqual([call.args[1] for call in draw_calls],['Dog #1']*8)
        final=next(event['performance'] for event in events if 'performance' in event)
        self.assertEqual(final['depth_frames'],0)
        self.assertEqual(final['depth_seconds'],0)
        self.assertEqual(final['depth_load_seconds'],0)
        self.assertEqual(final['distance_estimated_boxes'],0)
        self.assertIsNone(next(event['distance_estimation'] for event in events if 'distance_estimation' in event))


if __name__ == "__main__":
    unittest.main()