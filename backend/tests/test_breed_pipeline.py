"""Crop-only pipeline tests: no CUDA, Torch or Ultralytics needed."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.worker import clip_dog_box, crop_upscale_size, detection_label, infer_dog_regions, infer_frame_batch, select_breed, upscale_dog_crop


class Values:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def result(boxes=(), scores=(), classes=(), names=None):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=Values(boxes), conf=Values(scores),
                                                 cls=Values(classes)), names=names or {0: "beagle", 1: "pug"})


class Frame:
    """Minimal frame stand-in verifying half-open slices and clean copies."""
    def __init__(self, height=100, width=120, region=None):
        self.shape = (height, width, 3)
        self.region = region
        self.copied = False

    def __getitem__(self, index):
        ys, xs = index
        return Frame(ys.stop - ys.start, xs.stop - xs.start,
                     (xs.start, ys.start, xs.stop, ys.stop))

    def copy(self):
        self.copied = True
        return self


class Model:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [next(self.results) for _ in (frame if isinstance(frame, list) else [frame])]


class BreedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(device="cuda:0", imgsz=1536, breed_conf=0.25, breed_imgsz=960,
                                    breed_batch=4, crop_upscale=4.0, half=True)
        # Model-call tests stay dependency-free; actual resize is tested separately.
        resizer = patch("backend.worker.upscale_dog_crop", side_effect=lambda crop, *_: crop)
        resizer.start()
        self.addCleanup(resizer.stop)
        appearance = patch("backend.worker.crop_appearance", return_value=(1.0, 0.0))
        appearance.start()
        self.addCleanup(appearance.stop)

    def test_crop_bounds_and_invalid_regions(self):
        self.assertEqual(clip_dog_box([-2.3, 0.2, 121, 99.1], 120, 100), (0, 0, 120, 100))
        for box in ([2, 3, 2, 8], [5, 8, 1, 2], [0, 0, math.nan, 8], [130, 2, 140, 8]):
            self.assertIsNone(clip_dog_box(box, 120, 100))

    def test_each_dog_gets_one_clean_crop_and_original_box_is_retained(self):
        frame = Frame()
        dog = Model([result([[10, 20, 60, 80], [70, 10, 120, 70]], [.9, .8], [0, 0])])
        breed = Model([result([[0, 0, 50, 60]], [.8], [0]), result()])
        detections = infer_dog_regions(frame, dog, breed, self.args, [0])
        self.assertEqual([d["box"] for d in detections], [(10, 20, 60, 80), (70, 10, 120, 70)])
        crops = [crop for call in breed.calls for crop in call[0]]
        self.assertEqual([crop.region for crop in crops], [d["box"] for d in detections])
        self.assertTrue(all(crop.copied and crop is not frame for crop in crops))
        self.assertEqual(len(breed.calls), 1)
        self.assertEqual(breed.calls[0][1]["batch"], 2)
        self.assertEqual(breed.calls[0][1]["imgsz"], 960)
        self.assertEqual(dog.calls[0][1]["imgsz"], 1536)
        self.assertFalse(dog.calls[0][1]["rect"])
        self.assertTrue(all(call[1]["quantize"] == 16 for call in dog.calls + breed.calls))
        self.assertTrue(all(call[1]["device"] == "cuda:0" for call in dog.calls + breed.calls))
        self.assertEqual(detections[0]["breed"]["name"], "beagle")
        self.assertEqual(detection_label(detections[0]), "Dog")
        self.assertIsNone(detections[1]["breed"])
        self.assertEqual(detection_label(detections[1]), "Dog")

    def test_no_dogs_means_no_breed_model_calls(self):
        breed = Model([])
        self.assertEqual(infer_dog_regions(Frame(), Model([result()]), breed, self.args, [0]), [])
        self.assertEqual(breed.calls, [])

    def test_invalid_dog_boxes_are_skipped(self):
        breed = Model([])
        dog = Model([result([[4, 4, 4, 8]], [.9], [0])])
        self.assertEqual(infer_dog_regions(Frame(), dog, breed, self.args, [0]), [])
        self.assertEqual(breed.calls, [])

    def test_breed_selection_rejects_small_offcenter_low_score_and_nonfinite(self):
        predictions = result([[0, 0, 10, 10], [0, 0, 30, 100], [0, 0, 100, 100],
                              [0, 0, 100, 100], [0, 0, 100, 100]], [.99, .98, .1, math.nan, .7],
                             [0, 0, 0, 0, 1])
        self.assertEqual(select_breed(predictions, 100, 100, .25), {"name": "pug", "confidence": .7})

    def test_only_one_highest_eligible_breed_is_selected(self):
        predictions = result([[0, 0, 100, 100]] * 2, [.5, .8], [0, 1])
        self.assertEqual(select_breed(predictions, 100, 100, .25)["name"], "pug")
        self.assertIsNone(select_breed(SimpleNamespace(boxes=None), 100, 100, .25))

    def test_crop_enlargement_is_bounded_and_preserves_aspect(self):
        self.assertEqual(crop_upscale_size(100, 50, 960, 4), (400, 200))
        self.assertEqual(crop_upscale_size(600, 300, 960, 4), (960, 480))
        self.assertEqual(crop_upscale_size(1200, 600, 960, 4), (1200, 600))

    def test_native_lanczos_resize_and_no_resize_for_large_crop(self):
        from unittest.mock import Mock
        cv2 = SimpleNamespace(INTER_LANCZOS4=4, resize=Mock(return_value=Frame(200, 400)))
        with patch.dict("sys.modules", cv2=cv2):
            crop = Frame(50, 100)
            enlarged = upscale_dog_crop(crop, 960, 4)
            self.assertEqual(enlarged.shape[:2], (200, 400))
            cv2.resize.assert_called_once_with(crop, (400, 200), interpolation=4)
            large = Frame(600, 1200)
            self.assertIs(upscale_dog_crop(large, 960, 4), large)

    def test_bounded_batches_keep_original_order(self):
        self.args.breed_batch = 2
        dog = Model([result([[10, 20, 60, 80]] * 5, [.9] * 5, [0] * 5)])
        breed = Model([result()] * 5)
        detections = infer_dog_regions(Frame(), dog, breed, self.args, [0])
        self.assertEqual(len(detections), 5)
        self.assertEqual([call[1]["batch"] for call in breed.calls], [2, 2, 1])

    def test_frame_batch_keeps_crop_ownership_and_batches_across_frames(self):
        frames = [Frame(), Frame(), Frame()]
        dog = Model([result([[10, 20, 60, 80]], [.9], [0]), result(),
                     result([[70, 10, 120, 70]], [.8], [0])])
        breed = Model([result([[0, 0, 50, 60]], [.8], [0]), result([[0, 0, 50, 60]], [.8], [1])])
        regions = infer_frame_batch(frames, dog, breed, self.args, [0])
        self.assertEqual([len(items) for items in regions], [1, 0, 1])
        self.assertEqual(regions[0][0]["breed"]["name"], "beagle")
        self.assertEqual(regions[2][0]["breed"]["name"], "pug")
        self.assertEqual(dog.calls[0][1]["batch"], 3)
        self.assertEqual(len(breed.calls), 1)
        self.assertEqual(breed.calls[0][1]["batch"], 2)

    def test_selection_coordinates_match_enlarged_crop(self):
        with patch("backend.worker.upscale_dog_crop", return_value=Frame(240, 200)):
            dog = Model([result([[10, 20, 60, 80]], [.9], [0])])
            breed = Model([result([[0, 0, 200, 240]], [.8], [0])])
            detections = infer_dog_regions(Frame(), dog, breed, self.args, [0])
            self.assertEqual(detections[0]["box"], (10, 20, 60, 80))
            self.assertEqual(detections[0]["breed"]["name"], "beagle")


if __name__ == "__main__":
    unittest.main()