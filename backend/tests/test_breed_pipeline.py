"""Crop-only pipeline tests: no CUDA, Torch or Ultralytics needed."""
import math
import unittest
from types import SimpleNamespace

from backend.worker import clip_dog_box, detection_label, infer_dog_regions, select_breed


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
        return [next(self.results)]


class BreedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(device="cuda:0", imgsz=960, breed_conf=0.25)

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
        self.assertEqual([call[0].region for call in breed.calls], [d["box"] for d in detections])
        self.assertTrue(all(call[0].copied and call[0] is not frame for call in breed.calls))
        self.assertTrue(all(call[1]["device"] == "cuda:0" for call in dog.calls + breed.calls))
        self.assertEqual(detections[0]["breed"]["name"], "beagle")
        self.assertEqual(detection_label(detections[0]), "beagle 80%")
        self.assertIsNone(detections[1]["breed"])
        self.assertIn("breed unknown", detection_label(detections[1]))

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


if __name__ == "__main__":
    unittest.main()