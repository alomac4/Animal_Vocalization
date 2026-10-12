import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.compare_trained_detectors import annotate, comparison, file_hash, pick_examples, scene_stats


class ComparisonTests(unittest.TestCase):
    def test_statistics_are_counts_not_accuracy(self):
        dog = [{"detections": [1, 2]}, {"detections": []}, {"detections": [1]}]
        breed = [{"detections": [1]}, {"detections": [1]}, {"detections": [1]}]
        stats = scene_stats(dog, breed)
        self.assertEqual(stats["dog"]["boxes"], 3)
        self.assertEqual(stats["dog"]["frames_with_boxes"], 2)
        self.assertEqual([stats[k] for k in ("dog_more_boxes_frames", "same_box_count_frames", "breed_more_boxes_frames")], [1, 1, 1])
        self.assertNotIn("accuracy", stats)

    def test_example_selection_fixed_spread_and_difference(self):
        dog = [{"detections": [1] * (5 if i == 3 else 0)} for i in range(10)]
        breed = [{"detections": []} for _ in dog]
        selected = pick_examples(dog, breed)
        self.assertEqual(len(set(selected)), 6)
        self.assertTrue({0, 3, 5, 9}.issubset(selected))

    def test_render_preserves_source_and_dimensions(self):
        image = Image.new("RGB", (320, 200), "white")
        detections = [{"bbox_xyxy": [1, 1, 180, 190], "label": "dog", "confidence": 0.8}]
        before = image.tobytes()
        self.assertNotEqual(annotate(image, detections, "green").tobytes(), before)
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(comparison(image, detections, [], "test.jpg").size, (640, 262))

    def test_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"abc")
            self.assertEqual(file_hash(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()