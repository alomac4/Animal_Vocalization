import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_openimages_dogs import DOG, assign_splits, dog_rows, label_lines, negative_ids, prepare, select_records
from scripts.dog_training_pipeline import audit_dataset, training_config


class DogPreparationTests(unittest.TestCase):
    def test_box_filters_exclude_whole_image_and_retain_occlusion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "boxes.csv"
            fields = ["ImageID", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
                      "IsGroupOf", "IsDepiction", "IsInside", "IsOccluded", "IsTruncated"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                base = dict(zip(fields, ["good", DOG, 1, .1, .8, .2, .9, 0, 0, 0, 1, 1]))
                writer.writerow(base)
                writer.writerow(base)
                writer.writerow({**base, "ImageID": "group"})
                writer.writerow({**base, "ImageID": "group", "IsGroupOf": 1})
                writer.writerow({**base, "ImageID": "invalid", "XMax": 2})
                writer.writerow({**base, "ImageID": "depiction", "IsDepiction": 1})
                writer.writerow({**base, "ImageID": "inside", "IsInside": 1})
            boxes, excluded = dog_rows(path)
            self.assertEqual(boxes, {"good": [[.1, .8, .2, .9]]})
            self.assertEqual(excluded, {"group", "invalid", "depiction", "inside"})

    def test_only_verified_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            path.write_text(f"ImageID,LabelName,Confidence\na,{DOG},0\nb,{DOG},0\nb,{DOG},1\nc,{DOG},0\nd,/other,0\n")
            self.assertEqual(negative_ids(path, {"c"}), ["a"])

    def test_selection_prioritizes_multi_and_caps_total(self):
        positives = [{"id": str(i), "boxes": [[0, 1, 0, 1]] * (2 if i < 5 else 1)} for i in range(30)]
        negatives = [{"id": f"n{i}", "boxes": []} for i in range(10)]
        selected = select_records(positives, negatives, 10, 42)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(len(r["boxes"]) == 2 for r in selected), 5)
        self.assertEqual(sum(not r["boxes"] for r in selected), 1)
        self.assertEqual(selected, select_records(list(reversed(positives)), negatives, 10, 42))
        with self.assertRaises(ValueError):
            select_records(positives, negatives, 15001, 42)

    def test_split_deterministic_disjoint_and_stratified(self):
        records = [{"id": str(i), "boxes": [[0, 1, 0, 1]] * (i % 3)} for i in range(300)]
        splits = assign_splits(records, 42)
        self.assertEqual(splits, assign_splits(list(reversed(records)), 42))
        self.assertEqual([len(s) for s in splits.values()], [210, 45, 45])
        self.assertEqual(len({r["id"] for s in splits.values() for r in s}), 300)
        for subset in splits.values():
            self.assertEqual({min(len(r["boxes"]), 2) for r in subset[:8]}, {0, 1, 2})
        self.assertEqual(label_lines([[.2, .8, .1, .9]]), "0 0.50000000 0.50000000 0.60000000 0.80000000\n")

    def test_end_to_end_preparation_and_audit_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [{"id": f"{i:016x}", "source_split": "train", "boxes": [[.1, .9, .1, .9]] * (0 if i < 12 else 1)} for i in range(120)]
            (root / "selection.json").write_text(json.dumps({"maximum_images": 120, "seed": 42, "records": records}))

            def fake_download(url, destination, **kwargs):
                destination.parent.mkdir(parents=True, exist_ok=True)
                i = int(destination.stem, 16)
                Image.new("RGB", (16, 16), (i, i * 2, i // 2)).save(destination)

            with patch("scripts.prepare_openimages_dogs.download", side_effect=fake_download), patch("sys.stdout", new_callable=io.StringIO):
                report = prepare(root, maximum=120, workers=2)
            self.assertTrue(report["ready_for_training"])
            self.assertEqual(report["valid_unique_images"], 120)
            self.assertEqual(audit_dataset(root / "data.yaml")["valid_unique_images"], 120)
            self.assertEqual(sum(r["images"] for r in report["splits"].values()), 120)
            first = next((root / "labels").glob("*.txt"))
            first.write_text("1 0.5 0.5 0.5 0.5\n")
            with self.assertRaises(ValueError):
                audit_dataset(root / "data.yaml")

    def test_training_settings_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        previous = root / "weights" / "yolo12s_960_full_701515_20261007_014943" / "args.yaml"
        if not previous.is_file():
            self.skipTest("Previous local training run unavailable")
        config = training_config(previous)
        self.assertEqual(config["epochs"], 100)
        self.assertEqual(config["imgsz"], 960)
        self.assertEqual(config["batch"], 4)
        self.assertEqual(config["optimizer"], "auto")
        self.assertNotIn("model", config)
        self.assertNotIn("data", config)


if __name__ == "__main__":
    unittest.main()