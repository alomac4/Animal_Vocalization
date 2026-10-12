import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_full_dataset import prepare_full, split_counts, stratified_split
from scripts.train_breeds import validate_dataset


XML = """<annotation><size><width>12</width><height>10</height></size>
<object><name>{name}</name><bodybndbox><xmin>1</xmin><ymin>1</ymin>
<xmax>{xmax}</xmax><ymax>9</ymax></bodybndbox></object></annotation>"""


class SplitTests(unittest.TestCase):
    def test_full_dataset_counts(self):
        self.assertEqual(split_counts(70428), [49300, 10564, 10564])

    def test_exact_disjoint_deterministic_and_stratified(self):
        entries = [f"breed{breed}/image{image}.jpg" for breed in range(130) for image in range(37)]
        partition = stratified_split(entries)
        self.assertEqual(partition, stratified_split(list(reversed(entries))))
        self.assertNotEqual(partition, stratified_split(entries, 43))
        self.assertEqual([len(partition[s]) for s in ("train", "val", "test")], split_counts(len(entries)))
        sets = [set(partition[s]) for s in ("train", "val", "test")]
        self.assertEqual(set.union(*sets), set(entries))
        for first in range(3):
            self.assertEqual(len({x.split('/')[0] for x in sets[first]}), 130)
            for second in range(first + 1, 3):
                self.assertFalse(sets[first] & sets[second])

    def test_invalid_inputs_rejected(self):
        for entries in ([], ["breed/a.jpg"] * 10, [f"breed/{i}.jpg" for i in range(6)]):
            with self.assertRaises(ValueError):
                stratified_split(entries)


class FullPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.annotations, self.images, self.splits = [self.root / x for x in ("annotations", "images", "splits")]
        self.splits.mkdir()
        self.output = self.root / "yolo"
        self.entries = []
        for breed, name in enumerate(("Shiba_Dog", "French_bulldog"), 1):
            folder = f"10-n{breed:06d}-{name}"
            (self.annotations / folder).mkdir(parents=True)
            (self.images / folder).mkdir(parents=True)
            for index in range(20):
                relative = f"{folder}/{index}.png"
                Image.new("RGB", (12, 10), (breed * 50, index * 10, 5)).save(self.images / relative)
                (self.annotations / (relative + ".xml")).write_text(XML.format(name=name, xmax=11))
                self.entries.append(relative)
        (self.splits / "train.lst").write_text("\n".join(self.entries[::2]) + "\n")
        (self.splits / "validation.lst").write_text("\n".join(self.entries[1::2]) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            return prepare_full(self.annotations, self.splits, self.images, self.output,
                                expected_classes=2, expected_images=40)

    def test_full_conversion_and_test_manifest(self):
        report = self.prepare()
        self.assertTrue(report["ready_for_training"])
        self.assertEqual([report["splits"][s]["images"] for s in ("train", "val", "test")], [28, 6, 6])
        self.assertEqual(report["valid_unique_images"], 40)
        self.assertEqual(report["decoded_pixel_overlap"], 0)
        self.assertEqual(validate_dataset(self.output / "data.yaml")["test"], "test.txt")
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_invalid_and_duplicate_exclusions_precede_split(self):
        # Identical pixels and annotations: retain only one before partitioning.
        duplicate = self.entries[1]
        (self.images / duplicate).write_bytes((self.images / self.entries[0]).read_bytes())
        invalid = self.entries[2]
        (self.annotations / (invalid + ".xml")).write_text("invalid XML")
        report = self.prepare()
        self.assertEqual(report["valid_unique_images"], 38)
        self.assertEqual(report["excluded_images"], 2)
        self.assertEqual(len(report["duplicate_groups"]), 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual([report["splits"][s]["images"] for s in ("train", "val", "test")], split_counts(38))

    def test_duplicate_conflicting_annotations_both_excluded(self):
        duplicate = self.entries[1]
        (self.images / duplicate).write_bytes((self.images / self.entries[0]).read_bytes())
        (self.annotations / (duplicate + ".xml")).write_text(XML.format(name="Shiba_Dog", xmax=10))
        report = self.prepare()
        self.assertEqual(report["valid_unique_images"], 38)
        self.assertIsNone(report["duplicate_groups"][0]["kept"])

    def test_missing_download_is_fatal(self):
        (self.images / self.entries[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse((self.output / "data.yaml").exists())

    def test_three_way_overlap_rejected_by_trainer(self):
        self.prepare()
        line = (self.output / "train.txt").read_text().splitlines()[0]
        (self.output / "test.txt").write_text(line + "\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_dataset(self.output / "data.yaml")


if __name__ == "__main__":
    unittest.main()