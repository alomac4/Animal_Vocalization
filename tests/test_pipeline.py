from pathlib import Path
import tempfile
import unittest

import sys

# Support unittest discovery from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_dataset import build_classes, parse_annotation, prepare, read_split
from scripts.predict_breeds import results_to_payload, resolve_source
from scripts.project_paths import ROOT, DATASETS, MODELS, PRETRAINED, configure_ultralytics
from scripts.train_breeds import validate_dataset, make_smoke_dataset
from types import SimpleNamespace


XML = """<annotation><size><width>100</width><height>80</height></size>
<object><name>Shiba_Dog</name><headbndbox><xmin>1</xmin></headbndbox>
<bodybndbox><xmin>10</xmin><ymin>20</ymin><xmax>90</xmax><ymax>60</ymax>
</bodybndbox></object></annotation>"""


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.xml = self.root / "sample.xml"
        self.xml.write_text(XML, encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_body_box_normalization(self):
        result = parse_annotation(self.xml, {"Shiba_Dog": 0})
        self.assertEqual(result["labels"], [(0, 0.5, 0.5, 0.8, 0.5)])

    def test_multiple_objects(self):
        self.xml.write_text(XML.replace("</annotation>", XML[XML.index("<object>"):XML.index("</annotation>")] + "</annotation>"))
        self.assertEqual(len(parse_annotation(self.xml, {"Shiba_Dog": 0})["labels"]), 2)

    def test_unknown_breed_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown breed"):
            parse_annotation(self.xml, {})

    def test_head_box_not_used(self):
        self.xml.write_text(XML.replace("bodybndbox", "bndbox"))
        with self.assertRaisesRegex(ValueError, "Missing bodybndbox"):
            parse_annotation(self.xml, {"Shiba_Dog": 0})

    def test_invalid_box_rejected(self):
        self.xml.write_text(XML.replace("<xmax>90", "<xmax>5"))
        with self.assertRaisesRegex(ValueError, "Reversed"):
            parse_annotation(self.xml, {"Shiba_Dog": 0})

    def test_non_finite_rejected(self):
        self.xml.write_text(XML.replace("<xmin>10", "<xmin>nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parse_annotation(self.xml, {"Shiba_Dog": 0})

    def test_clipping_recorded(self):
        self.xml.write_text(XML.replace("<xmax>90", "<xmax>110"))
        result = parse_annotation(self.xml, {"Shiba_Dog": 0})
        self.assertEqual(result["clipped_boxes"], 1)
        self.assertAlmostEqual(result["labels"][0][3], 0.9)

    def test_official_split_format(self):
        path = self.root / "train.lst"
        path.write_text(".//1043-n000001-Shiba_Dog/n100001.jpeg\n")
        self.assertEqual(read_split(path), ["1043-n000001-Shiba_Dog/n100001.jpeg"])

    def test_unsafe_path_rejected(self):
        path = self.root / "train.lst"
        for value in ("../a.jpg", "C:/a.jpg", "/breed/a.jpg"):
            path.write_text(value)
            with self.assertRaises(ValueError):
                read_split(path)

    def test_duplicate_split_rejected(self):
        path = self.root / "train.lst"
        path.write_text("breed/a.jpg\nbreed/a.jpg\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_split(path)

    def test_class_order_uses_original_code(self):
        names, ids = build_classes(["200-n000008-Airedale/a.jpg", "1043-n000001-Shiba_Dog/b.jpg"])
        self.assertEqual(names, ["Shiba_Dog", "Airedale"])
        self.assertEqual(ids["Airedale"], 1)

    def make_sources(self, mismatch=False):
        from PIL import Image

        folder = "1043-n000001-Shiba_Dog"
        annotations, images, splits = (self.root / x for x in ("annotations", "images", "splits"))
        (annotations / folder).mkdir(parents=True)
        (images / folder).mkdir(parents=True)
        splits.mkdir()
        for split, filename in (("train", "a.jpg"), ("validation", "b.jpg")):
            (splits / (split + ".lst")).write_text(f".//{folder}/{filename}\n")
            (annotations / folder / (filename + ".xml")).write_text(XML)
            Image.new("RGB", (50, 40) if mismatch else (100, 80)).save(images / folder / filename)
        return annotations, images, splits

    def test_complete_conversion_and_no_overwrite(self):
        annotations, images, splits = self.make_sources()
        output = self.root / "converted"
        report = prepare(annotations, splits, output, images, expected_classes=1)
        self.assertTrue(report["ready_for_training"])
        self.assertTrue((output / "data.yaml").is_file())
        labels = list((output / "labels").rglob("*.txt"))
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].read_text(), "0 0.50000000 0.50000000 0.80000000 0.50000000\n")
        with self.assertRaises(FileExistsError):
            prepare(annotations, splits, output, images, expected_classes=1)

    def test_size_mismatch_leaves_no_training_yaml(self):
        annotations, images, splits = self.make_sources(mismatch=True)
        output = self.root / "converted"
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            prepare(annotations, splits, output, images, expected_classes=1)
        self.assertFalse((output / "data.yaml").exists())
        self.assertTrue((output / "preparation_report.json").exists())

    def test_overlap_rejected(self):
        annotations, images, splits = self.make_sources()
        (splits / "validation.lst").write_text((splits / "train.lst").read_text())
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(annotations, splits, self.root / "converted", images, expected_classes=1)

    def test_benchmark_partial_validation_coverage_is_explicit(self):
        from PIL import Image

        annotations, images, splits = self.make_sources()
        folder = "1044-n000002-French_bulldog"
        (annotations / folder).mkdir()
        (images / folder).mkdir()
        (annotations / folder / "c.jpg.xml").write_text(XML.replace("Shiba_Dog", "French_bulldog"))
        Image.new("RGB", (100, 80)).save(images / folder / "c.jpg")
        with (splits / "train.lst").open("a") as handle:
            handle.write(f".//{folder}/c.jpg\n")
        with self.assertRaisesRegex(ValueError, "Breeds missing"):
            prepare(annotations, splits, self.root / "strict", images, expected_classes=2)
        report = prepare(annotations, splits, self.root / "benchmark", images,
                         expected_classes=2, require_all_classes=False)
        self.assertTrue(report["ready_for_training"])
        self.assertFalse(report["require_all_classes_per_split"])
        self.assertEqual(report["missing_classes_per_split"]["val"], ["French_bulldog"])


class LayoutTests(unittest.TestCase):
    def test_workspace_paths(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(DATASETS, ROOT / "datasets")
        self.assertEqual(MODELS, ROOT / "models")
        self.assertEqual(PRETRAINED, ROOT / "models" / "pretrained")

    def test_local_library_paths(self):
        configure_ultralytics()
        from ultralytics import settings
        import ultralytics.utils as utils

        self.assertEqual(Path(settings["weights_dir"]), PRETRAINED)
        self.assertEqual(utils.WEIGHTS_DIR, PRETRAINED)
        self.assertEqual(Path(settings["datasets_dir"]), DATASETS)
        self.assertEqual(Path(settings.file_path).is_relative_to(ROOT / "logs"), True)

    def test_manifest_subset_after_conversion(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "train" / "dog.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            for split in ("train", "val"):
                (root / f"{split}.txt").write_text("./images/train/dog.jpg\n")
            path = root / "data.yaml"
            path.write_text(json.dumps({"path": root.as_posix(), "names": {0: "dog"},
                                        "train": "train.txt", "val": "val.txt"}))
            data = validate_dataset(path)
            subset = make_smoke_dataset(data, root / "subset")
            manifests = json.loads(subset.read_text())
            self.assertEqual(Path(manifests["train"]).read_text().strip(), image.resolve().as_posix())
            self.assertEqual(data["names"], {0: "dog"})


class InferenceTests(unittest.TestCase):
    def test_empty_detection_payload(self):
        result = SimpleNamespace(boxes=None, path="example.jpg", orig_shape=(80, 100))
        self.assertEqual(results_to_payload(result)["detections"], [])

    def test_breed_identity_separation(self):
        import torch

        box = SimpleNamespace(cls=torch.tensor([2]), conf=torch.tensor([0.8]),
                              xyxy=torch.tensor([[1, 2, 30, 40]]), id=None)
        result = SimpleNamespace(boxes=[box], path="example.jpg", orig_shape=(80, 100),
                                 names={2: "Labrador_retriever"})
        item = results_to_payload(result)["detections"][0]
        self.assertEqual(item["breed"]["label"], "Labrador_retriever")
        self.assertEqual(item["bbox_xyxy"], [1, 2, 30, 40])
        self.assertIsNone(item["identity"]["name"])
        self.assertEqual(item["identity"]["status"], "unassigned")

    def test_recursive_images_and_track_restriction(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Path(directory) / "breed"
            child.mkdir()
            image = child / "dog.jpg"
            image.touch()
            import glob

            self.assertIn(str(image.resolve()), glob.glob(resolve_source(directory), recursive=True))
            with self.assertRaisesRegex(ValueError, "one video"):
                resolve_source(directory, track=True)


if __name__ == "__main__":
    unittest.main()