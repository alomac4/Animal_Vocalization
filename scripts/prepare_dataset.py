"""Convert original Tsinghua Dogs body annotations to breed-aware YOLO labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET

def read_split(path: Path) -> list[str]:
    entries = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        raw = line.strip().replace("\\", "/")
        if not raw:
            continue
        while raw.startswith("./"):
            raw = raw[2:].lstrip("/")
        rel = PurePosixPath(raw)
        if rel.is_absolute() or len(rel.parts) != 2 or ".." in rel.parts:
            raise ValueError(f"Unsafe or unexpected split entry: {line!r}")
        if any(":" in part for part in rel.parts):
            raise ValueError(f"Unsafe split entry: {line!r}")
        entries.append(rel.as_posix())
    if not entries or len(entries) != len(set(entries)):
        raise ValueError(f"Empty split or duplicate entries in {path}")
    return entries


def breed_from_folder(folder: str) -> tuple[int, str]:
    parts = folder.split("-", 2)
    if len(parts) != 3 or not parts[1].startswith("n") or not parts[2]:
        raise ValueError(f"Unexpected breed folder: {folder}")
    return int(parts[1][1:]), parts[2]


def build_classes(entries: list[str]) -> tuple[list[str], dict[str, int]]:
    folders = sorted({PurePosixPath(x).parts[0] for x in entries},
                     key=lambda folder: breed_from_folder(folder)[0])
    names = [breed_from_folder(folder)[1] for folder in folders]
    codes = [breed_from_folder(folder)[0] for folder in folders]
    if len(names) != len(set(names)) or len(codes) != len(set(codes)):
        raise ValueError("Duplicate breed names or breed codes")
    return names, {name: i for i, name in enumerate(names)}


def annotation_path(root: Path, relative_image: str) -> Path:
    # Official files include the image extension: n100001.jpeg.xml.
    return root / (relative_image + ".xml")


def parse_annotation(path: Path, class_ids: dict[str, int]) -> dict:
    node = ET.parse(path).getroot()
    width = int(node.findtext("size/width", "0"))
    height = int(node.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotation dimensions: {width} x {height}")
    objects = node.findall("object")
    if not objects:
        raise ValueError("No annotated dogs; do not treat this as a negative image")
    labels = []
    clipped = 0
    for obj in objects:
        name = (obj.findtext("name") or "").strip()
        if name not in class_ids:
            raise ValueError(f"Unknown breed: {name!r}")
        box = obj.find("bodybndbox")
        if box is None:
            raise ValueError("Missing bodybndbox; head boxes are not a fallback")
        values = [float(box.findtext(key, "nan"))
                  for key in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite box coordinate")
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Reversed or empty body box: {values}")
        # Use image-boundary coordinates, allowing a boundary value of W or H.
        bounded = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                   max(0.0, min(width, x2)), max(0.0, min(height, y2))]
        clipped += int(bounded != values)
        x1, y1, x2, y2 = bounded
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Body box lies entirely outside the image")
        labels.append((class_ids[name], (x1 + x2) / (2 * width),
                       (y1 + y2) / (2 * height), (x2 - x1) / width,
                       (y2 - y1) / height))
    return {"width": width, "height": height, "labels": labels,
            "clipped_boxes": clipped}


def load_sources(annotations: Path, splits: Path) -> tuple[dict, list, dict]:
    split_entries = {"train": read_split(splits / "train.lst"),
                     "val": read_split(splits / "validation.lst")}
    overlap = set(split_entries["train"]) & set(split_entries["val"])
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} images")
    names, class_ids = build_classes(split_entries["train"] + split_entries["val"])
    # Different image extensions can otherwise collide in YOLO's .txt labels.
    keys = [str(PurePosixPath(x).with_suffix(".txt")).casefold()
            for values in split_entries.values() for x in values]
    if len(keys) != len(set(keys)):
        raise ValueError("Images would collide in the generated label paths")
    if not annotations.is_dir():
        raise FileNotFoundError(annotations)
    return split_entries, names, class_ids


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def link_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        raise OSError("Hardlink failed. Put output on the same volume as images, "
                      "or explicitly use --image-mode copy (uses extra disk space).") from exc


def prepare(annotations: Path, splits: Path, output: Path,
            images: Path | None = None, skip_invalid: bool = False,
            expected_classes: int = 130, image_mode: str = "hardlink",
            require_all_classes: bool = True) -> dict:
    annotations, splits, output = annotations.resolve(), splits.resolve(), output.resolve()
    images = images.resolve() if images is not None else None
    split_entries, names, class_ids = load_sources(annotations, splits)
    if len(names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} breeds, found {len(names)}")
    # Refuse reuse so stale labels cannot leak into a different run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    if images is not None and not images.is_dir():
        raise FileNotFoundError(images)
    output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "annotation_audit" if images is None else "conversion",
              "annotations": str(annotations), "images": str(images) if images else None,
              "coordinate_convention": "clipped continuous image boundaries [0,W] / [0,H]",
              "class_count": len(names), "classes": names, "splits": {},
              "require_all_classes_per_split": require_all_classes,
              "errors": [], "clipped_boxes": 0, "object_count_histogram": {},
              "image_storage": {}, "ready_for_training": False}
    histogram, storage = Counter(), Counter()
    try:
        if images is not None:
            from PIL import Image
        for split, entries in split_entries.items():
            accepted, per_class = [], Counter()
            for index, relative in enumerate(entries):
                try:
                    annotation = parse_annotation(annotation_path(annotations, relative), class_ids)
                    if images is not None:
                        source = images / relative
                        with Image.open(source) as image:
                            if image.size != (annotation["width"], annotation["height"]):
                                raise ValueError(f"Image/XML size mismatch: {image.size} vs "
                                                 f"{annotation['width'], annotation['height']}")
                            if image.getexif().get(274, 1) not in (None, 1):
                                raise ValueError("EXIF rotation present; review box orientation first")
                            image.verify()
                except (OSError, ValueError, ET.ParseError) as exc:
                    report["errors"].append({"split": split, "image": relative,
                                             "error": str(exc)})
                    if not skip_invalid:
                        raise ValueError(f"{split}/{relative}: {exc}") from exc
                    continue
                labels = annotation["labels"]
                histogram[len(labels)] += 1
                report["clipped_boxes"] += annotation["clipped_boxes"]
                per_class.update(names[label[0]] for label in labels)
                if images is not None:
                    destination = output / "images" / split / relative
                    storage[link_image(source, destination, image_mode)] += 1
                    label_path = output / "labels" / split / Path(relative).with_suffix(".txt")
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("".join(
                        f"{label[0]} " + " ".join(f"{x:.8f}" for x in label[1:]) + "\n"
                        for label in labels), encoding="utf-8")
                accepted.append(f"./images/{split}/{relative}")
                if (index + 1) % 5000 == 0:
                    print(f"{split}: {index + 1}/{len(entries)} checked", flush=True)
            report["splits"][split] = {"requested": len(entries), "accepted": len(accepted),
                                        "breed_instances": dict(per_class)}
            if images is not None:
                (output / f"{split}.txt").write_text("\n".join(accepted) + "\n", encoding="utf-8")
        report["object_count_histogram"] = dict(histogram)
        report["image_storage"] = dict(storage)
        write_json(output / "classes.json", {str(i): name for i, name in enumerate(names)})
        if images is not None:
            missing = {split: sorted(set(names) - set(info["breed_instances"]))
                       for split, info in report["splits"].items()}
            report["missing_classes_per_split"] = missing
            if require_all_classes and any(missing.values()):
                raise ValueError(f"Breeds missing after conversion: {missing}")
            # JSON is valid YAML; Ultralytics' YAML loader accepts it.
            write_json(output / "data.yaml", {"path": output.as_posix(), "train": "train.txt",
                                             "val": "val.txt", "names": dict(enumerate(names))})
            report["ready_for_training"] = True
        return report
    finally:
        write_json(output / "preparation_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Extracted High-Annotations directory (breed folders inside)")
    parser.add_argument("--splits", type=Path, required=True,
                        help="Extracted TrainAndValList directory")
    parser.add_argument("--images", type=Path, help="Extracted image root, with breed folders inside")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-invalid", action="store_true",
                        help="Skip invalid images/XMLs, recording each exclusion in the report")
    parser.add_argument("--expected-classes", type=int, default=130)
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    report = prepare(args.annotations, args.splits, args.output, args.images,
                     args.skip_invalid, args.expected_classes, args.image_mode)
    print(json.dumps({k: report[k] for k in ("mode", "class_count", "ready_for_training",
                                            "object_count_histogram", "clipped_boxes")}, indent=2))
    print(f"Excluded entries: {len(report['errors'])}; report: {args.output.resolve() / 'preparation_report.json'}")


if __name__ == "__main__":
    main()