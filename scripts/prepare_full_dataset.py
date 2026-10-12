"""Validate/deduplicate original images, then stratify valid breeds into 70/15/15."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

if __package__:
    from .prepare_dataset import annotation_path, load_sources, parse_annotation, link_image, write_json
else:
    from prepare_dataset import annotation_path, load_sources, parse_annotation, link_image, write_json

SPLITS = ("train", "val", "test")
WEIGHTS = (70, 15, 15)


def split_counts(total: int) -> list[int]:
    """Largest-remainder integer allocation; sums exactly to total."""
    counts = [total * weight // 100 for weight in WEIGHTS]
    order = sorted(range(3), key=lambda i: (-(total * WEIGHTS[i] % 100), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def stratified_split(entries: list[str], seed: int = 42) -> dict[str, list[str]]:
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("Input must contain unique image paths")
    groups = defaultdict(list)
    for entry in sorted(entries):
        groups[entry.split("/", 1)[0]].append(entry)
    if any(len(group) < 7 for group in groups.values()):
        raise ValueError("Each breed needs at least 7 unique valid images for a three-way stratified split")
    target = split_counts(len(entries))
    allocations = {breed: [len(group) * weight // 100 for weight in WEIGHTS]
                   for breed, group in groups.items()}
    remaining = [target[i] - sum(row[i] for row in allocations.values()) for i in range(3)]
    # Fill each breed's fractional remainder while respecting exact global totals.
    for breed in sorted(groups):
        row = allocations[breed]
        for _ in range(len(groups[breed]) - sum(row)):
            available = [i for i in range(3) if remaining[i] > 0]
            index = max(available, key=lambda i: (len(groups[breed]) * WEIGHTS[i] / 100 - row[i],
                                                 remaining[i], -i))
            row[index] += 1
            remaining[index] -= 1
    result = {split: [] for split in SPLITS}
    generator = random.Random(seed)
    for breed in sorted(groups):
        group = groups[breed]
        generator.shuffle(group)
        start = 0
        for split, count in zip(SPLITS, allocations[breed]):
            result[split].extend(group[start:start + count])
            start += count
    for values in result.values():
        generator.shuffle(values)
    if [len(result[split]) for split in SPLITS] != target:
        raise RuntimeError("Split allocation failed")
    return result


def prepare_full(annotations: Path, splits: Path, images: Path, output: Path,
                 seed: int = 42, expected_classes: int = 130,
                 expected_images: int = 70428) -> dict:
    from PIL import Image

    annotations, splits, images, output = [path.resolve() for path in (annotations, splits, images, output)]
    original, names, class_ids = load_sources(annotations, splits)
    entries = sorted(original["train"] + original["val"])
    if len(names) != expected_classes or len(entries) != expected_images:
        raise ValueError(f"Expected {expected_classes} breeds/{expected_images} images; "
                         f"found {len(names)}/{len(entries)}")
    if not images.is_dir():
        raise FileNotFoundError(images)
    # Missing downloads are fatal, never silently turn a partial download into a dataset.
    missing = [entry for entry in entries if not (images / entry).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} original images missing; first: {missing[0]}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"ready_for_training": False, "seed": seed, "requested_ratios": [0.70, 0.15, 0.15],
              "source_images": len(entries), "class_count": len(names), "errors": [],
              "duplicate_groups": [], "splits": {}, "clipped_boxes": 0,
              "deduplication": "SHA-256 of dimensions and decoded RGB pixels",
              "limitations": "No dog-identity labels; near-duplicates are not detected. Custom split, not official benchmark."}
    accepted = {}
    digest_groups = defaultdict(list)
    try:
        for index, relative in enumerate(entries, 1):
            try:
                annotation = parse_annotation(annotation_path(annotations, relative), class_ids)
                with Image.open(images / relative) as image:
                    if image.size != (annotation["width"], annotation["height"]):
                        raise ValueError("Image/XML dimension mismatch")
                    if image.getexif().get(274, 1) not in (None, 1):
                        raise ValueError("EXIF rotation present; box orientation needs review")
                    image.load()  # Verify full decoding, not just the JPEG header.
                    digest = hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()
            except (OSError, ValueError, ET.ParseError) as exc:
                report["errors"].append({"image": relative, "error": str(exc)})
                continue
            accepted[relative] = {"annotation": annotation, "sha256_rgb": digest}
            digest_groups[digest].append(relative)
            if index % 1000 == 0:
                print(f"Validation: {index}/{len(entries)} | accepted {len(accepted)} | "
                      f"excluded {len(report['errors'])}", flush=True)
        for digest, group in digest_groups.items():
            if len(group) < 2:
                continue
            signatures = {json.dumps(accepted[entry]["annotation"]["labels"]) for entry in group}
            # Conflicting labels: exclude the whole group rather than arbitrarily trust one annotation.
            keep = group[0] if len(signatures) == 1 else None
            report["duplicate_groups"].append({"sha256_rgb": digest, "images": group, "kept": keep,
                                               "conflicting_annotations": keep is None})
            for entry in group:
                if entry != keep:
                    del accepted[entry]
        partition = stratified_split(list(accepted), seed)
        inventory = {}
        for split in SPLITS:
            per_class, paths = Counter(), []
            for index, relative in enumerate(partition[split], 1):
                annotation = accepted[relative]["annotation"]
                destination = output / "images" / split / relative
                link_image(images / relative, destination, "hardlink")
                label_path = output / "labels" / split / Path(relative).with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("".join(f"{label[0]} " + " ".join(f"{x:.8f}" for x in label[1:]) + "\n"
                                               for label in annotation["labels"]), encoding="utf-8")
                per_class.update(names[label[0]] for label in annotation["labels"])
                report["clipped_boxes"] += annotation["clipped_boxes"]
                paths.append(f"./images/{split}/{relative}")
                inventory[relative] = {"split": split, "sha256_rgb": accepted[relative]["sha256_rgb"]}
                if index % 5000 == 0:
                    print(f"Conversion {split}: {index}/{len(partition[split])}", flush=True)
            missing_classes = sorted(set(names) - set(per_class))
            if missing_classes:
                raise ValueError(f"Breeds missing in {split}: {missing_classes}")
            (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
            report["splits"][split] = {"images": len(paths), "ratio": len(paths) / len(accepted),
                                       "breed_images": dict(Counter(entry.split('/')[0] for entry in partition[split])),
                                       "breed_instances": dict(per_class)}
        if len({row["sha256_rgb"] for row in inventory.values()}) != len(inventory):
            raise RuntimeError("Duplicate content remains after deduplication")
        report.update(valid_unique_images=len(accepted), excluded_images=len(entries) - len(accepted),
                      path_overlap=0, decoded_pixel_overlap=0)
        write_json(output / "image_inventory.json", inventory)
        write_json(output / "classes.json", dict(enumerate(names)))
        report["ready_for_training"] = True
        write_json(output / "preparation_report.json", report)
        # Write this last: the launcher can only train a fully prepared dataset.
        write_json(output / "data.yaml", {"path": output.as_posix(), "train": "train.txt",
                                          "val": "val.txt", "test": "test.txt", "names": dict(enumerate(names))})
        print(json.dumps({key: report[key] for key in ("valid_unique_images", "excluded_images", "path_overlap",
                                                      "decoded_pixel_overlap")}, indent=2), flush=True)
        print("Split counts: " + ", ".join(f"{split}={len(partition[split])}" for split in SPLITS), flush=True)
        return report
    finally:
        write_json(output / "preparation_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("annotations", "splits", "images", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_full(args.annotations, args.splits, args.images, args.output, args.seed)


if __name__ == "__main__":
    main()