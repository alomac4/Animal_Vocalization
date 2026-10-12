"""Download and validate a capped, one-class Open Images dog dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
import time
import urllib.request

from PIL import Image

BASE = "https://storage.googleapis.com/openimages/"
DOG = "/m/0bt9lr"
BOX_FILES = {
    "train": "v6/oidv6-train-annotations-bbox.csv",
    "validation": "v5/validation-annotations-bbox.csv",
    "test": "v5/test-annotations-bbox.csv",
}
NEGATIVE_FILE = "v5/train-annotations-human-imagelabels-boxable.csv"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(url: str, destination: Path, retries: int = 4, progress: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                remaining = int(response.headers.get("Content-Length", 0))
                total, done, last = offset + remaining, offset, time.monotonic()
                with partial.open("ab" if append else "wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        done += len(chunk)
                        if progress and time.monotonic() - last > 10:
                            print(f"Downloading {destination.name}: {done / 2**20:.0f}/{total / 2**20:.0f} MiB", flush=True)
                            last = time.monotonic()
                if remaining and done != total:
                    raise IOError(f"Incomplete download: {destination}")
            partial.replace(destination)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def dog_rows(path: Path) -> tuple[dict, set]:
    boxes, excluded = {}, set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["LabelName"] != DOG:
                continue
            image_id = row["ImageID"]
            if any(int(row.get(flag, "0") or 0) for flag in ("IsGroupOf", "IsDepiction", "IsInside")):
                excluded.add(image_id)
                continue
            x1, x2, y1, y2 = [float(row[k]) for k in ("XMin", "XMax", "YMin", "YMax")]
            if float(row.get("Confidence", 1)) != 1 or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                excluded.add(image_id)
                continue
            box = [x1, x2, y1, y2]
            if box not in boxes.setdefault(image_id, []):
                boxes[image_id].append(box)
    return {key: value for key, value in boxes.items() if key not in excluded}, excluded


def negative_ids(path: Path, forbidden: set) -> list[str]:
    absent, present = set(), set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["LabelName"] == DOG:
                (absent if float(row["Confidence"]) == 0 else present).add(row["ImageID"])
    return sorted(absent - present - forbidden)


def select_records(positives: list[dict], negatives: list[dict], maximum: int, seed: int) -> list[dict]:
    if not 10 <= maximum <= 15000:
        raise ValueError("Image cap must be between 10 and 15000")
    rng = random.Random(seed)
    multi = sorted([r for r in positives if len(r["boxes"]) >= 2], key=lambda r: r["id"])
    single = sorted([r for r in positives if len(r["boxes"]) == 1], key=lambda r: r["id"])
    negatives = sorted(negatives, key=lambda r: r["id"])
    for group in (multi, single, negatives):
        rng.shuffle(group)
    positive_limit = maximum - min(len(negatives), maximum // 10)
    chosen = (multi + single)[:positive_limit]
    return chosen + negatives[:min(maximum // 10, maximum - len(chosen))]


def label_lines(boxes: list[list[float]]) -> str:
    return "".join(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}\n"
                   for x1, x2, y1, y2 in sorted(boxes))


def assign_splits(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng, splits = random.Random(seed), {s: [] for s in ("train", "val", "test")}
    # Stratify on negative, single-dog, and multi-dog scenes.
    for category in (0, 1, 2):
        group = sorted([r for r in records if min(len(r["boxes"]), 2) == category], key=lambda r: r["id"])
        rng.shuffle(group)
        a, b = int(len(group) * .7), int(len(group) * .15)
        for split, part in zip(splits, (group[:a], group[a:a+b], group[a+b:])):
            splits[split].extend(part)
    # The shared smoke trainer selects the first eight entries. Interleave
    # categories so its tiny subset exercises positives and negatives.
    for split, subset in splits.items():
        buckets = [[r for r in subset if min(len(r["boxes"]), 2) == category] for category in (2, 1, 0)]
        splits[split] = [r for group in zip_longest(*buckets) for r in group if r is not None]
    return splits


def fetch_image(record: dict, root: Path) -> dict:
    image = root / "images" / f"{record['id']}.jpg"
    url = f"https://open-images-dataset.s3.amazonaws.com/{record['source_split']}/{record['id']}.jpg"
    download(url, image)
    try:
        with Image.open(image) as opened:
            opened.load()
            # Reject orientation changes rather than silently misaligning boxes.
            orientation = opened.getexif().get(274, 1)
            if orientation not in (1, None):
                raise ValueError(f"EXIF orientation {orientation} requires annotation audit")
            rgb = opened.convert("RGB")
            digest = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
            width, height = rgb.size
            if min(width, height) < 8:
                raise ValueError("Image dimensions too small")
    except Exception:
        # Do not reuse corrupt image downloads on a subsequent attempt.
        image.unlink(missing_ok=True)
        raise
    return {**record, "image": image.as_posix(), "width": width, "height": height, "pixel_sha256": digest}


def prepare(root: Path, maximum: int = 15000, seed: int = 42, workers: int = 12) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    report_path, yaml_path = root / "preparation_report.json", root / "data.yaml"
    if yaml_path.exists():
        report = json.loads(report_path.read_text())
        if not report.get("ready_for_training") or report["maximum_images"] != maximum or report["seed"] != seed:
            raise ValueError("Existing dataset does not match requested preparation")
        return report
    metadata = root / "metadata"
    plan_path = root / "selection.json"
    if not plan_path.exists():
        positives, forbidden = [], set()
        for split, relative in BOX_FILES.items():
            path = metadata / Path(relative).name
            print(f"METADATA: {split} bounding boxes", flush=True)
            download(BASE + relative, path, progress=True)
            boxes, excluded = dog_rows(path)
            forbidden.update(boxes)
            forbidden.update(excluded)
            positives.extend({"id": key, "source_split": split, "boxes": value} for key, value in boxes.items())
            print(f"{split}: {len(boxes)} eligible dog images, {len(excluded)} excluded images", flush=True)
        path = metadata / Path(NEGATIVE_FILE).name
        download(BASE + NEGATIVE_FILE, path, progress=True)
        negatives = [{"id": key, "source_split": "train", "boxes": []} for key in negative_ids(path, forbidden)]
        ids = [r["id"] for r in positives]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated IDs across official splits require audit")
        plan = select_records(positives, negatives, maximum, seed)
        write_json(plan_path, {"maximum_images": maximum, "seed": seed, "records": plan,
                              "eligible_positives": len(positives), "eligible_negatives": len(negatives)})
    selection = json.loads(plan_path.read_text())
    if selection["maximum_images"] != maximum or selection["seed"] != seed:
        raise ValueError("Cached selection differs from requested settings")
    records = selection["records"]
    print(f"Downloading/validating {len(records)} images ({sum(len(r['boxes'])>=2 for r in records)} multi-dog)", flush=True)
    valid, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_image, r, root): r for r in records}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                valid.append(future.result())
            except Exception as exc:
                errors.append({"id": futures[future]["id"], "error": str(exc)})
            if index % 100 == 0 or index == len(records):
                print(f"Images checked: {index}/{len(records)}; failures: {len(errors)}", flush=True)
                write_json(root / "download_state.json", {"checked": index, "total": len(records), "failures": errors})
    if len(valid) < len(records) * .98 or len(valid) < 100:
        raise RuntimeError("Too many missing/invalid images; refusing training. Rerun to retry downloads.")
    groups = {}
    for record in valid:
        groups.setdefault(record["pixel_sha256"], []).append(record)
    unique, duplicates = [], []
    for group in groups.values():
        group.sort(key=lambda r: r["id"])
        signatures = {label_lines(r["boxes"]) for r in group}
        if len(group) > 1:
            duplicates.append({"ids": [r["id"] for r in group], "conflicting_labels": len(signatures) != 1})
        if len(signatures) == 1:
            unique.append(group[0])
    if len(unique) < 100:
        raise ValueError("Too few unique images")
    labels = root / "labels"
    labels.mkdir(exist_ok=True)
    splits = assign_splits(unique, seed)
    split_report = {}
    for split, subset in splits.items():
        if not subset or not any(r["boxes"] for r in subset):
            raise ValueError(f"No positive images in {split}")
        for record in subset:
            (labels / f"{record['id']}.txt").write_text(label_lines(record["boxes"]), encoding="utf-8")
        (root / f"{split}.txt").write_text("\n".join(r["image"] for r in subset) + "\n", encoding="utf-8")
        split_report[split] = {"images": len(subset), "boxes": sum(len(r["boxes"]) for r in subset),
                               "multi_dog_images": sum(len(r["boxes"]) >= 2 for r in subset),
                               "negative_images": sum(not r["boxes"] for r in subset)}
    write_json(root / "image_provenance.json", unique)
    report = {"ready_for_training": True, "maximum_images": maximum, "seed": seed,
              "selected_images": len(records), "valid_unique_images": len(unique), "splits": split_report,
              "requested_ratios": [.7, .15, .15], "errors": errors, "duplicate_groups": duplicates,
              "source": "Open Images V6/V5 bounding boxes (linked by V7 download page)",
              "split_policy": "Resplit official source pools, stratified by dog count, after exact decoded-pixel deduplication",
              "negative_policy": "Human-verified Dog Confidence=0, excluding positive/flagged dog IDs",
              "filters": ["exclude entire image if any dog group/depiction/inside/invalid box", "retain occluded/truncated dogs"],
              "sources": [BASE + p for p in BOX_FILES.values()] + [BASE + NEGATIVE_FILE]}
    write_json(report_path, report)
    write_json(yaml_path, {"path": root.as_posix(), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": {0: "dog"}})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 10 <= args.max_images <= 15000 or args.workers < 1:
        parser.error("Require 10..15000 images and at least one download worker")
    prepare(args.output.resolve(), args.max_images, args.seed, args.workers)


if __name__ == "__main__":
    main()