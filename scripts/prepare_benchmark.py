"""Fetch a small stratified original-image sample using HTTP ZIP range reads."""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import random
import struct
import time
import urllib.request
import zipfile
import zlib

if __package__:
    from .download_dataset import official_image_urls
    from .prepare_dataset import build_classes, prepare, read_split, parse_annotation, annotation_path
    from .project_paths import DATASETS
else:
    from download_dataset import official_image_urls
    from prepare_dataset import build_classes, prepare, read_split, parse_annotation, annotation_path
    from project_paths import DATASETS


def stratified_sample(entries: list[str], count: int, seed: int) -> list[str]:
    if count <= 0 or count > len(entries):
        raise ValueError("Sample count must be positive and no larger than the split")
    groups = defaultdict(list)
    for entry in entries:
        groups[entry.split("/", 1)[0]].append(entry)
    rng = random.Random(seed)
    folders = sorted(groups)
    rng.shuffle(folders)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        for folder in folders:
            if groups[folder]:
                selected.append(groups[folder].pop())
            if len(selected) == count:
                return selected
    return selected


class RemoteSplitZip(io.RawIOBase):
    """Seekable view of numbered ZIP parts; downloads only requested byte ranges."""

    def __init__(self, urls: list[str], sizes: list[int]):
        super().__init__()
        self.urls = urls
        self.sizes = sizes
        self.boundaries = [0]
        for size in sizes:
            self.boundaries.append(self.boundaries[-1] + size)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.boundaries[-1] + offset
        else:
            raise ValueError("Invalid seek origin")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.boundaries[-1]:
            raise ValueError("Range outside archive")
        chunks = []
        while length:
            index = bisect.bisect_right(self.boundaries, offset) - 1
            start = offset - self.boundaries[index]
            amount = min(length, self.sizes[index] - start)
            expected = f"bytes {start}-{start + amount - 1}/"
            for attempt in range(3):
                try:
                    request = urllib.request.Request(self.urls[index], headers={
                        "Range": f"bytes={start}-{start + amount - 1}"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                            raise IOError("Server did not honor byte range; refused full archive download")
                        payload = response.read(amount + 1)
                    if len(payload) != amount:
                        raise IOError("Incomplete archive byte range")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            chunks.append(payload)
            offset += amount
            length -= amount
        return b"".join(chunks)

    def read(self, size=-1):
        remaining = max(0, self.boundaries[-1] - self.position)
        length = remaining if size is None or size < 0 else min(size, remaining)
        payload = self.read_at(self.position, length)
        self.position += length
        return payload


def fetch_member(archive: RemoteSplitZip, info: zipfile.ZipInfo, destination: Path) -> None:
    if info.flag_bits & 1:
        raise ValueError("Encrypted ZIP member unsupported")
    header = archive.read_at(info.header_offset, 30)
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"Invalid local header: {info.filename}")
    name_size, extra_size = fields[-2:]
    raw = archive.read_at(info.header_offset + 30, name_size + extra_size + info.compress_size)
    name = raw[:name_size].decode("utf-8" if info.flag_bits & 0x800 else "cp437")
    if name != info.filename:
        raise ValueError("ZIP local and central names differ")
    payload = raw[name_size + extra_size:]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Unsupported compression")
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise ValueError(f"CRC/size mismatch: {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASETS / "tsinghua_dogs" / "benchmark")
    parser.add_argument("--metadata", type=Path, default=DATASETS / "tsinghua_dogs" / "metadata")
    parser.add_argument("--train-count", type=int, default=192)
    parser.add_argument("--val-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = args.root.resolve()
    metadata = args.metadata.resolve()
    splits = metadata / "TrainAndValList"
    originals = {"train": read_split(splits / "train.lst"), "val": read_split(splits / "validation.lst")}
    selected = {"train": stratified_sample(originals["train"], args.train_count, args.seed),
                "val": stratified_sample(originals["val"], args.val_count, args.seed + 1)}
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"
    selection = {"seed": args.seed, "selected": selected,
                 "purpose": "batch throughput/memory benchmark; not an accuracy evaluation"}
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Existing benchmark selection differs; use a new --root")
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if (root / "yolo" / "data.yaml").exists():
        if not (root / "validated_selection.json").exists():
            raise ValueError("Existing benchmark is missing its validated selection")
        print(f"Benchmark dataset already prepared: {root / 'yolo' / 'data.yaml'}", flush=True)
        return
    all_entries = [entry for entries in selected.values() for entry in entries]
    missing = [entry for entry in all_entries if not (root / "original" / entry).is_file()]
    if missing:
        urls = official_image_urls()

        def remote_size(url):
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
                return int(response.headers["Content-Length"])

        with ThreadPoolExecutor(max_workers=9) as pool:
            sizes = list(pool.map(remote_size, urls))
        archive = RemoteSplitZip(urls, sizes)
        with zipfile.ZipFile(archive) as package:
            members = {entry: package.getinfo("high-resolution/" + entry) for entry in missing}
        total_bytes = sum(info.compress_size for info in members.values())
        if total_bytes > 2**30:
            raise ValueError("Sample download exceeds the 1 GiB budget")
        print(f"Downloading {len(missing)} unique images, {total_bytes / 2**20:.1f} MiB, via ZIP byte ranges", flush=True)

        def fetch(entry):
            fetch_member(archive, members[entry], root / "original" / entry)
            return entry

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for index, entry in enumerate(pool.map(fetch, missing), start=1):
                if index % 16 == 0 or index == len(missing):
                    print(f"Fetched {index}/{len(missing)} images", flush=True)
    # Never rotate/rescale ambiguous original boxes just to complete a benchmark.
    # Replace invalid sample records with validated originals of the same breed.
    from PIL import Image

    names, class_ids = build_classes(originals["train"] + originals["val"])
    excluded = []
    archive = None

    def ensure_archive():
        nonlocal archive
        if archive is None:
            urls = official_image_urls()
            sizes = []
            for url in urls:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
                    sizes.append(int(response.headers["Content-Length"]))
            archive = RemoteSplitZip(urls, sizes)
        return archive

    def validate(entry):
        annotation = parse_annotation(annotation_path(metadata / "High-Annotations", entry), class_ids)
        with Image.open(root / "original" / entry) as image:
            if image.size != (annotation["width"], annotation["height"]):
                raise ValueError(f"Image/XML dimensions differ: {image.size} vs {(annotation['width'], annotation['height'])}")
            if image.getexif().get(274, 1) not in (None, 1):
                raise ValueError("EXIF rotation requires annotation review")
            image.verify()

    rng = random.Random(args.seed)
    for split, entries in selected.items():
        for index, entry in enumerate(entries):
            try:
                validate(entry)
                continue
            except (OSError, ValueError) as exc:
                excluded.append({"image": entry, "reason": str(exc)})
                print(f"Replacing invalid original: {entry}: {exc}", flush=True)
            folder = entry.split("/", 1)[0]
            candidates = [x for x in originals[split] if x.startswith(folder + "/") and x not in all_entries]
            rng.shuffle(candidates)
            for candidate in candidates:
                destination = root / "original" / candidate
                if not destination.exists():
                    remote = ensure_archive()
                    with zipfile.ZipFile(remote) as package:
                        info = package.getinfo("high-resolution/" + candidate)
                    fetch_member(remote, info, destination)
                try:
                    validate(candidate)
                except (OSError, ValueError) as exc:
                    excluded.append({"image": candidate, "reason": str(exc)})
                    continue
                entries[index] = candidate
                all_entries.append(candidate)
                break
            else:
                raise RuntimeError(f"No valid replacement for {entry}")
    (root / "validated_selection.json").write_text(json.dumps({"selected": selected, "excluded": excluded}, indent=2), encoding="utf-8")
    subset_splits = root / "TrainAndValList"
    subset_splits.mkdir(exist_ok=True)
    for split, filename in (("train", "train.lst"), ("val", "validation.lst")):
        (subset_splits / filename).write_text("\n".join(selected[split]) + "\n", encoding="utf-8")
    output = root / "yolo"
    if not (output / "data.yaml").exists():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Inspect failed preparation before removing/retrying: {output}")
        names, _ = build_classes([entry for entries in selected.values() for entry in entries])
        report = prepare(metadata / "High-Annotations", subset_splits, output,
                         images=root / "original", expected_classes=len(names), require_all_classes=False)
        print(f"Prepared {len(names)} breed classes; clipped boxes: {report['clipped_boxes']}", flush=True)
    print(f"Benchmark dataset: {output / 'data.yaml'}", flush=True)


if __name__ == "__main__":
    main()