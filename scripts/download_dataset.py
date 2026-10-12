"""Download official metadata or resumable high-resolution archives.

The optional smoke sample streams only the beginning of the first archive;
it is explicitly NOT a representative breed dataset or evaluation set.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import time
import urllib.request
import zipfile
import zlib

if __package__:
    from .prepare_dataset import read_split
    from .project_paths import DATASETS
else:
    from prepare_dataset import read_split
    from project_paths import DATASETS

SITE = "https://cg.cs.tsinghua.edu.cn/ThuDogs/"


def official_image_urls() -> list[str]:
    with urllib.request.urlopen(SITE, timeout=60) as response:
        html = response.read().decode("utf-8")
    # Anchor text identifies the numbered parts, preserving official order.
    import re

    urls = re.findall(r'<a[^>]*href="([^"]+)"[^>]*>high-resolution\.\d+</a>', html)
    if len(urls) != 9:
        raise RuntimeError("Expected nine official archive links; dataset page may have changed")
    return urls


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers.get("Content-Length", "0"))
    if not size:
        raise RuntimeError(f"Server did not provide a file size for {url}")
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"Existing file has wrong size: {destination}")
        print(f"Already downloaded: {destination}", flush=True)
        return
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial download is larger than remote file: {partial}")
    if offset == size:
        partial.replace(destination)
        return
    if shutil.disk_usage(destination.parent).free < size - offset + 2**30:
        raise OSError(f"Insufficient disk space for {destination}")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise RuntimeError("Server did not honor resume offset; partial file left unchanged")
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Received HTML instead of a dataset archive")
        print(f"Downloading {destination.name}: {offset / 2**30:.2f}/{size / 2**30:.2f} GiB", flush=True)
        with partial.open("ab" if offset else "wb") as handle:
            received = offset
            started = last_update = time.monotonic()
            while chunk := response.read(4 * 2**20):
                handle.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last_update >= 10 or received == size:
                    rate = (received - offset) / max(now - started, 0.001)
                    eta = (size - received) / max(rate, 1)
                    print(f"{destination.name}: {received / size:.1%} | "
                          f"{received / 2**30:.2f}/{size / 2**30:.2f} GiB | "
                          f"{rate / 2**20:.2f} MiB/s | ETA {eta / 60:.1f} min", flush=True)
                    last_update = now
    if partial.stat().st_size != size:
        raise IOError(f"Incomplete download: {partial}; rerun to resume")
    partial.replace(destination)


def extract_metadata(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            destination = output / member.filename
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif not destination.exists() or destination.stat().st_size != member.file_size:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 2**20))
        if not chunk:
            raise EOFError("Unexpected end of archive stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def smoke_sample(url: str, metadata: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split_root = metadata / "TrainAndValList"
    wanted = {"train": read_split(split_root / "train.lst")[:4],
              "val": read_split(split_root / "validation.lst")[:2]}
    targets = {entry for values in wanted.values() for entry in values}
    found = set()
    consumed = 0
    with urllib.request.urlopen(url, timeout=120) as stream:
        while found != targets:
            header = read_exact(stream, 30)
            fields = struct.unpack("<4s5H3I2H", header)
            signature, _, flags, compression, _, _, crc, compressed, uncompressed, name_size, extra_size = fields
            if signature != b"PK\x03\x04" or flags & 8 or compressed == 0xFFFFFFFF:
                raise ValueError("Unsupported ZIP header for streaming sample; use full extraction")
            filename = read_exact(stream, name_size).decode("utf-8" if flags & 0x800 else "cp437")
            read_exact(stream, extra_size)
            consumed += 30 + name_size + extra_size + compressed
            if consumed > 256 * 2**20:
                raise RuntimeError("Smoke sample exceeded 256 MiB; stopped to avoid a large download")
            payload = read_exact(stream, compressed)
            prefix = "high-resolution/"
            relative = filename[len(prefix):] if filename.startswith(prefix) else ""
            if relative not in targets:
                continue
            if compression == 8:
                payload = zlib.decompress(payload, -15)
            elif compression != 0:
                raise ValueError(f"Unsupported ZIP compression: {compression}")
            if len(payload) != uncompressed or zlib.crc32(payload) != crc:
                raise ValueError(f"Corrupt image: {filename}")
            destination = output / "images" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            found.add(relative)
            print(f"Smoke image {len(found)}/{len(targets)}: {relative}", flush=True)
    splits = output / "TrainAndValList"
    splits.mkdir()
    for split, filename in (("train", "train.lst"), ("val", "validation.lst")):
        (splits / filename).write_text("\n".join(wanted[split]) + "\n", encoding="utf-8")
    print(f"Downloaded {consumed / 2**20:.1f} MiB to obtain six original sample images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATASETS / "tsinghua_dogs" / "metadata")
    parser.add_argument("--images", type=Path, help="Download all high-resolution archive parts to this directory")
    parser.add_argument("--extract", action="store_true", help="Extract full images using 7-Zip after download")
    parser.add_argument("--seven-zip", type=Path, default=Path(r"C:\Program Files\7-Zip\7z.exe"))
    parser.add_argument("--smoke-sample", type=Path, help="Fetch only six official images to test training")
    args = parser.parse_args()
    if args.extract and not args.images:
        parser.error("--extract requires --images")
    if args.images and args.smoke_sample:
        parser.error("Choose a full download or a smoke sample, not both")
    metadata = args.metadata.resolve()
    for name in ("high-annotations.zip", "TrainValSplit.zip"):
        download(SITE + name, metadata / name)
        extract_metadata(metadata / name, metadata)
    if args.images:
        images = args.images.resolve()
        images.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(images).free < 90 * 2**30:
            raise OSError("Reserve at least 90 GiB for downloaded archives and extracted images")
        urls = official_image_urls()
        for index, url in enumerate(urls, start=1):
            for attempt in range(1, 6):
                try:
                    download(url, images / f"high-resolution.zip.{index:03d}")
                    break
                except (OSError, ValueError) as exc:
                    if attempt == 5:
                        raise
                    print(f"Download interrupted: {exc}; retry {attempt}/4 in 15s", flush=True)
                    time.sleep(15)
        if args.extract:
            if not args.seven_zip.is_file():
                raise FileNotFoundError(args.seven_zip)
            subprocess.run([str(args.seven_zip), "x", str(images / "high-resolution.zip.001"),
                            f"-o{images}", "-y"], check=True)
    if args.smoke_sample:
        smoke_sample(official_image_urls()[0], metadata, args.smoke_sample.resolve())


if __name__ == "__main__":
    main()