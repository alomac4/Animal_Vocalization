import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_benchmark import RemoteSplitZip, fetch_member, stratified_sample
from scripts.benchmark_batches import summarize_epochs


class SampleTests(unittest.TestCase):
    def test_deterministic_unique_stratification(self):
        entries = [f"breed_{breed}/image_{image}.jpg" for breed in range(4) for image in range(6)]
        sample = stratified_sample(entries, 12, 42)
        self.assertEqual(sample, stratified_sample(entries, 12, 42))
        self.assertEqual(len(set(sample)), 12)
        self.assertEqual({x.split("/")[0] for x in sample[:4]}, {f"breed_{x}" for x in range(4)})

    def test_invalid_sample_size(self):
        for count in (0, -1, 2):
            with self.assertRaises(ValueError):
                stratified_sample(["breed/a.jpg"], count, 42)

    def test_seek_origins(self):
        archive = RemoteSplitZip(["a", "b"], [10, 20])
        self.assertEqual(archive.seek(-5, 2), 25)
        self.assertEqual(archive.seek(-10, 1), 15)
        self.assertEqual(archive.tell(), 15)
        with self.assertRaises(ValueError):
            archive.seek(-1)

    def test_range_crosses_part_boundary(self):
        parts = {"https://test.invalid/a": b"abcd", "https://test.invalid/b": b"efghij"}

        class Response(io.BytesIO):
            status = 206

        def request_response(request, timeout):
            start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
            payload = parts[request.full_url]
            response = Response(payload[start:end + 1])
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            return response

        with patch("scripts.prepare_benchmark.urllib.request.urlopen", side_effect=request_response):
            archive = RemoteSplitZip(list(parts), [4, 6])
            self.assertEqual(archive.read_at(2, 6), b"cdefgh")
            archive.seek(7)
            self.assertEqual(archive.read(), b"hij")
            self.assertEqual(archive.read(), b"")

    def test_range_bounds(self):
        archive = RemoteSplitZip(["a"], [10])
        for offset, length in ((-1, 2), (0, -1), (9, 2)):
            with self.assertRaises(ValueError):
                archive.read_at(offset, length)

    def test_zip_member_crc_and_extraction(self):
        content = b"Original image bytes" * 30
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr("high-resolution/breed/dog.jpg", content)
        payload = stream.getvalue()
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            info = package.getinfo("high-resolution/breed/dog.jpg")
        archive = RemoteSplitZip(["a"], [len(payload)])
        archive.read_at = lambda offset, length: payload[offset:offset + length]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dog.jpg"
            fetch_member(archive, info, destination)
            self.assertEqual(destination.read_bytes(), content)
            info.CRC = 0
            with self.assertRaisesRegex(ValueError, "CRC"):
                fetch_member(archive, info, destination)


class TimingTests(unittest.TestCase):
    def test_warmup_excluded_and_actual_batches_reported(self):
        rows = [dict(seconds=20.0, images=16, batch_seconds=[10.0, 10.0], batch_sizes=[8, 8],
                     peak_allocated_gib=2.0, peak_reserved_gib=3.0),
                dict(seconds=2.0, images=16, batch_seconds=[0.5, 0.5], batch_sizes=[8, 8],
                     peak_allocated_gib=4.0, peak_reserved_gib=5.0)]
        report = summarize_epochs(rows, 1)
        self.assertEqual(report["images_per_second"], 8.0)
        self.assertEqual(report["measured_images"], 16)
        self.assertEqual(report["actual_batch_sizes"], [8])
        self.assertEqual(report["median_batch_ms"], 500.0)
        self.assertEqual(report["peak_train_reserved_gib"], 5.0)

    def test_empty_measurements_rejected(self):
        with self.assertRaises(ValueError):
            summarize_epochs([], 1)


if __name__ == "__main__":
    unittest.main()