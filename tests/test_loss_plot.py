import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_training_losses import COLUMNS, read_snapshot, plot_snapshot


class LossSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "run" / "results.csv"
        self.source.parent.mkdir()
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for epoch in (1, 2):
                writer.writerow({key: epoch if key == "epoch" else 1 / epoch for key in COLUMNS})

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_rows_parsed(self):
        rows, ignored = read_snapshot(self.source)
        self.assertEqual([row["epoch"] for row in rows], [1, 2])
        self.assertFalse(ignored)

    def test_partial_appended_epoch_ignored(self):
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write("3,0.12,0.")
        rows, ignored = read_snapshot(self.source)
        self.assertEqual(len(rows), 2)
        self.assertTrue(ignored)

    def test_malformed_completed_row_rejected(self):
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write("3,broken\n")
        with self.assertRaisesRegex(ValueError, "Invalid completed CSV row"):
            read_snapshot(self.source)

    def test_run_directory_output_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            plot_snapshot(self.source, self.source.parent / "losses.png")

    def test_plot_rendered_without_modifying_source(self):
        before = self.source.read_bytes()
        modified_at = self.source.stat().st_mtime_ns
        destination = self.root / "outputs" / "losses.png"
        result = plot_snapshot(self.source, destination)
        self.assertEqual(result["last_completed_epoch"], 2)
        self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(destination.with_suffix(".csv").is_file())
        self.assertTrue(destination.with_suffix(".json").is_file())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.source.stat().st_mtime_ns, modified_at)

    def test_exact_epoch_snapshot_excludes_later_rows(self):
        destination = self.root / "outputs" / "epoch1.png"
        result = plot_snapshot(self.source, destination, through_epoch=1)
        self.assertEqual(result["last_completed_epoch"], 1)
        self.assertEqual(result["completed_epochs"], 1)
        rows, _ = read_snapshot(destination.with_suffix(".csv"))
        self.assertEqual([row["epoch"] for row in rows], [1])

    def test_unavailable_epoch_refused(self):
        with self.assertRaisesRegex(ValueError, "not yet available"):
            plot_snapshot(self.source, self.root / "outputs" / "later.png", through_epoch=40)

    def test_invalid_epoch_refused(self):
        for epoch in (0, -1, True, 1.5):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "positive integer"):
                plot_snapshot(self.source, self.root / "outputs" / "invalid.png", through_epoch=epoch)


if __name__ == "__main__":
    unittest.main()