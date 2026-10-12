import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_training_losses import COLUMNS, read_snapshot
from scripts.watch_training_losses import LossMilestoneMonitor


class LossMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "run" / "results.csv"
        self.source.parent.mkdir()
        self.output = self.root / "plots"

    def tearDown(self):
        self.temporary.cleanup()

    def write_epochs(self, count):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for epoch in range(1, count + 1):
                writer.writerow({key: epoch if key == "epoch" else 1 / epoch for key in COLUMNS})

    def test_missing_csv_waits(self):
        monitor = LossMilestoneMonitor(self.source, self.output)
        self.assertFalse(monitor.check())
        self.assertEqual(monitor.state["status"], "waiting")
        self.assertEqual(monitor.state["generated"], {})
        self.assertTrue((self.output / "monitor_state.json").is_file())

    def test_below_milestone_does_not_plot(self):
        self.write_epochs(39)
        monitor = LossMilestoneMonitor(self.source, self.output)
        with patch("scripts.watch_training_losses.plot_snapshot") as plot:
            self.assertFalse(monitor.check())
            plot.assert_not_called()

    def test_due_milestones_use_exact_limits(self):
        self.write_epochs(75)
        monitor = LossMilestoneMonitor(self.source, self.output)
        with patch("scripts.watch_training_losses.plot_snapshot") as plot:
            self.assertFalse(monitor.check())
            self.assertEqual([call.kwargs["through_epoch"] for call in plot.call_args_list], [40, 70])
            monitor.check()
            self.assertEqual(plot.call_count, 2)

    def test_render_restart_and_completion(self):
        self.write_epochs(3)
        before = self.source.read_bytes()
        monitor = LossMilestoneMonitor(self.source, self.output, epochs=(1, 2, 3))
        self.assertTrue(monitor.check())
        self.assertEqual(monitor.state["status"], "completed")
        for epoch in (1, 2, 3):
            png = self.output / f"losses_epoch_{epoch:03d}.png"
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            rows, _ = read_snapshot(png.with_suffix(".csv"))
            self.assertEqual(len(rows), epoch)
        restarted = LossMilestoneMonitor(self.source, self.output, epochs=(1, 2, 3))
        with patch("scripts.watch_training_losses.plot_snapshot") as plot:
            self.assertTrue(restarted.check())
            plot.assert_not_called()
        self.assertEqual(before, self.source.read_bytes())

    def test_failure_stops_with_pending_epochs(self):
        state = self.root / "pipeline.json"
        state.write_text(json.dumps({"status": "failed"}))
        monitor = LossMilestoneMonitor(self.source, self.output, pipeline_state=state)
        self.assertTrue(monitor.check())
        self.assertEqual(monitor.state["missing_epochs"], [40, 70, 100])
        self.assertEqual(monitor.state["training_status"], "failed")

    def test_completed_training_stops_if_target_missing(self):
        self.write_epochs(39)
        (self.source.parent / "training_state.json").write_text(json.dumps({"status": "completed"}))
        monitor = LossMilestoneMonitor(self.source, self.output)
        self.assertTrue(monitor.check())
        self.assertEqual(monitor.state["status"], "stopped_before_all_milestones")

    def test_invalid_output_and_milestones(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            LossMilestoneMonitor(self.source, self.source.parent / "plots")
        for epochs in ((), (0,), (True,), (1.5,)):
            with self.subTest(epochs=epochs), self.assertRaisesRegex(ValueError, "positive integer"):
                LossMilestoneMonitor(self.source, self.output, epochs=epochs)


if __name__ == "__main__":
    unittest.main()