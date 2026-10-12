import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_breeds import EpochReporter


class EpochReporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trainer = SimpleNamespace(
            save_dir=self.root, start_epoch=0, epoch=0, epochs=100, batch_size=4,
            args=SimpleNamespace(imgsz=960), best=self.root / "weights" / "best.pt",
            last=self.root / "weights" / "last.pt", tloss=[1.0, 2.0, 3.0],
            metrics={"metrics/mAP50(B)": 0.25}, lr={"lr/pg0": 0.001},
            label_loss_items=lambda values: dict(zip(
                ("train/box_loss", "train/cls_loss", "train/dfl_loss"), values)))
        self.reporter = EpochReporter()

    def tearDown(self):
        self.temporary.cleanup()

    def test_live_summary_and_persistent_state(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.reporter.on_train_start(self.trainer)
            self.reporter.on_epoch_start(self.trainer)
            self.reporter.on_epoch_end(self.trainer)
            self.assertIn("[EPOCH 1/100] COMPLETE", output.getvalue())
            self.assertIn("train/box_loss=1", output.getvalue())
            self.assertIn("metrics/mAP50(B)=0.25", output.getvalue())
            self.assertIn("ETA=", output.getvalue())
        state = json.loads((self.root / "training_state.json").read_text())
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["status"], "epoch_complete")
        self.assertEqual(state["batch"], 4)
        self.assertEqual(state["imgsz"], 960)
        self.assertEqual(state["losses"]["train/cls_loss"], 2.0)
        self.assertGreaterEqual(state["eta_seconds"], 0)
        self.assertEqual(len((self.root / "epoch_state.jsonl").read_text().splitlines()), 1)
        self.assertFalse((self.root / "training_state.tmp").exists())

    def test_final_validation_not_logged_as_extra_epoch(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            self.reporter.on_train_start(self.trainer)
            self.reporter.on_epoch_start(self.trainer)
            self.reporter.on_epoch_end(self.trainer)
            self.trainer.epoch = 100
            self.trainer.metrics = {"metrics/mAP50(B)": 0.5}
            self.reporter.on_epoch_end(self.trainer)
            self.reporter.on_train_end(self.trainer)
        state = json.loads((self.root / "training_state.json").read_text())
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["final_best_metrics"]["metrics/mAP50(B)"], 0.5)
        self.assertEqual(len((self.root / "epoch_state.jsonl").read_text().splitlines()), 1)

    def test_resume_reports_only_new_epochs(self):
        self.trainer.start_epoch = 20
        self.trainer.epoch = 20
        with patch("sys.stdout", new_callable=io.StringIO), \
                patch("scripts.train_breeds.time.perf_counter", side_effect=[100.0, 101.0, 110.0]):
            self.reporter.on_train_start(self.trainer)
            self.reporter.on_epoch_start(self.trainer)
            self.reporter.on_epoch_end(self.trainer)
        state = json.loads((self.root / "training_state.json").read_text())
        self.assertEqual(state["epoch"], 21)
        self.assertEqual(state["eta_seconds"], 790.0)

    def test_callbacks_registered(self):
        callbacks = {}
        model = SimpleNamespace(add_callback=lambda event, callback: callbacks.update({event: callback}))
        self.reporter.attach(model)
        self.assertEqual(set(callbacks), {"on_train_start", "on_train_epoch_start",
                                          "on_fit_epoch_end", "on_train_end"})


if __name__ == "__main__":
    unittest.main()