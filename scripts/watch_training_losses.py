"""Save exact milestone loss plots while an existing training process continues."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

if __package__:
    from .plot_training_losses import plot_snapshot, read_snapshot
    from .project_paths import OUTPUTS
else:
    from plot_training_losses import plot_snapshot, read_snapshot
    from project_paths import OUTPUTS


class LossMilestoneMonitor:
    def __init__(self, source: Path, output: Path, epochs=(40, 70, 100),
                 pipeline_state: Path | None = None):
        self.source = source.resolve()
        self.output = output.resolve()
        self.epochs = tuple(sorted(set(epochs)))
        if not self.epochs or any(isinstance(e, bool) or not isinstance(e, int) or e < 1
                                 for e in self.epochs):
            raise ValueError("Milestones must be positive integer epochs")
        if self.output.is_relative_to(self.source.parent):
            raise ValueError("Save plots outside the training run directory")
        self.pipeline_state = pipeline_state
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = {"status": "waiting", "source_csv": str(self.source),
                      "milestones": list(self.epochs), "generated": {}, "last_completed_epoch": 0}
        self.last_error = None

    def save_state(self):
        self.state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        destination = self.output / "monitor_state.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def snapshot_exists(self, destination: Path, epoch: int) -> bool:
        try:
            metadata = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
            return (metadata["last_completed_epoch"] == epoch
                    and Path(metadata["source_csv"]).resolve() == self.source
                    and destination.is_file() and destination.with_suffix(".csv").is_file())
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def check(self) -> bool:
        """Poll once. Return True when all milestones exist or training has stopped."""
        try:
            rows, _ = read_snapshot(self.source)
            latest = int(rows[-1]["epoch"])
            self.state["last_completed_epoch"] = latest
            self.state.pop("last_read_error", None)
            for epoch in self.epochs:
                key = str(epoch)
                if key in self.state["generated"] or latest < epoch:
                    continue
                destination = self.output / f"losses_epoch_{epoch:03d}.png"
                if not self.snapshot_exists(destination, epoch):
                    plot_snapshot(self.source, destination, through_epoch=epoch)
                self.state["generated"][key] = str(destination)
                print(f"Loss snapshot through epoch {epoch}: {destination}", flush=True)
        except (OSError, ValueError) as exc:
            # The CSV may not exist before the first validation finishes, or
            # may be temporarily unavailable while Windows appends a record.
            self.state["last_read_error"] = str(exc)
            if str(exc) != self.last_error:
                print(f"Waiting for completed results: {exc}", flush=True)
                self.last_error = str(exc)

        if len(self.state["generated"]) == len(self.epochs):
            self.state["status"] = "completed"
        else:
            self.state["status"] = "waiting"
            # Training has separate terminal status files; do not keep a
            # monitor alive forever after a reported interruption/failure.
            paths = [self.source.parent / "training_state.json"]
            if self.pipeline_state is not None:
                paths.append(self.pipeline_state)
            for path in paths:
                try:
                    training = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    continue
                if training.get("status") in {"failed", "interrupted", "completed"}:
                    self.state.update(status="stopped_before_all_milestones",
                                      training_status=training["status"],
                                      missing_epochs=[e for e in self.epochs
                                                      if str(e) not in self.state["generated"]])
                    break
        self.save_state()
        return self.state["status"] != "waiting"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, nargs="+", default=[40, 70, 100])
    parser.add_argument("--pipeline-state", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--once", action="store_true", help="Check once without waiting")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    output = args.output or OUTPUTS / "training_plots" / args.csv.resolve().parent.name
    monitor = LossMilestoneMonitor(args.csv, output, args.epochs, args.pipeline_state)
    print(f"Watching {monitor.source}; milestones {monitor.epochs}", flush=True)
    try:
        while True:
            done = monitor.check()
            if done or args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        monitor.state["status"] = "monitor_interrupted"
        monitor.save_state()
        return
    if monitor.state["status"] == "stopped_before_all_milestones":
        raise SystemExit(1)


if __name__ == "__main__":
    main()