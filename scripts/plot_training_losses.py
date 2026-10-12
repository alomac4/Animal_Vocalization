"""Plot a read-only snapshot of completed epochs without importing the trainer or CUDA."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path

if __package__:
    from .project_paths import OUTPUTS
else:
    from project_paths import OUTPUTS

COMPONENTS = (("box_loss", "Box loss"), ("cls_loss", "Classification loss"),
              ("dfl_loss", "Distribution focal loss (DFL)"))
COLUMNS = ["epoch", *[f"{split}/{component}" for component, _ in COMPONENTS
                         for split in ("train", "val")]]


def read_snapshot(source: Path) -> tuple[list[dict[str, float]], bool]:
    # Read once and close immediately. Only terminated records are accepted so
    # an epoch being appended by the trainer cannot become a partial data point.
    text = source.read_bytes().decode("utf-8-sig")
    incomplete_tail = bool(text and not text.endswith("\n"))
    if incomplete_tail:
        text = text[:text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    if not reader.fieldnames:
        raise ValueError("No completed CSV header available")
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    missing = set(COLUMNS) - set(reader.fieldnames)
    if missing:
        raise ValueError(f"Missing loss columns: {sorted(missing)}")
    rows = []
    for line_number, raw in enumerate(reader, 2):
        try:
            row = {key: float(raw[key]) for key in COLUMNS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid completed CSV row at line {line_number}") from exc
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite loss/epoch at line {line_number}")
        epoch = row["epoch"]
        if epoch < 1 or not epoch.is_integer() or (rows and epoch <= rows[-1]["epoch"]):
            raise ValueError(f"Epochs must be positive integers in increasing order: line {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("No completed epochs available yet")
    return rows, incomplete_tail


def plot_snapshot(source: Path, destination: Path, through_epoch: int | None = None) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source.parent):
        raise ValueError("Save plots outside the training run directory to keep the run untouched")
    if destination.suffix.lower() != ".png":
        raise ValueError("Output must be a .png file")
    rows, ignored_tail = read_snapshot(source)
    if through_epoch is not None:
        if isinstance(through_epoch, bool) or not isinstance(through_epoch, int) or through_epoch < 1:
            raise ValueError("Snapshot epoch must be a positive integer")
        rows = [row for row in rows if row["epoch"] <= through_epoch]
        if not rows or rows[-1]["epoch"] != through_epoch:
            raise ValueError(f"Epoch {through_epoch} is not yet available in completed CSV rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Agg is CPU-only and does not require a window, a GPU, torch or Ultralytics.
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = datetime.now().astimezone().isoformat(timespec="seconds")
    last_epoch = int(rows[-1]["epoch"])
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    try:
        for axis, (component, title) in zip(axes, COMPONENTS):
            marker = "o" if len(rows) == 1 else None
            axis.plot(epochs, [row[f"train/{component}"] for row in rows],
                      label="Training", color="#2563eb", linewidth=2, marker=marker)
            axis.plot(epochs, [row[f"val/{component}"] for row in rows],
                      label="Validation", color="#ea580c", linewidth=2, marker=marker)
            axis.set(title=title, xlabel="Completed epoch", ylabel="Recorded loss")
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.suptitle(f"{source.parent.name}\nTrain / validation loss snapshot through epoch {last_epoch}",
                        fontsize=13)
        temporary = destination.with_suffix(".tmp")
        figure.savefig(temporary, format="png", dpi=160)
        temporary.replace(destination)
    finally:
        plt.close(figure)
    snapshot = destination.with_suffix(".csv")
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"source_csv": str(source), "plot": str(destination), "snapshot_csv": str(snapshot),
                "captured_at": captured, "completed_epochs": len(rows), "last_completed_epoch": last_epoch,
                "ignored_unterminated_tail": ignored_tail, "latest_losses": rows[-1],
                "note": "Read-only, CPU-only snapshot. No trainer, checkpoint or training settings modified."}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Active run's results.csv (read only)")
    parser.add_argument("--output", type=Path, help="PNG outside the training run directory")
    parser.add_argument("--through-epoch", type=int, help="Include only completed epochs up to this milestone")
    args = parser.parse_args()
    destination = args.output or (OUTPUTS / "training_plots" /
        f"{args.csv.resolve().parent.name}_losses_{datetime.now():%Y%m%d_%H%M%S}.png")
    print(json.dumps(plot_snapshot(args.csv, destination, args.through_epoch), indent=2), flush=True)


if __name__ == "__main__":
    main()