"""Visible, logged download -> 70/15/15 preparation -> training -> held-out evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .project_paths import ROOT, LOGS
else:
    from project_paths import ROOT, LOGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\Datasets\TsinghuaDogs"))
    parser.add_argument("--name", default=f"yolo12s_960_full_701515_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    metadata, archives = data_root / "metadata", data_root / "original"
    converted = data_root / "yolo_70_15_15_seed42"
    dataset = converted / "data.yaml"
    scripts = ROOT / "scripts"
    project = ROOT / "weights"
    run = project / args.name
    LOGS.mkdir(parents=True, exist_ok=True)
    state_path = LOGS / f"{args.name}.pipeline.json"
    state = {"status": "starting", "data_root": str(data_root), "dataset": str(dataset),
             "run_directory": str(run), "batch": 4, "imgsz": 960, "epochs": 100,
             "split": [70, 15, 15], "seed": 42}

    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    def execute(stage, script, *parameters):
        state.update(status="running", stage=stage, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_state()
        command = [sys.executable, "-u", str(scripts / script), *map(str, parameters)]
        print(f"\n{'=' * 72}\nSTAGE: {stage}\n{subprocess.list2cmdline(command)}\n{'=' * 72}", flush=True)
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}) as child:
            state["child_pid"] = child.pid
            save_state()
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if child.wait() != 0:
                raise RuntimeError(f"Stage '{stage}' failed with exit code {child.returncode}; training not advanced")

    save_state()
    with (LOGS / f"{args.name}.pipeline.log").open("a", encoding="utf-8") as log:
        try:
            if run.exists():
                raise FileExistsError(f"Refusing to reuse training run: {run}")
            if not dataset.exists():
                extracted = archives / ".extraction_complete"
                if not extracted.exists():
                    execute("download_and_extract", "download_dataset.py", "--metadata", metadata,
                            "--images", archives, "--extract")
                    extracted.write_text("7-Zip extraction succeeded\n", encoding="utf-8")
                execute("validate_deduplicate_split_convert", "prepare_full_dataset.py",
                        "--annotations", metadata / "High-Annotations", "--splits", metadata / "TrainAndValList",
                        "--images", archives / "high-resolution", "--output", converted, "--seed", "42")
            report = json.loads((converted / "preparation_report.json").read_text(encoding="utf-8"))
            if not report.get("ready_for_training") or report.get("source_images") != 70428:
                raise ValueError("Full dataset is not verified ready")
            if report.get("requested_ratios") != [0.7, 0.15, 0.15] or report.get("seed") != 42:
                raise ValueError("Dataset split does not match requested 70/15/15 seed 42")
            state["split_counts"] = {split: report["splits"][split]["images"] for split in ("train", "val", "test")}
            execute("preflight", "train_breeds.py", "--data", dataset, "--batch", "4", "--check-only")
            execute("one_epoch_integration_check", "train_breeds.py", "--data", dataset,
                    "--batch", "4", "--project", project, "--name", args.name, "--smoke-test")
            execute("full_training", "train_breeds.py", "--data", dataset, "--batch", "4",
                    "--project", project, "--name", args.name)
            execute("held_out_test_evaluation", "evaluate_breeds.py", "--data", dataset,
                    "--weights", run / "weights" / "best.pt", "--output", run)
            state.update(status="completed", stage="completed")
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            print(f"\nPIPELINE {state['status'].upper()}: {exc}", flush=True)
            raise
        finally:
            save_state()


if __name__ == "__main__":
    main()