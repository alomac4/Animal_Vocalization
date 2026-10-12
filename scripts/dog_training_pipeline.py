"""Logged Open Images preparation -> smoke test -> dog training -> test evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

if __package__:
    from .project_paths import ROOT, LOGS
    from .prepare_openimages_dogs import write_json
    from .train_breeds import validate_dataset
else:
    from project_paths import ROOT, LOGS
    from prepare_openimages_dogs import write_json
    from train_breeds import validate_dataset


def training_config(previous: Path) -> dict:
    saved = yaml.safe_load(previous.read_text(encoding="utf-8"))
    # Preserve saved optimizer, augmentation, loss and validation settings.
    excluded = {"task", "mode", "model", "data", "project", "name", "save_dir", "exist_ok", "resume", "cfg"}
    config = {key: value for key, value in saved.items() if key not in excluded}
    expected = {"epochs": 100, "imgsz": 960, "batch": 4, "workers": 4, "amp": True, "seed": 42,
                "patience": 0, "close_mosaic": 10}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Previous training configuration differs from approved settings")
    return config


def audit_dataset(dataset: Path) -> dict:
    data = validate_dataset(dataset)
    if data["names"] != {0: "dog"}:
        raise ValueError("Dataset must have exactly class 0: dog")
    root = Path(data["path"])
    report = json.loads((root / "preparation_report.json").read_text())
    if not report.get("ready_for_training") or not 100 <= report["valid_unique_images"] <= 15000:
        raise ValueError("Dataset preparation report is not valid")
    provenance = {r["image"]: r for r in json.loads((root / "image_provenance.json").read_text())}
    seen, count = set(), 0
    for split in ("train", "val", "test"):
        for line in (root / data[split]).read_text().splitlines():
            image = Path(line)
            record = provenance[image.as_posix()]
            if record["pixel_sha256"] in seen:
                raise ValueError("Decoded-pixel duplicate between dataset entries")
            seen.add(record["pixel_sha256"])
            label = root / "labels" / (image.stem + ".txt")
            rows = label.read_text().splitlines()
            if len(rows) != len(record["boxes"]):
                raise ValueError(f"Wrong number of boxes in {label}")
            for row in rows:
                values = row.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError(f"Invalid dog label: {label}")
                x, y, w, h = map(float, values[1:])
                if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1
                        and x - w / 2 >= -1e-7 and y - h / 2 >= -1e-7
                        and x + w / 2 <= 1 + 1e-7 and y + h / 2 <= 1 + 1e-7):
                    raise ValueError(f"Out-of-bounds box: {label}")
            count += 1
    if count != report["valid_unique_images"]:
        raise ValueError("Manifest count disagrees with report")
    return report


def main() -> None:
    # Redirected Windows console streams can otherwise use cp1252 and fail on
    # Unicode progress bars before any epoch starts.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\Datasets\OpenImagesDogs\yolo_70_15_15_seed42"))
    parser.add_argument("--previous-args", type=Path, default=ROOT / "weights" / "yolo12s_960_full_701515_20261007_014943" / "args.yaml")
    parser.add_argument("--name", default=f"yolo12s_960_openimages_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--single-dog-augmentation", action="store_true")
    parser.add_argument("--smoke-only", action="store_true", help="Stop successfully after the integration test")
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.strip():
        parser.error("--name must be a single directory name")
    project = ROOT / "models" / "dog_recognition"
    run = project / args.name
    model = project / "yolo12s.pt"
    root = args.data_root.resolve()
    dataset = root / "data.yaml"
    if run.exists() or (project / f"smoke_{args.name}").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    if not model.is_file():
        raise FileNotFoundError(model)
    LOGS.mkdir(parents=True, exist_ok=True)
    configuration = LOGS / f"{args.name}.training.yaml"
    configuration.write_text(yaml.safe_dump(training_config(args.previous_args.resolve()), sort_keys=False), encoding="utf-8")
    state_path = LOGS / f"{args.name}.pipeline.json"
    state = {"status": "starting", "dataset": str(dataset), "run_directory": str(run),
             "configuration": str(configuration), "initial_weights": str(model), "pipeline_pid": os.getpid(),
             "max_images": 15000, "epoch_target": 100, "log": str(LOGS / f"{args.name}.pipeline.log")}

    def execute(stage: str, script: str, *parameters) -> None:
        state.update(status="running", stage=stage, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(state_path, state)
        command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, parameters)]
        print(f"\nSTAGE: {stage}\n{subprocess.list2cmdline(command)}", flush=True)
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}) as child:
            state["child_pid"] = child.pid
            write_json(state_path, state)
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if child.wait() != 0:
                raise RuntimeError(f"Stage {stage} failed with exit code {child.returncode}; pipeline stopped")
        state.pop("child_pid", None)

    with (LOGS / f"{args.name}.pipeline.log").open("a", encoding="utf-8") as log:
        try:
            execute("download_validate_split", "prepare_openimages_dogs.py", "--output", root)
            report = audit_dataset(dataset)
            state["split_counts"] = report["splits"]
            common = ["--data", dataset, "--model", model, "--config", configuration,
                      "--project", project, "--name", args.name]
            if args.single_dog_augmentation:
                common.append("--single-dog-augmentation")
                state["single_dog_augmentation"] = True
            execute("preflight", "train_breeds.py", *common, "--check-only")
            execute("one_epoch_smoke_test", "train_breeds.py", *common, "--smoke-test")
            smoke = project / f"smoke_{args.name}"
            smoke_summary = json.loads((smoke / "training_summary.json").read_text())
            if smoke_summary["class_names"] != {"0": "dog"}:
                raise ValueError("Smoke checkpoint class map is not dog-only")
            if args.smoke_only:
                state.update(status="completed", stage="smoke_test_completed")
                return
            execute("full_training", "train_breeds.py", *common)
            execute("held_out_test_evaluation", "evaluate_breeds.py", "--data", dataset,
                    "--weights", run / "weights" / "best.pt", "--output", run)
            state.update(status="completed", stage="completed")
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            print(f"PIPELINE {state['status'].upper()}: {exc}", flush=True)
            raise
        finally:
            state.pop("child_pid", None)
            write_json(state_path, state)


if __name__ == "__main__":
    main()