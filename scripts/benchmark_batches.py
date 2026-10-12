"""Benchmark full training batches in isolated processes; never change training defaults."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

if __package__:
    from .project_paths import ROOT, DATASETS, MODELS, PRETRAINED, OUTPUTS, LOGS, configure_ultralytics
    from .train_breeds import validate_dataset
else:
    from project_paths import ROOT, DATASETS, MODELS, PRETRAINED, OUTPUTS, LOGS, configure_ultralytics
    from train_breeds import validate_dataset


def summarize_epochs(epochs: list[dict], warmup_epochs: int) -> dict:
    measured = epochs[warmup_epochs:]
    if not measured or any(row["seconds"] <= 0 or row["images"] <= 0 for row in measured):
        raise ValueError("No valid measured epochs after warmup")
    images = sum(row["images"] for row in measured)
    seconds = sum(row["seconds"] for row in measured)
    return {"measured_images": images, "measured_train_seconds": seconds,
            "images_per_second": images / seconds,
            "epoch_images_per_second": [row["images"] / row["seconds"] for row in measured],
            "median_batch_ms": statistics.median(
                duration * 1000 for row in measured for duration in row["batch_seconds"]),
            "actual_batch_sizes": sorted({size for row in epochs for size in row["batch_sizes"]}),
            "peak_train_allocated_gib": max(row["peak_allocated_gib"] for row in epochs),
            "peak_train_reserved_gib": max(row["peak_reserved_gib"] for row in epochs)}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def worker(args) -> None:
    import torch
    import yaml

    configure_ultralytics()
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.models.yolo.detect import DetectionTrainer

    data = validate_dataset(args.data.resolve())
    train_entries = (Path(data["path"]) / data["train"]).read_text().splitlines()
    count = len([line for line in train_entries if line.strip()])
    if count < args.batch or count % args.batch:
        raise ValueError("Benchmark requires full batches: image count must be a multiple of requested batch")
    if len(data["names"]) != 130:
        raise ValueError("Benchmark must use the full 130-class detection head")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")
    device = args.device
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    report = {"requested_batch": args.batch, "status": "running", "epochs": [],
              "validation_passes": [], "class_count": len(data["names"]), "train_images": count,
              "gpu": properties.name, "total_vram_gib": properties.total_memory / 2**30,
              "torch": torch.__version__, "ultralytics": ultralytics_version,
              "warmup_epochs_excluded": args.warmup_epochs,
              "timing": "CUDA-synchronized epoch wall time including data loading and augmentation; excludes validation/setup",
              "purpose": "throughput and memory only, not model accuracy",
              "physical_gpu_baseline": None}
    try:
        baseline = subprocess.check_output([
            "nvidia-smi", f"--id={device}", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        report["physical_gpu_baseline"] = baseline
    except (OSError, subprocess.SubprocessError):
        pass
    config = yaml.safe_load((ROOT / "configs" / "breed_train.yaml").read_text())
    # Fixed optimizer for a speed comparison: optimizer=auto chooses a different
    # regime for these tiny runs than for 100 epochs on the full dataset.
    config.update(epochs=args.epochs, batch=args.batch, workers=args.workers, device=device,
                  imgsz=992, optimizer="AdamW", lr0=0.0001, warmup_epochs=0.0, nbs=64,
                  val=True, save=False, save_period=-1, plots=False, close_mosaic=0,
                  data=str(args.data.resolve()), project=str(args.run_project.resolve()),
                  name=f"batch_{args.batch}_repeat_{args.repeat}", exist_ok=False)
    report["config"] = config

    class TimedTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            size = int(batch["img"].shape[0])
            if size != args.batch or self.batch_size != args.batch:
                raise RuntimeError(f"Batch changed: requested {args.batch}, actual {size}, configured {self.batch_size}")
            self.benchmark_batch_size = size
            batch = super().preprocess_batch(batch)
            if tuple(batch["img"].shape[-2:]) != (992, 992):
                raise RuntimeError("Unexpected training resolution")
            return batch

        def save_model(self):
            # No benchmark weights: metrics/run args remain in models/benchmarks.
            return False

        def final_eval(self):
            # Validation is already performed every epoch; no best.pt is saved.
            return None

    active = {}

    def epoch_start(trainer):
        # This version auto-halves on first-epoch OOM; forbid that in a benchmark.
        trainer._oom_retries = 3
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        active.clear()
        active.update(start=time.perf_counter(), images=0, batch_sizes=[], batch_seconds=[])

    def batch_start(trainer):
        torch.cuda.synchronize(device)
        active["batch_start"] = time.perf_counter()

    def batch_end(trainer):
        torch.cuda.synchronize(device)
        active["batch_seconds"].append(time.perf_counter() - active["batch_start"])
        active["images"] += trainer.benchmark_batch_size
        active["batch_sizes"].append(trainer.benchmark_batch_size)
        loss = trainer.loss.detach()
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError("Non-finite training loss")

    def epoch_end(trainer):
        torch.cuda.synchronize(device)
        row = {"epoch": trainer.epoch + 1, "seconds": time.perf_counter() - active["start"],
               "images": active["images"], "batch_sizes": active["batch_sizes"],
               "batch_seconds": active["batch_seconds"],
               "gradient_accumulation": trainer.accumulate,
               "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
        if row["images"] != count:
            raise RuntimeError("Epoch image count differs from benchmark manifest")
        report["epochs"].append(row)
        write_report(args.result, report)

    def val_start(validator):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def val_end(validator):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        report["validation_passes"].append({
            "loader_batch_size": validator.dataloader.batch_size,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "free_vram_end_gib": free / 2**30})

    started = time.perf_counter()
    try:
        model = YOLO(str(PRETRAINED / "yolo12s.pt"))
        for event, callback in [("on_train_epoch_start", epoch_start), ("on_train_batch_start", batch_start),
                                ("on_train_batch_end", batch_end), ("on_train_epoch_end", epoch_end),
                                ("on_val_start", val_start), ("on_val_end", val_end)]:
            model.add_callback(event, callback)
        # Invoke the trainer directly: YOLO.train() insists on reloading a saved
        # checkpoint even though this benchmark intentionally saves no weights.
        trainer = TimedTrainer(overrides={**config, "model": str(PRETRAINED / "yolo12s.pt")},
                               _callbacks=model.callbacks)
        trainer.train()
        report.update(summarize_epochs(report["epochs"], args.warmup_epochs))
        report["peak_validation_reserved_gib"] = max(row["peak_reserved_gib"] for row in report["validation_passes"])
        report["peak_reserved_gib"] = max(report["peak_train_reserved_gib"], report["peak_validation_reserved_gib"])
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "oom" if "out of memory" in str(exc).lower() else "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        report["total_wall_seconds"] = time.perf_counter() - started
        write_report(args.result, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key in ["requested_batch", "status", "images_per_second", "peak_reserved_gib", "error"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASETS / "tsinghua_dogs" / "benchmark" / "yolo" / "data.yaml")
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-project", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs <= args.warmup_epochs or args.warmup_epochs < 1 or args.repeats < 1 or args.workers < 0:
        parser.error("Need >=1 warmup epoch, more total epochs, positive repeats, nonnegative workers")
    if any(batch <= 0 for batch in args.batches) or (args.worker and (not args.batch or not args.result or not args.run_project)):
        parser.error("Invalid batch/worker parameters")
    if args.worker:
        worker(args)
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output.resolve() if args.output else OUTPUTS / "benchmarks" / stamp
    output.mkdir(parents=True, exist_ok=False)
    logs = LOGS / "benchmarks" / output.name
    logs.mkdir(parents=True, exist_ok=True)
    run_project = MODELS / "benchmarks" / output.name
    results = []
    for repeat in range(1, args.repeats + 1):
        # Reverse second pass to reduce bias from increasing GPU temperature/order.
        order = args.batches if repeat % 2 else list(reversed(args.batches))
        for batch in order:
            result = output / f"batch_{batch}_repeat_{repeat}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data", str(args.data.resolve()),
                       "--batch", str(batch), "--repeat", str(repeat), "--epochs", str(args.epochs),
                       "--warmup-epochs", str(args.warmup_epochs), "--workers", str(args.workers),
                       "--device", str(args.device), "--result", str(result), "--run-project", str(run_project)]
            print(f"Benchmarking batch {batch}, repeat {repeat}/{args.repeats}", flush=True)
            with (logs / f"batch_{batch}_repeat_{repeat}.log").open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
            row = json.loads(result.read_text()) if result.exists() else {
                "requested_batch": batch, "status": "failed", "error": f"Worker exited {completed.returncode}; see log"}
            results.append(row)
            print(f"  {row['status']}: {row.get('images_per_second', 0):.2f} images/s; "
                  f"{row.get('peak_reserved_gib', 0):.2f} GiB peak reserved", flush=True)
            write_report(output / "results.json", {"runs": results, "complete": False})
    summary = []
    for batch in args.batches:
        rows = [row for row in results if row["requested_batch"] == batch]
        passed = [row for row in rows if row["status"] == "ok"]
        item = {"batch": batch, "status": "ok" if len(passed) == args.repeats else "failed",
                "passed_runs": len(passed), "total_runs": args.repeats}
        if passed:
            speeds = [row["images_per_second"] for row in passed]
            item.update(images_per_second=statistics.median(speeds),
                        min_images_per_second=min(speeds), max_images_per_second=max(speeds),
                        peak_allocated_gib=max(row["peak_train_allocated_gib"] for row in passed),
                        peak_reserved_gib=max(row["peak_reserved_gib"] for row in passed),
                        peak_validation_reserved_gib=max(row["peak_validation_reserved_gib"] for row in passed))
        summary.append(item)
    valid = [row for row in summary if row["status"] == "ok"]
    fastest = max(valid, key=lambda row: row["images_per_second"])["batch"] if valid else None
    report = {"complete": True, "dataset": str(args.data.resolve()), "summary": summary,
              "fastest_measured_batch": fastest, "runs": results,
              "limitations": ["Small stratified sample, not a full-dataset throughput/accuracy test",
                              "Fixed AdamW and no learning-rate warmup for fair timing; production optimizer is auto",
                              "PyTorch reserved memory is not total physical GPU usage",
                              "GPU load, storage, crowded labels and long-run thermals may change results",
                              "Training configuration was not modified"]}
    write_report(output / "results.json", report)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["batch", "status", "passed_runs", "images_per_second", "min_images_per_second",
                  "max_images_per_second", "peak_allocated_gib", "peak_reserved_gib", "peak_validation_reserved_gib"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Fastest measured batch: {fastest}; report: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()