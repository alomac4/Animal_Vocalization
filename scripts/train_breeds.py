"""Train the dog breed detector, or run a one-epoch memory/workflow smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__:
    from .project_paths import ROOT, DATASETS, PRETRAINED, configure_ultralytics
else:
    from project_paths import ROOT, DATASETS, PRETRAINED, configure_ultralytics


def validate_dataset(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(k in data for k in ("path", "train", "val", "names")):
        raise ValueError("Dataset YAML must define path, train, val, and names")
    root = Path(data["path"])
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names = data["names"]
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Class indices must be contiguous starting at zero")
        data["names"] = names
    elif not isinstance(names, list) or not names:
        raise ValueError("Invalid class names")
    split_names = ("train", "val", "test") if data.get("test") else ("train", "val")
    split_paths = {}
    for split in split_names:
        manifest = root / data[split]
        if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing/empty {split} manifest: {manifest}")
        if "test" in split_names:
            entries = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            paths = [(Path(line) if Path(line).is_absolute() else manifest.parent / line).resolve()
                     for line in entries]
            if len(set(paths)) != len(paths):
                raise ValueError(f"Duplicate images in {split} manifest")
            if any(not image.is_file() for image in paths):
                raise ValueError(f"Missing image files in {split} manifest")
            split_paths[split] = set(paths)
    for first, paths in split_paths.items():
        for second, other in split_paths.items():
            if first < second and paths & other:
                raise ValueError(f"{first}/{second} overlap: {len(paths & other)} images")
    data["path"] = root.as_posix()
    return data


def make_smoke_dataset(data: dict, destination: Path, count: int = 8) -> Path:
    root = Path(data["path"])
    destination.mkdir(parents=True, exist_ok=False)
    subset = dict(data)
    for split in ("train", "val"):
        manifest = root / data[split]
        lines = manifest.read_text(encoding="utf-8").splitlines()
        selected = []
        for line in (line.strip() for line in lines):
            if not line:
                continue
            image = Path(line)
            if not image.is_absolute():
                image = (manifest.parent / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            selected.append(image.as_posix())
            if len(selected) == count:
                break
        subset[split] = (destination / f"{split}.txt").as_posix()
        Path(subset[split]).write_text("\n".join(selected) + "\n", encoding="utf-8")
    path = destination / "data.yaml"
    path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    return path


class EpochReporter:
    """Print live epoch summaries and persist state without polling the trainer."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.epoch_started = self.started
        self.last_epoch = 0
        self.completed_epochs = 0
        self.state = {}
        self.active_epoch = None

    def attach(self, model) -> None:
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_epoch_start", self.on_epoch_start)
        model.add_callback("on_fit_epoch_end", self.on_epoch_end)
        model.add_callback("on_train_end", self.on_train_end)

    def write_state(self, trainer) -> None:
        destination = Path(trainer.save_dir) / "training_state.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def on_train_start(self, trainer) -> None:
        self.started = time.perf_counter()
        self.last_epoch = trainer.start_epoch
        self.completed_epochs = 0
        self.state = {"status": "running", "epoch": trainer.start_epoch,
                      "total_epochs": trainer.epochs, "batch": trainer.batch_size,
                      "imgsz": trainer.args.imgsz,
                      "best_weights": str(trainer.best), "last_weights": str(trainer.last)}
        self.write_state(trainer)
        print(f"\nTRAINING STARTED | batch={trainer.batch_size} | "
              f"resolution={trainer.args.imgsz} x {trainer.args.imgsz} | epochs={trainer.epochs}\n"
              f"Checkpoints: {Path(trainer.last).parent}\n"
              f"Live state: {Path(trainer.save_dir) / 'training_state.json'}", flush=True)

    def on_epoch_start(self, trainer) -> None:
        self.epoch_started = time.perf_counter()
        self.active_epoch = trainer.epoch + 1
        self.state.update(status="training", epoch=self.active_epoch)
        self.write_state(trainer)
        print(f"\n[EPOCH {self.active_epoch}/{trainer.epochs}] Training...", flush=True)

    def on_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch + 1
        # Final evaluation of best.pt also invokes on_fit_epoch_end; do not
        # report it as another training epoch or mix best metrics with last loss.
        if epoch != self.active_epoch or epoch <= self.last_epoch:
            return
        now = time.perf_counter()
        self.completed_epochs += 1
        elapsed = now - self.started
        losses = {key: float(value) for key, value in trainer.label_loss_items(trainer.tloss).items()}
        metrics = {key: float(value) for key, value in (trainer.metrics or {}).items()}
        rates = {key: float(value) for key, value in trainer.lr.items()}
        self.state.update(status="epoch_complete", epoch=epoch,
                          epoch_seconds=now - self.epoch_started, elapsed_seconds=elapsed,
                          eta_seconds=elapsed / self.completed_epochs * max(0, trainer.epochs - epoch),
                          losses=losses, metrics=metrics, learning_rates=rates)
        self.write_state(trainer)
        with (Path(trainer.save_dir) / "epoch_state.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.state) + "\n")
        values = " | ".join(f"{key}={value:.5g}" for key, value in {**losses, **metrics, **rates}.items())
        print(f"\n[EPOCH {epoch}/{trainer.epochs}] COMPLETE | "
              f"epoch={self.state['epoch_seconds']:.1f}s | elapsed={elapsed / 60:.1f} min | "
              f"ETA={self.state['eta_seconds'] / 60:.1f} min\n{values}", flush=True)
        self.last_epoch = epoch
        self.active_epoch = None

    def on_train_end(self, trainer) -> None:
        self.state.update(status="completed", elapsed_seconds=time.perf_counter() - self.started,
                          eta_seconds=0,
                          final_best_metrics={key: float(value) for key, value in (trainer.metrics or {}).items()})
        self.write_state(trainer)
        print(f"\nTRAINING COMPLETED | epochs completed through {self.last_epoch}\n"
              f"Best weights: {trainer.best}\nLast weights: {trainer.last}", flush=True)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Converted data.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "breed_train.yaml")
    parser.add_argument("--model", default=str(PRETRAINED / "yolo12s.pt"))
    parser.add_argument("--batch", type=float, help="Integer batch or AutoBatch fraction such as 0.65")
    parser.add_argument("--device", help="CUDA device index, or cpu for debugging")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project", type=Path, default=ROOT / "weights")
    parser.add_argument("--name", help="Run directory name; defaults to yolo12s_<resolution>")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch, up to 8 images per split, same resolution")
    parser.add_argument("--check-only", action="store_true", help="Validate configuration and CUDA without training")
    parser.add_argument("--resume", type=Path, help="Resume last.pt using its saved training settings")
    parser.add_argument("--single-dog-augmentation", action="store_true",
                        help="Train-only mild motion blur and edge truncation on single-dog source images")
    args = parser.parse_args()

    import torch
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trainer_options = {}
    augmentation_policy = None
    if args.single_dog_augmentation:
        if __package__:
            from .dog_augmentations import POLICY, SingleDogTrainer
        else:
            from dog_augmentations import POLICY, SingleDogTrainer
        trainer_options["trainer"] = SingleDogTrainer
        augmentation_policy = POLICY
        print("Single-dog augmentation:", json.dumps(POLICY, indent=2), flush=True)
    if args.batch is not None:
        if args.batch != -1 and (args.batch <= 0 or (args.batch >= 1 and not args.batch.is_integer())):
            parser.error("--batch must be a positive integer, a fraction in (0,1), or -1")
        config["batch"] = int(args.batch) if args.batch.is_integer() else args.batch
    if args.device is not None:
        config["device"] = args.device
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers cannot be negative")
        config["workers"] = args.workers
    if config["imgsz"] % 32:
        parser.error("Image size must be divisible by 32")
    gpu = config.get("device", 0) != "cpu"
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install the CUDA PyTorch wheels documented in README.md")
    if gpu:
        device = int(str(config["device"]))
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}; VRAM: {properties.total_memory / 2**30:.2f} GiB", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(json.dumps(config, indent=2), flush=True)
    if args.resume:
        if args.data or args.smoke_test or args.batch is not None:
            parser.error("--resume uses saved settings; do not combine with --data, --smoke-test or --batch")
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        if args.check_only:
            return
        configure_ultralytics()
        from ultralytics import YOLO

        model = YOLO(str(args.resume.resolve()))
        EpochReporter().attach(model)
        model.train(resume=True, **trainer_options)
        return
    if args.data is None:
        parser.error("--data is required unless --resume is used")
    path = args.data.resolve()
    data = validate_dataset(path)
    if args.single_dog_augmentation and data["names"] not in ({0: "dog"}, ["dog"]):
        parser.error("--single-dog-augmentation is restricted to a one-class dog dataset")
    print(f"Dataset: {path}; breeds: {len(data['names'])}", flush=True)
    counts = {split: len([line for line in (Path(data['path']) / data[split]).read_text(encoding="utf-8").splitlines()
                         if line.strip()]) for split in ("train", "val", "test") if data.get(split)}
    print(f"Dataset images: train={counts['train']}, validation={counts['val']}", flush=True)
    if "test" in counts:
        print(f"Held-out test images: {counts['test']} (not used during training)", flush=True)
    if counts["train"] < 1000:
        print("NOTE: This is a small-subset run, not full-dataset breed training.", flush=True)
    if args.check_only:
        return
    name = args.name or f"yolo12s_{config['imgsz']}"
    if args.smoke_test:
        path = make_smoke_dataset(data, DATASETS / "tsinghua_dogs" / "smoke_subsets" / f"smoke_data_{time.time_ns()}")
        config.update(epochs=1, patience=0, workers=0, close_mosaic=0,
                      save_period=-1, plots=False)
        name = f"smoke_{name}"
    config.update(data=str(path), project=str(args.project.resolve()), name=name, exist_ok=False)
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    configure_ultralytics()
    from ultralytics import YOLO

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        # Bare checkpoint names belong in pretrained; other relative paths are
        # intentionally resolved against the caller's current directory.
        model_path = PRETRAINED / model_path if model_path.parent == Path(".") else model_path.resolve()
    model = YOLO(str(model_path))
    if augmentation_policy is not None:
        def record_policy(trainer):
            (Path(trainer.save_dir) / "augmentation_policy.json").write_text(
                json.dumps(augmentation_policy, indent=2) + "\n", encoding="utf-8")
        model.add_callback("on_train_start", record_policy)
    EpochReporter().attach(model)
    started = time.perf_counter()
    model.train(**config, **trainer_options)
    summary = {"model": str(model_path), "dataset": str(path), "smoke_test": args.smoke_test,
               "elapsed_seconds": time.perf_counter() - started, "config": config,
               "class_names": model.names, "custom_augmentation": augmentation_policy,
               "best_weights": str(model.trainer.best), "last_weights": str(model.trainer.last)}
    if gpu:
        summary["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        summary["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    output = Path(model.trainer.save_dir) / "training_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training summary: {output}", flush=True)


if __name__ == "__main__":
    main()