"""Evaluate the selected checkpoint once on the held-out test manifest."""

import argparse
import json
from pathlib import Path

if __package__:
    from .project_paths import configure_ultralytics
    from .train_breeds import validate_dataset
else:
    from project_paths import configure_ultralytics
    from train_breeds import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = validate_dataset(args.data.resolve())
    if not data.get("test"):
        parser.error("A held-out test manifest is required")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    configure_ultralytics()
    from ultralytics import YOLO

    metrics = YOLO(str(args.weights.resolve())).val(
        data=str(args.data.resolve()), split="test", imgsz=960, batch=4, device=0,
        workers=4, project=str(args.output.resolve()), name="test_evaluation", exist_ok=False)
    result = {"split": "test", "weights": str(args.weights.resolve()),
              "data": str(args.data.resolve()), "metrics": {key: float(value) for key, value in metrics.results_dict.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()