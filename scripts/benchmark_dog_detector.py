"""Compare a pretrained generic dog detector with saved breed-model predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

if __package__:
    from .project_paths import ROOT, configure_ultralytics
else:
    from project_paths import ROOT, configure_ultralytics


SCENES = ("easy_gsd_gate", "medium_backyard_pack")
EXAMPLES = {
    "easy_gsd_gate": ("f000000.jpg", "f000596.jpg", "f001790.jpg", "f002408.jpg", "f003005.jpg"),
    "medium_backyard_pack": ("f000000.jpg", "f000426.jpg", "f001327.jpg", "f003412.jpg", "f003870.jpg"),
}


def annotate(image: Image.Image, detections: list[dict], color: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for box in detections:
        xy = box["bbox_xyxy"]
        label = box.get("breed", box)
        text = f"{label['label']} {label['confidence']:.2f}"
        draw.rectangle(xy, outline=color, width=3)
        x, y = xy[0], max(0, xy[1] - 14)
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill="black")
        draw.text((x, y), text, fill=color)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/pretrained/yolo12s.pt")
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "Benchmarks/Dog_recog_model")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/benchmarks/uploaded_folders_review")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/generic_dog_detector_review")
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    previous = json.loads((args.baseline / "summary.json").read_text(encoding="utf-8"))
    options = dict(imgsz=960, conf=0.25, iou=0.5, agnostic_nms=True, batch=8, device="0")
    for key, value in (("imgsz", 960), ("confidence", 0.25), ("iou", 0.5),
                       ("agnostic_nms", True), ("batch", 8)):
        if previous["settings"][key] != value:
            raise ValueError(f"Baseline settings differ: {key}")
    configure_ultralytics()
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matched benchmark")
    model = YOLO(str(args.weights))
    dog_ids = [key for key, name in model.names.items() if name.lower() == "dog"]
    if model.task != "detect" or len(dog_ids) != 1:
        raise ValueError("Checkpoint must be a detector with exactly one generic dog class")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"weights": str(args.weights),
               "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               "ultralytics_version": ultralytics.__version__, "settings": options,
               "dog_class_id": dog_ids[0], "ground_truth_available": False,
               "limitations": "Counts are not recall or accuracy. No tracking, training, or controlled speed test.",
               "scenes": {}}
    for scene in SCENES:
        frames = args.benchmark_root / ("frames_" + scene) / "dog_benchmark" / scene / "frames"
        sources = sorted(frames.glob("*.jpg"))
        baseline_rows = [json.loads(line) for line in
                         (args.baseline / (scene + "_best.jsonl")).read_text(encoding="utf-8").splitlines()]
        baseline = {Path(row["source"]).name: row for row in baseline_rows}
        if len(sources) != 150 or len(baseline) != 150 or {p.name for p in sources} != set(baseline):
            raise ValueError(f"Expected the same 150 unique frame names for {scene}")
        annotated = args.output / "annotated_dog" / scene
        comparisons = args.output / "comparisons" / scene
        annotated.mkdir(parents=True, exist_ok=True)
        comparisons.mkdir(parents=True, exist_ok=True)
        predictions = []
        results = model.predict(source=str(frames), classes=dog_ids, stream=True, verbose=False, **options)
        with (args.output / (scene + "_dog.jsonl")).open("w", encoding="utf-8") as handle:
            for result in results:
                name = Path(result.path).name
                old = baseline[name]
                h, w = result.orig_shape
                if (h, w) != (old["image_height"], old["image_width"]):
                    raise ValueError(f"Baseline image dimensions differ for {name}")
                boxes = [{"bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                          "class_id": int(box.cls.item()), "label": "dog",
                          "confidence": round(float(box.conf.item()), 6)} for box in result.boxes]
                row = {"source": str(Path(result.path).resolve()), "image_height": h,
                       "image_width": w, "detections": boxes}
                predictions.append(row)
                handle.write(json.dumps(row) + "\n")
                with Image.open(result.path) as original:
                    image = original.convert("RGB")
                dog_view = annotate(image, boxes, "lime")
                dog_view.save(annotated / name)
                breed_view = annotate(image, old["detections"], "orange")
                pair = Image.new("RGB", (w * 2, h + 28), "black")
                pair.paste(breed_view, (0, 28))
                pair.paste(dog_view, (w, 28))
                draw = ImageDraw.Draw(pair)
                draw.text((8, 8), f"Breed best: {len(old['detections'])} | {name}", fill="orange")
                draw.text((w + 8, 8), f"Generic dog: {len(boxes)} | {name}", fill="lime")
                pair.save(comparisons / name)
        if len(predictions) != 150 or len({Path(row['source']).name for row in predictions}) != 150:
            raise RuntimeError(f"Incomplete inference for {scene}")
        counts = [len(row["detections"]) for row in predictions]
        comparison = []
        for row in predictions:
            name = Path(row["source"]).name
            comparison.append({"frame": name, "breed_count": len(baseline[name]["detections"]),
                               "dog_count": len(row["detections"])})
        stats = {"frames": len(counts), "frames_with_detections": sum(count > 0 for count in counts),
                 "total_detections": sum(counts), "count_distribution": dict(Counter(counts)),
                 "breed_frames_with_detections": sum(bool(row["detections"]) for row in baseline_rows),
                 "breed_total_detections": sum(len(row["detections"]) for row in baseline_rows),
                 "dog_only_frames": [row["frame"] for row in comparison if row["dog_count"] and not row["breed_count"]],
                 "breed_only_frames": [row["frame"] for row in comparison if row["breed_count"] and not row["dog_count"]],
                 "frame_comparison": comparison}
        summary["scenes"][scene] = stats
        names = list(dict.fromkeys(EXAMPLES[scene] + tuple(sources[i].name for i in (30, 60, 90))))
        sheet = Image.new("RGB", (960, len(names) * 292), "black")
        for index, name in enumerate(names):
            with Image.open(comparisons / name) as pair:
                pair.thumbnail((960, 292))
                sheet.paste(pair, (0, index * 292))
        sheet.save(args.output / (scene + "_comparison_sheet.jpg"))
        print(scene, json.dumps({key: value for key, value in stats.items()
                                 if key not in ("frame_comparison", "dog_only_frames", "breed_only_frames")}))
        print("Dog-only frames:", len(stats["dog_only_frames"]), "Breed-only frames:", len(stats["breed_only_frames"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark to {args.output}")


if __name__ == "__main__":
    main()