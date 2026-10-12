"""Matched GPU predictions and visual gallery for trained dog/breed detectors."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
from html import escape
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("easy_gsd_gate", "medium_backyard_pack")
DOG = ROOT / "models/dog_recognition/yolo12s_960_openimages_aug_20261011_150702/weights/best.pt"
BREED = ROOT / "models/breed_recognition/yolo12s_960_full_701515_20261007_014943/weights/best.pt"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def annotate(image, detections, color):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    size = max(14, round(image.width / 75))
    face = font(size)
    for detection in detections:
        xy = detection["bbox_xyxy"]
        draw.rectangle(xy, outline=color, width=max(2, round(image.width / 450)))
        text = f"{detection['label']} {detection['confidence']:.2f}"
        width = draw.textbbox((0, 0), text, font=face)[2] + 10
        x = max(0, min(xy[0], image.width - width))
        y = max(0, xy[1] - size - 12)
        draw.rectangle((x, y, x + width, y + size + 10), fill="#20150f")
        draw.text((x + 5, y + 2), text, fill=color, font=face)
    return image


def comparison(image, dog, breed, name):
    width, height = image.size
    pair = Image.new("RGB", (width * 2, height + 62), "#20150f")
    pair.paste(annotate(image, dog, "#79e3a2"), (0, 62))
    pair.paste(annotate(image, breed, "#ffb568"), (width, 62))
    draw = ImageDraw.Draw(pair)
    face = font(max(18, round(width / 60)))
    draw.text((16, 9), f"DOG-ONLY | {len(dog)} boxes | {name}", fill="#79e3a2", font=face)
    draw.text((width + 16, 9), f"BREED DETECTOR | {len(breed)} boxes | {name}", fill="#ffb568", font=face)
    draw.text((16, 36), "Trained best.pt | independent predictions", fill="white", font=font(15))
    draw.text((width + 16, 36), "Trained best.pt | breed labels are predictions, not truth", fill="white", font=font(15))
    return pair


def scene_stats(dog, breed):
    counts = {kind: [len(row["detections"]) for row in rows] for kind, rows in (("dog", dog), ("breed", breed))}
    return {"frames": len(dog), **{kind: {"boxes": sum(values), "frames_with_boxes": sum(bool(n) for n in values),
                "count_distribution": dict(Counter(values))} for kind, values in counts.items()},
            "dog_more_boxes_frames": sum(a > b for a, b in zip(counts["dog"], counts["breed"])),
            "breed_more_boxes_frames": sum(b > a for a, b in zip(counts["dog"], counts["breed"])),
            "same_box_count_frames": sum(a == b for a, b in zip(counts["dog"], counts["breed"]))}


def pick_examples(dog, breed, count=6):
    """Fixed spread plus largest count differences, explicitly not a random sample."""
    chosen = list(dict.fromkeys((0, len(dog) // 2, len(dog) - 1)))
    ranked = sorted(range(len(dog)), key=lambda i: abs(len(dog[i]["detections"]) - len(breed[i]["detections"])), reverse=True)
    for i in ranked:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= count:
            break
    return sorted(chosen)


def write_gallery(output, summary, scenes):
    chunks = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
              "<title>Trained dog vs breed — GPU comparison</title><style>body{margin:0;background:#fff8ec;color:#3b2118;font:16px 'Segoe UI',sans-serif}main{max-width:1600px;margin:auto;padding:32px}h1{margin-top:0}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;border-bottom:1px solid #dfcab3;padding:12px}code{overflow-wrap:anywhere}img{display:block;width:100%;height:auto}details{margin:18px 0;background:#fffcf6;border:1px solid #e8d8c3;border-radius:12px;padding:16px}summary{cursor:pointer;font-weight:600}a{color:#a94418}small{line-height:1.6}figure{margin:20px 0}figcaption{padding:10px} .model{padding:16px;background:#f3e6d4;margin:12px 0;border-radius:12px}</style><main>",
              "<h1>Dog-only vs breed detector</h1><p>Left: trained dog-only model (green). Right: trained breed detector (orange). Both run independently on the same original frames.</p>",
              "<p><strong>Not an accuracy benchmark:</strong> these 300 video frames have no verified boxes or breeds. More boxes may mean recovered dogs, duplicates, or false positives. Adjacent frames are correlated. Detector scores are not calibrated probabilities or directly comparable between models.</p>",
              f"<p>GPU: {escape(summary['gpu'])} · image size {summary['settings']['imgsz']} · confidence {summary['settings']['conf']} · NMS IoU {summary['settings']['iou']} · class-agnostic NMS · batch 1 · FP32. Fresh inference for both checkpoints; no old saved predictions reused.</p>"]
    for kind, model in summary["models"].items():
        chunks.append(f"<div class='model'><strong>{kind.upper()}</strong><br><code>{escape(model['source'])}</code><br><small>SHA256: {model['sha256']}<br>Checkpoint epoch (1-based): {model['checkpoint_epoch_1based']} · training run status: {escape(model['training_status'])} · completed epochs: {model['completed_epochs']}<br>Mean reported inference: {model['mean_inference_ms']:.2f} ms/image (first frame excluded; not a controlled latency test)</small></div>")
    chunks.append("<table><thead><tr><th>Scene</th><th>Frames</th><th>Dog boxes / frames with boxes</th><th>Breed boxes / frames with boxes</th><th>Dog more / equal / breed more</th></tr></thead><tbody>")
    for scene, stats in summary["scenes"].items():
        chunks.append(f"<tr><td><a href='#{scene}'>{scene}</a></td><td>{stats['frames']}</td><td>{stats['dog']['boxes']} / {stats['dog']['frames_with_boxes']}</td><td>{stats['breed']['boxes']} / {stats['breed']['frames_with_boxes']}</td><td>{stats['dog_more_boxes_frames']} / {stats['same_box_count_frames']} / {stats['breed_more_boxes_frames']}</td></tr>")
    chunks.append("</tbody></table><p>Each scene starts with six explicitly selected examples: first, middle, last, and largest box-count differences. Expand any of the 300 frame links below for the full comparison.</p>")
    for scene, rows in scenes.items():
        chunks.append(f"<h2 id='{scene}'>{scene}</h2><a href='{scene}_comparison_sheet.jpg'>Open six-example contact sheet</a>")
        for row in rows:
            chunks.append(f"<details{' open' if row['example'] else ''}><summary>{row['name']} — dog {row['dog_count']} / breed {row['breed_count']}</summary><figure><a href='{row['image']}'><img loading='lazy' src='{row['image']}' alt='Side by side predictions for {scene} {row['name']}'></a><figcaption>Left: dog-only. Right: breed. <a href='{scene}/dog/{row['name']}'>Dog image</a> · <a href='{scene}/breed/{row['name']}'>Breed image</a></figcaption></figure></details>")
    chunks.append("<p>Raw benchmark-only predictions: per-scene predictions_dog.jsonl and predictions_breed.jsonl. Full provenance: summary.json. These saved artifacts are an explicit offline comparison, not persistent app inference history.</p></main></html>")
    (output / "index.html").write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dog-weights", type=Path, default=DOG)
    parser.add_argument("--breed-weights", type=Path, default=BREED)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "Benchmarks/Dog_recog_model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "checkpoints"
    snapshots.mkdir()
    import os
    os.environ["YOLO_CONFIG_DIR"] = str(output / "yolo-settings")
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(2)
    settings = dict(device="cuda:0", imgsz=960, conf=0.25, iou=0.5, agnostic_nms=True,
                    batch=1, half=False, max_det=300, rect=True)
    sources = {scene: sorted((args.benchmark_root / f"frames_{scene}" / "dog_benchmark" / scene / "frames").glob("*.jpg")) for scene in SCENES}
    if any(len(paths) != 150 for paths in sources.values()):
        raise ValueError("Expected exactly 150 benchmark images in each scene")
    summary = {"created_at": datetime.now().astimezone().isoformat(), "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "ultralytics": ultralytics.__version__, "settings": settings,
               "ground_truth_available": False, "models": {}, "scenes": {},
               "example_selection": "First, middle, last, then largest absolute box-count differences; not random.",
               "limitations": "No precision, recall, mAP or breed accuracy without truth. Counts are not unique dogs. Not a controlled timing benchmark."}
    predictions = {}
    for kind, source in (("dog", args.dog_weights.resolve()), ("breed", args.breed_weights.resolve())):
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        snapshot = snapshots / f"{kind}-best.pt"
        snapshot.write_bytes(data)
        if file_hash(source) != digest or file_hash(snapshot) != digest:
            raise RuntimeError("Checkpoint changed while being snapshotted")
        state = json.loads((source.parents[1] / "training_state.json").read_text(encoding="utf-8"))
        model = YOLO(str(snapshot))
        if model.task != "detect" or (kind == "dog" and list(model.names.values()) != ["dog"]) or (kind == "breed" and len(model.names) != 130):
            raise ValueError(f"Unexpected {kind} model class map: {model.names}")
        epoch = model.ckpt.get("epoch", -1)
        predictions[kind] = {}
        speeds = []
        for scene, paths in sources.items():
            rows = []
            for result in model.predict(source=[str(p) for p in paths], stream=True, verbose=False, **settings):
                rows.append({"source": str(Path(result.path).resolve()), "image_sha256": file_hash(result.path),
                             "image_height": result.orig_shape[0], "image_width": result.orig_shape[1],
                             "detections": [{"bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                                              "class_id": int(box.cls.item()), "label": model.names[int(box.cls.item())],
                                              "confidence": round(float(box.conf.item()), 6)} for box in result.boxes],
                             "speed_ms": result.speed})
                speeds.append(result.speed["inference"])
            if len(rows) != len(paths) or [row["source"] for row in rows] != [str(p.resolve()) for p in paths]:
                raise RuntimeError("Incomplete or reordered prediction output")
            predictions[kind][scene] = rows
            folder = output / scene
            folder.mkdir(exist_ok=True)
            (folder / f"predictions_{kind}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            print(f"{kind}: {scene}: {len(rows)} images, {sum(len(r['detections']) for r in rows)} boxes", flush=True)
        summary["models"][kind] = {"source": str(source), "snapshot": str(snapshot), "sha256": digest,
                                    "class_names": model.names, "checkpoint_epoch_1based": epoch + 1 if epoch >= 0 else None,
                                    "training_status": state["status"], "completed_epochs": state["epoch"],
                                    "mean_inference_ms": statistics.mean(speeds[1:])}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    galleries = {}
    for scene in SCENES:
        dog, breed = predictions["dog"][scene], predictions["breed"][scene]
        summary["scenes"][scene] = scene_stats(dog, breed)
        for subfolder in ("dog", "breed", "comparison"):
            (output / scene / subfolder).mkdir()
        examples = pick_examples(dog, breed)
        gallery = []
        tiles = []
        for i, (left, right) in enumerate(zip(dog, breed)):
            if left["image_sha256"] != right["image_sha256"]:
                raise RuntimeError("Source image differs between models")
            name = Path(left["source"]).name
            with Image.open(left["source"]) as original:
                image = original.convert("RGB")
            annotate(image, left["detections"], "#79e3a2").save(output / scene / "dog" / name, quality=95)
            annotate(image, right["detections"], "#ffb568").save(output / scene / "breed" / name, quality=95)
            pair = comparison(image, left["detections"], right["detections"], name)
            pair.save(output / scene / "comparison" / name, quality=95)
            gallery.append({"name": name, "image": f"{scene}/comparison/{name}", "dog_count": len(left["detections"]),
                            "breed_count": len(right["detections"]), "example": i in examples})
            if i in examples:
                tiles.append(ImageOps.pad(pair, (1440, 460), color="#20150f"))
        sheet = Image.new("RGB", (1440, 460 * len(tiles)), "#20150f")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (0, i * 460))
        sheet.save(output / f"{scene}_comparison_sheet.jpg", quality=94)
        galleries[scene] = gallery
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_gallery(output, summary, galleries)
    print(json.dumps(summary["scenes"], indent=2), flush=True)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()