"""Backend-friendly breed detection output; pet identities are separate metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .project_paths import configure_ultralytics
else:
    from project_paths import configure_ultralytics


def results_to_payload(result) -> dict:
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                "breed": {"class_id": class_id, "label": result.names[class_id],
                          "confidence": round(float(box.conf.item()), 6)},
                "identity": {"dog_id": None, "name": None, "status": "unassigned"},
                "track_id": int(box.id.item()) if box.id is not None else None,
            })
    return {"source": str(result.path), "image_height": result.orig_shape[0],
            "image_width": result.orig_shape[1], "detections": detections}


def resolve_source(source: str, track: bool = False):
    directory = Path(source)
    if not directory.is_dir():
        return source
    if track:
        raise ValueError("--track requires one video/stream, not a directory of unrelated images")
    configure_ultralytics()
    from ultralytics.data.utils import IMG_FORMATS

    if not any(path.is_file() and path.suffix.lower().lstrip(".") in IMG_FORMATS
               for path in directory.rglob("*")):
        raise FileNotFoundError(f"No supported images in {directory}")
    # A glob uses the file-stream loader. A list of strings can eagerly decode
    # every image into RAM before inference in Ultralytics' source dispatcher.
    import glob

    return glob.escape(str(directory.resolve())) + "/**/*.*"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, directory, or video path")
    parser.add_argument("--output", type=Path, required=True, help="JSON Lines output (one result per image/frame)")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=992)
    parser.add_argument("--device", default="0")
    parser.add_argument("--track", action="store_true", help="Use only with frames from a single video/stream")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("Confidence and IoU must be between 0 and 1")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz must be positive and divisible by 32")
    configure_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    options = dict(source=resolve_source(args.source, args.track), conf=args.confidence, iou=args.iou, imgsz=args.imgsz,
                   device=args.device, stream=True, verbose=False, agnostic_nms=True)
    # Breed classes are mutually exclusive per dog: suppress overlapping breed hypotheses.
    results = model.track(**options) if args.track else model.predict(**options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(results):
            payload = results_to_payload(result)
            payload["result_index"] = index
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Saved detections to {args.output.resolve()}")


if __name__ == "__main__":
    main()