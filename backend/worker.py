"""CUDA dog detection followed by crop-only breed detection; rendered H.264 MP4."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def emit(**event):
    print(json.dumps(event), flush=True)


def clip_dog_box(box, width, height):
    """Return valid half-open pixel bounds, or None for a degenerate detection."""
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        return None
    x1, y1 = max(0, min(width, math.floor(box[0]))), max(0, min(height, math.floor(box[1])))
    x2, y2 = max(0, min(width, math.ceil(box[2]))), max(0, min(height, math.ceil(box[3])))
    return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def select_breed(result, width, height, threshold):
    """One conservative label per crop; never substitute breed boxes for dog boxes."""
    if result.boxes is None:
        return None
    candidates = []
    for box, score, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                    result.boxes.cls.tolist()):
        bounds = clip_dog_box(box, width, height)
        if bounds is None or not math.isfinite(score) or score < threshold or not math.isfinite(class_id):
            continue
        x1, y1, x2, y2 = bounds
        # Reject tiny background dogs and detections not containing the crop center.
        if not (x1 <= width / 2 <= x2 and y1 <= height / 2 <= y2):
            continue
        if (x2 - x1) * (y2 - y1) < 0.20 * width * height:
            continue
        name = result.names.get(int(class_id))
        if name and str(name).lower() != "dog":
            candidates.append((float(score), str(name).replace("_", " ")))
    if not candidates:
        return None
    score, name = max(candidates, key=lambda item: item[0])
    return {"name": name, "confidence": score}


def infer_dog_regions(frame, dog_model, breed_model, args, dog_classes):
    """Infer all labels before drawing, so no annotation pixels reach either model."""
    result = dog_model.predict(frame, device=args.device, imgsz=args.imgsz, conf=0.25,
                               iou=0.5, agnostic_nms=True, classes=dog_classes,
                               batch=1, half=False, verbose=False, save=False)[0]
    detections = []
    if result.boxes is None:
        return detections
    height, width = frame.shape[:2]
    for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        bounds = clip_dog_box(box, width, height)
        if bounds is None or not math.isfinite(confidence):
            continue
        x1, y1, x2, y2 = bounds
        # Exactly the detected region, copied from the clean original frame.
        crop = frame[y1:y2, x1:x2].copy()
        breed_result = breed_model.predict(crop, device=args.device, imgsz=args.imgsz,
                                           conf=args.breed_conf, iou=0.5, agnostic_nms=True,
                                           batch=1, half=False, verbose=False, save=False)[0]
        breed = select_breed(breed_result, x2 - x1, y2 - y1, args.breed_conf)
        detections.append({"box": bounds, "confidence": float(confidence), "breed": breed})
    return detections


def detection_label(detection):
    breed = detection["breed"]
    if breed:
        return f"{breed['name']} {breed['confidence']:.0%}"
    return f"Dog {detection['confidence']:.0%} | breed unknown"


def process(args):
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU inference is required. Check NVIDIA drivers and the inference Python environment.")
    gpu_index = int(args.device.split(":", 1)[1])
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device {args.device} is unavailable.")
    emit(device=args.device, gpu=torch.cuda.get_device_name(gpu_index))
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError("Cannot decode this video. Try a standard MP4 or MOV file.")
    encoder = None
    silent = args.output.with_name("silent.mp4")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps < 1 or fps > 120:
            raise ValueError("Unsupported frame rate. Use a video between 1 and 120 fps.")
        if expected_frames > args.max_duration * fps + 1:
            raise ValueError(f"Video is too long. Maximum duration is {args.max_duration // 60} minutes.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("This video has no decodable frames.")
        height, width = frame.shape[:2]
        if max(height, width) > 4096 or min(height, width) < 16:
            raise ValueError("Use a video up to 4096 pixels per side.")
        # Limit output size for encoding cost and mobile playback, not model device.
        scale = min(1, 1280 / max(height, width))
        target_width = max(2, int(width * scale) // 2 * 2)
        target_height = max(2, int(height * scale) // 2 * 2)
        model = YOLO(str(args.weights))
        breed_model = YOLO(str(args.breed_weights))
        if model.task != "detect" or breed_model.task != "detect":
            raise ValueError("Both checkpoints must be object detection models.")
        if len(breed_model.names) < 2 or all(name.lower() == "dog" for name in breed_model.names.values()):
            raise ValueError("Configured breed checkpoint has no breed classes.")
        dog_classes = [int(key) for key, name in model.names.items() if name.lower() == "dog"]
        if not dog_classes:
            raise ValueError("Configured weights must be a dog detector, not a breed detector.")
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{target_width}x{target_height}", "-r", str(fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(silent)]
        with args.output.with_name("encoding.log").open("wb") as error_log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
            frames = 0
            detections = 0
            breed_labeled = 0
            breed_crops = 0
            emit(status="detecting", progress=0 if expected_frames > 0 else None)
            try:
                while ok:
                    if frames >= math.ceil(args.max_duration * fps):
                        raise ValueError("Decoded video exceeds the maximum duration.")
                    if frame.shape[:2] != (height, width):
                        raise ValueError("Videos with changing frame dimensions are unsupported.")
                    frame = cv2.resize(frame, (target_width, target_height))
                    regions = infer_dog_regions(frame, model, breed_model, args, dog_classes)
                    for detection in regions:
                        x1, y1, x2, y2 = detection["box"]
                        color = (29, 76, 185)
                        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2)
                        label = detection_label(detection)
                        font_scale = 0.5
                        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
                        if text_width > target_width - 8:
                            font_scale *= (target_width - 8) / text_width
                            text_width = target_width - 8
                        text_x = max(0, min(x1, target_width - text_width - 4))
                        text_y = min(target_height - 4, max(17, y1 - 6))
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (255, 248, 236), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, color, 1, cv2.LINE_AA)
                        detections += 1
                        breed_crops += 1
                        breed_labeled += int(detection["breed"] is not None)
                    encoder.stdin.write(frame.tobytes())
                    frames += 1
                    if frames == 1 or frames % 5 == 0:
                        emit(status="detecting", frames=frames, detection_count=detections,
                             breed_crop_count=breed_crops, breed_labeled_count=breed_labeled,
                             progress=min(0.99, frames / expected_frames) if expected_frames > 0 else None)
                    ok, frame = capture.read()
            finally:
                encoder.stdin.close()
                try:
                    encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
            if encoder.returncode:
                raise RuntimeError("Video encoding failed.")
        emit(status="rendering", progress=None, frames=frames, detection_count=detections,
             breed_crop_count=breed_crops, breed_labeled_count=breed_labeled)
        # Take optional audio from original input; never overlay boxes in the client.
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(silent), "-i", str(args.input), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "copy", "-c:a", "aac", "-threads", "2", "-t", str(frames / fps),
                        "-movflags", "+faststart", str(args.output)], check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=sys.stderr)
    finally:
        capture.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        silent.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--breed-weights", type=Path, required=True)
    parser.add_argument("--breed-conf", type=float, default=0.25)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--max-duration", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    try:
        process(args)
    except Exception as exc:
        emit(error=str(exc)[:240])
        args.output.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()