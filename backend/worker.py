"""CPU-only dog detector. Burns boxes into H.264 MP4 and preserves input audio."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def emit(**event):
    print(json.dumps(event), flush=True)


def process(args):
    # Never allocate training GPU memory. Limit CPU competition as well.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
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
        # Re-encode at up to 1280 px for reasonable CPU speed and mobile playback.
        scale = min(1, 1280 / max(height, width))
        target_width = max(2, int(width * scale) // 2 * 2)
        target_height = max(2, int(height * scale) // 2 * 2)
        model = YOLO(str(args.weights))
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
            emit(status="detecting", progress=0 if expected_frames > 0 else None)
            try:
                while ok:
                    if frames >= math.ceil(args.max_duration * fps):
                        raise ValueError("Decoded video exceeds the maximum duration.")
                    if frame.shape[:2] != (height, width):
                        raise ValueError("Videos with changing frame dimensions are unsupported.")
                    frame = cv2.resize(frame, (target_width, target_height))
                    result = model.predict(frame, device="cpu", imgsz=640, conf=0.25,
                                           classes=dog_classes, verbose=False)[0]
                    if result.boxes is not None:
                        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                            x1, y1, x2, y2 = [int(value) for value in box]
                            x1, x2 = max(0, x1), min(target_width - 1, x2)
                            y1, y2 = max(0, y1), min(target_height - 1, y2)
                            color = (29, 76, 185)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"Dog {confidence:.0%}"
                            text_y = max(17, y1 - 6)
                            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 248, 236), 3, cv2.LINE_AA)
                            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, color, 1, cv2.LINE_AA)
                            detections += 1
                    encoder.stdin.write(frame.tobytes())
                    frames += 1
                    if frames == 1 or frames % 5 == 0:
                        emit(status="detecting", frames=frames, detection_count=detections,
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
        emit(status="rendering", progress=None, frames=frames, detection_count=detections)
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
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--max-duration", type=int, default=300)
    args = parser.parse_args()
    try:
        process(args)
    except Exception as exc:
        emit(error=str(exc)[:240])
        args.output.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()