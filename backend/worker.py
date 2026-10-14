"""CUDA dog/crop-breed inference, track majority voting, and two-pass H.264 rendering."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .tracking import DogTracker, apply_track_breeds
    from .frame_batches import decoded_batches, bounded_batch_size
    from .distance import DistancePreset, format_distance
else:
    from tracking import DogTracker, apply_track_breeds
    from frame_batches import decoded_batches, bounded_batch_size
    from distance import DistancePreset, format_distance


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


def crop_upscale_size(width, height, target, max_scale):
    """Only enlarge, preserving aspect ratio; avoid huge intermediate crop allocations."""
    scale = min(max_scale, max(1.0, target / max(width, height)))
    return max(2, round(width * scale)), max(2, round(height * scale))


def upscale_dog_crop(crop, target, max_scale):
    """Lanczos interpolation, not neural super-resolution or a segmentation mask."""
    import cv2
    height, width = crop.shape[:2]
    size = crop_upscale_size(width, height, target, max_scale)
    if size == (width, height):
        return crop
    return cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4)


def crop_appearance(crop):
    """Small central HSV descriptor, not learned animal re-identification."""
    import cv2
    height, width = crop.shape[:2]
    inset_x, inset_y = width // 6, height // 6
    center = crop[inset_y:height - inset_y, inset_x:width - inset_x]
    hsv = cv2.cvtColor(cv2.resize(center, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    colors = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    brightness = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel()
    return tuple(float(value) for value in (.8 * colors / max(1, float(colors.sum()))).tolist()) + tuple(
        float(value) for value in (.2 * brightness / max(1, float(brightness.sum()))).tolist())


def infer_dog_regions(frame, dog_model, breed_model, args, dog_classes, timings=None, upscaler=None):
    """Single-frame compatibility entry point; video processing uses bounded batches."""
    return infer_frame_batch([frame], dog_model, breed_model, args, dog_classes, timings, upscaler)[0]


def infer_frame_batch(frames, dog_model, breed_model, args, dog_classes, timings=None, upscaler=None):
    """Batch original frames, then batch crops across frames without losing ownership."""
    if not frames:
        return []
    start = time.perf_counter()
    dog_conf = getattr(args, "dog_conf", .25)
    results = dog_model.predict(frames[0] if len(frames) == 1 else frames,
                               device=args.device, imgsz=args.imgsz, rect=False, conf=dog_conf,
                               iou=0.5, agnostic_nms=True, classes=dog_classes,
                               batch=len(frames), quantize=16 if args.half else 32, verbose=False, save=False)
    if len(results) != len(frames):
        raise RuntimeError("Dog model returned a different number of results than frames")
    detections = [[] for _ in frames]
    if timings is not None:
        timings["dog_seconds"] += time.perf_counter() - start
    valid = []
    for frame_index, (frame, result) in enumerate(zip(frames, results)):
        if result.boxes is None:
            continue
        height, width = frame.shape[:2]
        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            bounds = clip_dog_box(box, width, height)
            if bounds is not None and math.isfinite(confidence) and dog_conf <= confidence <= 1:
                valid.append((frame_index, bounds, float(confidence)))
    if not valid:
        return detections
    # Keep at most breed_batch enhanced crops alive, even in dense scenes.
    for offset in range(0, len(valid), args.breed_batch):
        start = time.perf_counter()
        chunk = valid[offset:offset + args.breed_batch]
        crops = []
        appearances = []
        for frame_index, (x1, y1, x2, y2), _ in chunk:
            crop = frames[frame_index][y1:y2, x1:x2].copy()
            appearances.append(crop_appearance(crop))
            crops.append(crop)
        if upscaler and callable(getattr(upscaler, "enhance_many", None)):
            crops = upscaler.enhance_many(crops, args.breed_imgsz, args.crop_upscale)
        else:
            crops = [upscaler.enhance(crop, args.breed_imgsz, args.crop_upscale) if upscaler
                     else upscale_dog_crop(crop, args.breed_imgsz, args.crop_upscale) for crop in crops]
        if timings is not None:
            timings["upscale_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        results = breed_model.predict(crops, device=args.device, imgsz=args.breed_imgsz, rect=False,
                                      conf=args.breed_conf, iou=0.5, agnostic_nms=True,
                                      batch=len(crops), quantize=16 if args.half else 32, verbose=False, save=False)
        if len(results) != len(crops):
            raise RuntimeError("Breed model returned a different number of results than dog crops")
        for (frame_index, bounds, confidence), crop, breed_result, appearance in zip(chunk, crops, results, appearances):
            crop_height, crop_width = crop.shape[:2]
            breed = select_breed(breed_result, crop_width, crop_height, args.breed_conf)
            detections[frame_index].append({"box": bounds, "confidence": confidence, "breed": breed,
                                           "appearance": appearance})
        if timings is not None:
            timings["breed_seconds"] += time.perf_counter() - start
    return detections


def detection_label(detection):
    """Keep the exported video uncluttered; breed details belong in job metadata."""
    label = f"Dog #{detection['track_id']}" if detection.get("track_id") else "Dog"
    return f"{label} | {format_distance(detection['distance'])}" if "distance" in detection else label


def dog_statistics(winners):
    """One row per final identity, using the exact same display IDs as rendering."""
    rows = {}
    for detection in apply_track_breeds([{"track_id": key} for key in winners], winners):
        dog_id, breed = detection["track_id"], detection["breed"]
        rows[dog_id] = {"dog_id": dog_id, "name": f"Dog #{dog_id}",
                        "breed": breed["name"] if breed else None,
                        "votes": breed["votes"] if breed else 0,
                        "known_votes": breed["known_votes"] if breed else 0,
                        "observed_frames": winners[detection["source_track_id"]].get("observed_frames", 0)}
    return [rows[key] for key in sorted(rows)]


def process(args):
    import cv2
    import torch
    from ultralytics import YOLO

    total_start = time.perf_counter()
    depth_client = None
    distance_backend = getattr(args, "distance_backend", "pinhole")
    distance_preset = (DistancePreset(getattr(args, "distance_focal_mm", 26), getattr(args, "distance_dog_extent_m", .65))
                       if getattr(args, "distance_enabled", False) and distance_backend == "pinhole" else None)
    timings = {"dog_seconds": 0.0, "upscale_seconds": 0.0, "breed_seconds": 0.0,
               "decode_seconds": 0.0, "draw_encode_seconds": 0.0, "tracking_seconds": 0.0,
                "render_decode_seconds": 0.0, "depth_seconds": 0.0, "depth_load_seconds": 0.0}
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU inference is required. Check NVIDIA drivers and the inference Python environment.")
    gpu_index = int(args.device.split(":", 1)[1])
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device {args.device} is unavailable.")
    emit(device=args.device, gpu=torch.cuda.get_device_name(gpu_index), imgsz=args.imgsz,
         breed_imgsz=args.breed_imgsz, half=args.half,
         crop_upscaler="realesr-general-x4v3" if getattr(args, "upscaler_weights", None) else "opencv_lanczos4",
          crop_max_scale=args.crop_upscale, breed_batch=args.breed_batch,
          dog_batch=getattr(args, "dog_batch", 4), upscaler_batch=getattr(args, "upscaler_batch", 8))
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError("Cannot decode this video. Try a standard MP4 or MOV file.")
    encoder = None
    silent = args.output.with_name("silent.mp4")
    observations = args.output.with_name("track-observations.jsonl")
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
        upscaler = None
        if getattr(args, "upscaler_weights", None):
            if __package__:
                from .super_resolution import NeuralCropUpscaler
            else:
                from super_resolution import NeuralCropUpscaler
            upscaler = NeuralCropUpscaler(args.upscaler_weights, args.device, args.half,
                                         getattr(args, "upscaler_tile", 192),
                                         getattr(args, "upscaler_batch", 8))
        if model.task != "detect" or breed_model.task != "detect":
            raise ValueError("Both checkpoints must be object detection models.")
        if len(breed_model.names) < 2 or all(name.lower() == "dog" for name in breed_model.names.values()):
            raise ValueError("Configured breed checkpoint has no breed classes.")
        dog_classes = [int(key) for key, name in model.names.items() if name.lower() == "dog"]
        if not dog_classes:
            raise ValueError("Configured weights must be a dog detector, not a breed detector.")
        # First pass: infer every frame and record boxes/IDs, never decoded frames.
        # This spool is job-local and deleted on success, failure or cancellation.
        tracker_backend = getattr(args, "tracker_backend", "motion_color")
        if tracker_backend != "motion_color":
            if __package__:
                from .temporal_tracking import TemporalDogTracker
            else:
                from temporal_tracking import TemporalDogTracker
            tracker = TemporalDogTracker(fps, tracker_backend,
                                         min_hits=getattr(args, "track_min_hits", 3),
                                         high_conf=getattr(args, "track_high_conf", .5),
                                         new_conf=getattr(args, "track_new_conf", .6),
                                         low_conf=getattr(args, "dog_conf", .25),
                                         buffer_seconds=getattr(args, "track_reconnect_seconds", 15))
        else:
            tracker = DogTracker(max_gap=max(1, round(fps * getattr(args, "track_buffer_seconds", 3))),
                             minimum_iou=.15, new_conf=getattr(args, "track_new_conf", .6),
                             high_conf=getattr(args, "track_high_conf", .5),
                             low_conf=getattr(args, "dog_conf", .25),
                             min_hits=getattr(args, "track_min_hits", 3),
                              confirmation_window=max(3, round(fps)), tentative_gap=max(1, round(fps * .5)),
                              reconnect_gap=max(0, round(fps * getattr(args, "track_reconnect_seconds", 15))))
        frames = detections = breed_labeled = breed_crops = 0
        effective_batch = bounded_batch_size(frame, getattr(args, "dog_batch", 1))
        emit(status="detecting", progress=0 if expected_frames > 0 else None)
        with observations.open("w", encoding="utf-8") as spool:
            batches = decoded_batches(capture, frame, effective_batch, timings)
            try:
                for batch in batches:
                    if frames + len(batch) > math.ceil(args.max_duration * fps):
                        raise ValueError("Decoded video exceeds the maximum duration.")
                    batch_regions = infer_frame_batch(batch, model, breed_model, args, dog_classes, timings, upscaler)
                    for frame_index, regions in enumerate(batch_regions):
                        tracking_start = time.perf_counter()
                        tracked_regions = (tracker.update(regions, frames) if tracker_backend == "motion_color"
                                           else tracker.update(regions, frames, batch[frame_index]))
                        timings["tracking_seconds"] += time.perf_counter() - tracking_start
                        detections += len(regions)
                        breed_crops += len(regions)
                        breed_labeled += sum(region["breed"] is not None for region in regions)
                        spool.write(json.dumps({"frame": frames, "detections": [
                            {"box": region["box"], "confidence": region["confidence"],
                             "track_id": region["track_id"]} for region in tracked_regions]}) + "\n")
                        frames += 1
                        if frames == 1 or frames % 5 == 0:
                            emit(status="detecting", frames=frames, detection_count=detections,
                                 breed_crop_count=breed_crops, breed_labeled_count=breed_labeled,
                                 progress=min(0.99, frames / expected_frames) if expected_frames > 0 else None)
            finally:
                batches.close()
        # Require enough actual detections, not merely a long first-to-last span.
        # Short videos are capped at their decoded length so smoke/short clips work.
        minimum_presence = min(frames, max(math.ceil(fps * getattr(args, "track_presence_seconds", 3)),
                                          math.ceil(frames * getattr(args, "track_clip_presence_ratio", .03))))
        winners = tracker.final_consensus(
            max_link_gap=max(0, round(fps * getattr(args, "track_reconnect_seconds", 15))),
            min_observations=minimum_presence,
            min_relative_presence=getattr(args, "track_presence_ratio", .05))
        identities = {entry["identity_id"] for entry in winners.values()}
        tracked_count = len(identities)
        labeled_tracks = len({entry["identity_id"] for entry in winners.values() if entry["breed"] is not None})
        sr_metrics = {name: getattr(upscaler, name, 0) for name in
                      ("neural_crops", "skipped_crops", "tiles", "model_calls", "host_transfers", "host_pixels")}
        if getattr(args, "distance_enabled", False) and distance_backend == "depth_pro":
            # Rendering no longer needs detection/breed/SR weights. Release their
            # GPU tensors before loading the large depth model in its own process.
            import gc
            del model, breed_model, upscaler
            upscaler = None
            gc.collect()
            torch.cuda.empty_cache()
            if __package__:
                from .depth_distance import DepthProClient
            else:
                from depth_distance import DepthProClient
            start = time.perf_counter()
            depth_client = DepthProClient(args.depth_python, args.depth_weights, args.device)
            timings["depth_load_seconds"] = time.perf_counter() - start
        emit(status="rendering", progress=None, frames=frames, detection_count=detections,
             breed_crop_count=breed_crops, breed_labeled_count=breed_labeled,
             tracked_dog_count=tracked_count, breed_labeled_track_count=labeled_tracks,
             dog_statistics=dog_statistics(winners),
              distance_estimation=(depth_client.metadata() if depth_client else
                                   distance_preset.metadata() if distance_preset else None))
        # Restart decode instead of retaining frames in memory; final votes apply
        # to early/unknown observations as well as later known ones.
        capture.release()
        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            raise ValueError("Cannot reopen the input video for rendering.")
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{target_width}x{target_height}", "-r", str(fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(silent)]
        with args.output.with_name("encoding.log").open("wb") as error_log, observations.open(encoding="utf-8") as spool:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
            rendered_frames = 0
            consensus_labeled = 0
            distance_available = distance_unavailable = 0
            try:
                for line in spool:
                    record = json.loads(line)
                    if record["frame"] != rendered_frames:
                        raise RuntimeError("Tracking records are out of frame order.")
                    decode_start = time.perf_counter()
                    ok, frame = capture.read()
                    timings["render_decode_seconds"] += time.perf_counter() - decode_start
                    if not ok:
                        raise ValueError("Video decode ended before all tracked frames were rendered.")
                    if frame.shape[:2] != (height, width):
                        raise ValueError("Videos with changing frame dimensions are unsupported.")
                    regions = apply_track_breeds(record["detections"], winners)
                    if depth_client and regions:
                        start = time.perf_counter()
                        distances = depth_client.estimate(frame, regions)
                        timings["depth_seconds"] += time.perf_counter() - start
                        for detection, distance in zip(regions, distances):
                            detection["distance"] = distance
                    draw_start = time.perf_counter()
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    for detection in regions:
                        if distance_preset:
                            detection["distance"] = distance_preset.estimate(detection["box"], width, height)
                        if "distance" in detection:
                            distance_available += int(detection["distance"] is not None)
                            distance_unavailable += int(detection["distance"] is None)
                        bx1, by1, bx2, by2 = detection["box"]
                        x1, y1 = int(bx1 * target_width / width), int(by1 * target_height / height)
                        x2 = min(target_width - 1, math.ceil(bx2 * target_width / width))
                        y2 = min(target_height - 1, math.ceil(by2 * target_height / height))
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
                        consensus_labeled += int(detection["breed"] is not None)
                    if distance_preset or depth_client:
                        notice = (depth_client or distance_preset).notice()
                        notice_scale = min(.45, max(.15, (target_width - 8) / max(1, cv2.getTextSize(
                            notice, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0])))
                        cv2.putText(frame, notice, (4, target_height-6), cv2.FONT_HERSHEY_SIMPLEX,
                                    notice_scale, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(frame, notice, (4, target_height-6), cv2.FONT_HERSHEY_SIMPLEX,
                                    notice_scale, (255,248,236), 1, cv2.LINE_AA)
                    encoder.stdin.write(frame.tobytes())
                    timings["draw_encode_seconds"] += time.perf_counter() - draw_start
                    rendered_frames += 1
                    if rendered_frames == 1 or rendered_frames % 5 == 0:
                        emit(status="rendering", progress=min(0.99, rendered_frames / frames),
                             rendered_frames=rendered_frames)
            finally:
                encoder.stdin.close()
                try:
                    encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
            if encoder.returncode:
                raise RuntimeError("Video encoding failed.")
            if rendered_frames != frames:
                raise RuntimeError("The final video is missing tracked frames.")
        emit(status="rendering", progress=None, frames=frames, detection_count=detections,
             breed_crop_count=breed_crops, breed_labeled_count=breed_labeled,
             consensus_labeled_count=consensus_labeled, rendered_frames=rendered_frames)
        # Take optional audio from original input; never overlay boxes in the client.
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(silent), "-i", str(args.input), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "copy", "-c:a", "aac", "-threads", "2", "-t", str(frames / fps),
                        "-movflags", "+faststart", str(args.output)], check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=sys.stderr)
        emit(performance={**{key: round(value, 4) for key, value in timings.items()},
                          "total_seconds": round(time.perf_counter() - total_start, 4),
                          "video_seconds": round(frames / fps, 4), "frames": frames,
                          "tracked_dog_count": tracked_count, "breed_labeled_track_count": labeled_tracks,
                           "confirmed_tracklet_count": len(winners),
                           "linked_tracklet_count": len(winners) - tracked_count,
                           "dog_batch": effective_batch, "breed_batch": args.breed_batch,
                           "upscaler_batch": getattr(args, "upscaler_batch", 8),
                           "candidate_track_count": tracker.next_id - 1,
                            "tracker_backend": tracker_backend,
                             "distance_estimated_boxes": distance_available,
                             "distance_unavailable_boxes": distance_unavailable,
                             "depth_frames": depth_client.frames if depth_client else 0,
                              "depth_peak_allocated_mib": depth_client.peak_allocated_mib if depth_client else 0,
                            "recovered_track_count": tracker.recovered_tracks,
                            "presence_rejected_identities": tracker.presence_rejected_identities,
                            "presence_rejected_tracklets": tracker.presence_rejected_tracklets,
                            "presence_minimum_frames": minimum_presence,
                            "discarded_candidate_track_count": tracker.next_id - 1 - len(winners),
                           "consensus_labeled_count": consensus_labeled,
                           "neural_upscaled_crops": sr_metrics["neural_crops"],
                           "upscaler_bypassed_crops": sr_metrics["skipped_crops"],
                           "upscaler_tiles": sr_metrics["tiles"],
                            "upscaler_model_calls": sr_metrics["model_calls"],
                            "upscaler_host_transfers": sr_metrics["host_transfers"],
                            "upscaler_host_bytes": sr_metrics["host_pixels"] * 3})
    finally:
        if depth_client:
            depth_client.close()
        capture.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        silent.unlink(missing_ok=True)
        observations.unlink(missing_ok=True)


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
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--breed-imgsz", type=int, default=960)
    parser.add_argument("--breed-batch", type=int, default=4)
    parser.add_argument("--dog-batch", type=int, default=1)
    parser.add_argument("--crop-upscale", type=float, default=4.0)
    parser.add_argument("--upscaler-weights", type=Path)
    parser.add_argument("--upscaler-tile", type=int, default=192)
    parser.add_argument("--upscaler-batch", type=int, default=8)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dog-conf", type=float, default=.25)
    parser.add_argument("--track-high-conf", type=float, default=.5)
    parser.add_argument("--track-new-conf", type=float, default=.6)
    parser.add_argument("--track-min-hits", type=int, default=3)
    parser.add_argument("--track-buffer-seconds", type=float, default=3)
    parser.add_argument("--track-link-seconds", type=float, default=6)
    parser.add_argument("--track-reconnect-seconds", type=float, default=15)
    parser.add_argument("--track-presence-seconds", type=float, default=3)
    parser.add_argument("--track-presence-ratio", type=float, default=.05)
    parser.add_argument("--track-clip-presence-ratio", type=float, default=.03)
    parser.add_argument("--tracker-backend", choices=("motion_color", "ocsort", "botsort", "fasttrack"), default="ocsort")
    parser.add_argument("--distance-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--distance-focal-mm", type=float, default=26)
    parser.add_argument("--distance-dog-extent-m", type=float, default=.65)
    parser.add_argument("--distance-backend", choices=("depth_pro", "pinhole"), default="depth_pro")
    parser.add_argument("--depth-python", type=Path)
    parser.add_argument("--depth-weights", type=Path)
    args = parser.parse_args()
    try:
        if any(size < 32 or size > 2048 or size % 32 for size in (args.imgsz, args.breed_imgsz)):
            raise ValueError("Model input sizes must be multiples of 32 between 32 and 2048")
        if not 1 <= args.breed_batch <= 8 or not 1 <= args.crop_upscale <= 4:
            raise ValueError("Breed batch must be 1–8 and crop upscale limit must be 1–4")
        if not 64 <= args.upscaler_tile <= 256:
            raise ValueError("Upscaler tile must be 64–256 pixels")
        if not 1 <= args.dog_batch <= 8 or not 1 <= args.upscaler_batch <= 8:
            raise ValueError("Dog and upscaler batches must be 1–8")
        if not 0 <= args.track_link_seconds <= 10:
            raise ValueError("Track linking window must be 0–10 seconds")
        if not 0 <= args.track_reconnect_seconds <= 30 or not 0 <= args.track_presence_seconds <= 10 or not 0 <= args.track_presence_ratio <= .1:
            raise ValueError("Invalid identity recovery/presence settings")
        if not 0 <= args.track_clip_presence_ratio <= .1:
            raise ValueError("Clip presence ratio must be 0–0.1")
        if not 0 < args.dog_conf <= args.track_high_conf <= args.track_new_conf <= 1:
            raise ValueError("Tracking requires ordered low/high/new confidence thresholds")
        if args.track_min_hits != 3 or not 1 <= args.track_buffer_seconds <= 10:
            raise ValueError("Use three confirmation hits and a 1–10 second track buffer")
        process(args)
    except Exception as exc:
        emit(error=str(exc)[:240])
        args.output.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()