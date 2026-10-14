"""Controlled temporary-video profile: serial decode versus prefetch/batching."""
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from backend.settings import Settings
from backend.worker import process
from backend.super_resolution import NeuralCropUpscaler


def serial_decode(capture, first, requested, timings):
    yield [first]
    while True:
        start = time.perf_counter()
        ok, frame = capture.read()
        timings['decode_seconds'] += time.perf_counter() - start
        if not ok:
            return
        yield [frame]


class SerialUpscaler:
    def __init__(self, *args, **kwargs):
        self.inner = NeuralCropUpscaler(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'enhance_many':
            raise AttributeError(name)
        return getattr(self.inner, name)


def main():
    config = Settings.load()
    ffmpeg = subprocess.check_output([str(ROOT / 'backend/.venv/Scripts/python.exe'), '-c',
        'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True).strip()
    images = sorted(next((ROOT / 'Benchmarks/Dog_recog_model').rglob('frames_easy_gsd_gate')).rglob('f*.jpg'))
    paths = [images[round(i * (len(images) - 1) / 11)] for i in range(12)]
    variants = [('serial_decode_frame1_crop4_tile4', 1, 4, 4, True),
                ('prefetch_frame1_crop4_tile8', 1, 4, 8, False),
                ('prefetch_frame4_crop8_tile8', 4, 8, 8, False)]
    reports = {name: [] for name, *_ in variants}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = cv2.imread(str(paths[0]))
        height, width = first.shape[:2]
        source = root / 'profile.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot create controlled benchmark clip')
        for image in paths:
            frame = cv2.imread(str(image))
            for _ in range(10):
                writer.write(frame)
        writer.release()
        for trial in range(2):
            for name, dog_batch, breed_batch, sr_batch, serial in variants[::1 if trial == 0 else -1]:
                args = SimpleNamespace(input=source, output=root / 'result.mp4', weights=config.weights,
                    breed_weights=config.breed_weights, device=config.device, imgsz=config.imgsz,
                    breed_imgsz=config.breed_imgsz, half=config.half, breed_conf=config.breed_conf,
                    crop_upscale=config.crop_upscale, breed_batch=breed_batch, dog_batch=dog_batch,
                    upscaler_weights=config.upscaler_weights, upscaler_tile=config.upscaler_tile,
                    upscaler_batch=sr_batch, dog_conf=config.dog_conf, track_high_conf=config.track_high_conf,
                    track_new_conf=config.track_new_conf, track_min_hits=config.track_min_hits,
                    track_buffer_seconds=config.track_buffer_seconds, track_link_seconds=config.track_link_seconds,
                    max_duration=config.max_duration_seconds, ffmpeg=ffmpeg)
                events = []
                with patch('backend.worker.emit', side_effect=lambda **event: events.append(event)):
                    if serial:
                        with patch('backend.worker.decoded_batches', serial_decode), \
                                patch('backend.super_resolution.NeuralCropUpscaler', SerialUpscaler):
                            process(args)
                    else:
                        process(args)
                result = next(event['performance'] for event in events if 'performance' in event)
                assert result['frames'] == 120, result
                reports[name].append(result)
                print(json.dumps({'variant': name, 'trial': trial, 'seconds': result['total_seconds'],
                                  'dogs': result['tracked_dog_count']}), flush=True)
    report = {'variants': [{'name': name, 'median_seconds': statistics.median(t['total_seconds'] for t in trials),
                            'trials': trials} for name, trials in reports.items()],
              'notes': '120-frame synthetic clip made from 12 sampled gate stills, repeated 10 times each; two trials in reversed order. Includes model initialization, decode, tracking, final voting, render and mux. Excludes network. Not continuous-video tracking accuracy or exact user clip timing.'}
    (ROOT / 'backend/runtime/prefetch-video-profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()