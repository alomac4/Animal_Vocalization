"""Local settings; secrets never belong in mobile source or a public repository."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    token: str
    runtime: Path
    weights: Path
    inference_python: Path
    max_upload_bytes: int = 250 * 1024 * 1024
    max_jobs: int = 2
    retention_seconds: int = 1800
    process_timeout_seconds: int = 1200
    max_duration_seconds: int = 300
    model_label: str = "Pretrained baseline — fine-tuned model not connected yet"
    device: str = "cuda:0"
    imgsz: int = 1536
    breed_weights: Path | None = None
    breed_conf: float = 0.25
    breed_imgsz: int = 960
    breed_batch: int = 4
    dog_batch: int = 1
    upscaler_batch: int = 8
    track_link_seconds: float = 6.0
    track_reconnect_seconds: float = 15.0
    track_presence_seconds: float = 3.0
    track_presence_ratio: float = .05
    track_clip_presence_ratio: float = .03
    tracker_backend: str = "ocsort"
    distance_enabled: bool = False
    distance_focal_mm: float = 26.0
    distance_dog_extent_m: float = .65
    distance_backend: str = "depth_pro"
    depth_python: Path = ROOT / "backend/.depth-venv/Scripts/python.exe"
    depth_weights: Path = ROOT / "backend/assets/depth_pro.pt"
    crop_upscale: float = 4.0
    half: bool = True
    dog_conf: float = .25
    track_high_conf: float = .5
    track_new_conf: float = .6
    track_min_hits: int = 3
    track_buffer_seconds: float = 3.0
    upscaler_weights: Path | None = None
    upscaler_tile: int = 192

    @classmethod
    def load(cls) -> Settings:
        config_path = Path(os.environ.get("BARK_CONFIG", ROOT / "backend" / "server.local.json"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        token = os.environ.get("BARK_API_TOKEN", config["token"])
        if len(token) < 32:
            raise ValueError("Use an API token with at least 32 characters")
        device = str(config.get("device", "cuda:0"))
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("Inference requires an explicit CUDA device, such as cuda:0")
        imgsz = int(config.get("imgsz", 1536))
        if imgsz < 32 or imgsz > 2048 or imgsz % 32:
            raise ValueError("Inference image size must be a multiple of 32 between 32 and 2048")
        if not config.get("breed_weights"):
            raise ValueError("Configure the crop-only breed checkpoint with backend/configure.py")
        breed_conf = float(config.get("breed_conf", 0.25))
        if not 0 < breed_conf <= 1:
            raise ValueError("Breed confidence threshold must be greater than zero and at most one")
        breed_imgsz = int(config.get("breed_imgsz", 960))
        breed_batch = int(config.get("breed_batch", 4))
        dog_batch = int(config.get("dog_batch", 1))
        upscaler_batch = int(config.get("upscaler_batch", 8))
        track_link_seconds = float(config.get("track_link_seconds", 6))
        track_reconnect_seconds = float(config.get("track_reconnect_seconds", 15))
        track_presence_seconds = float(config.get("track_presence_seconds", 3))
        track_presence_ratio = float(config.get("track_presence_ratio", .05))
        track_clip_presence_ratio = float(config.get("track_clip_presence_ratio", .03))
        tracker_backend = str(config.get("tracker_backend", "ocsort"))
        from .distance import DistancePreset
        distance_enabled = config.get("distance_enabled", False)
        distance_backend = str(config.get("distance_backend", "depth_pro"))
        process_timeout = int(config.get("process_timeout_seconds", 3600 if distance_enabled and distance_backend == "depth_pro" else 1200))
        if not 60 <= process_timeout <= 7200:
            raise ValueError("Process timeout must be 60–7200 seconds")
        if distance_backend not in ("depth_pro", "pinhole"):
            raise ValueError("Unknown distance backend")
        if not isinstance(distance_enabled, bool):
            raise ValueError("distance_enabled must be boolean")
        distance = DistancePreset(float(config.get("distance_focal_mm", 26)),
                                  float(config.get("distance_dog_extent_m", .65)))
        if tracker_backend not in ("motion_color", "ocsort", "botsort", "fasttrack"):
            raise ValueError("Unsupported tracker backend")
        if not (0 <= track_reconnect_seconds <= 30 and 0 <= track_presence_seconds <= 10
                and 0 <= track_presence_ratio <= .1 and 0 <= track_clip_presence_ratio <= .1):
            raise ValueError("Invalid identity recovery/presence settings")
        if not 1 <= dog_batch <= 8 or not 1 <= upscaler_batch <= 8:
            raise ValueError("Dog and upscaler batches must be 1–8")
        if not 0 <= track_link_seconds <= 10:
            raise ValueError("Track linking window must be 0–10 seconds")
        crop_upscale = float(config.get("crop_upscale", 4.0))
        half = config.get("half", True)
        if breed_imgsz < 32 or breed_imgsz > 2048 or breed_imgsz % 32:
            raise ValueError("Breed input size must be a multiple of 32 between 32 and 2048")
        if not 1 <= breed_batch <= 8 or not 1 <= crop_upscale <= 4:
            raise ValueError("Breed batch must be 1–8 and crop upscale limit must be 1–4")
        if not isinstance(half, bool):
            raise ValueError("half must be a boolean")
        dog_conf = float(config.get("dog_conf", .25))
        track_high_conf = float(config.get("track_high_conf", .5))
        track_new_conf = float(config.get("track_new_conf", .6))
        track_min_hits = int(config.get("track_min_hits", 3))
        track_buffer_seconds = float(config.get("track_buffer_seconds", 3))
        upscaler_tile = int(config.get("upscaler_tile", 192))
        if not 64 <= upscaler_tile <= 256:
            raise ValueError("Upscaler tile must be 64–256 pixels")
        if not 0 < dog_conf <= track_high_conf <= track_new_conf <= 1:
            raise ValueError("Require 0 < dog_conf <= track_high_conf <= track_new_conf <= 1")
        if track_min_hits != 3 or not 1 <= track_buffer_seconds <= 10:
            raise ValueError("Use three confirmation hits and a 1–10 second track buffer")
        return cls(token=token, runtime=ROOT / "backend" / "runtime" / "jobs",
                   weights=Path(config["weights"]),
                   inference_python=Path(config["inference_python"]),
                    model_label=config.get("model_label", cls.model_label), device=device, imgsz=imgsz,
                    process_timeout_seconds=process_timeout,
                   breed_weights=Path(config["breed_weights"]), breed_conf=breed_conf,
                   breed_imgsz=breed_imgsz, breed_batch=breed_batch, crop_upscale=crop_upscale, half=half,
                   dog_conf=dog_conf, track_high_conf=track_high_conf, track_new_conf=track_new_conf,
                    track_min_hits=track_min_hits, track_buffer_seconds=track_buffer_seconds,
                    upscaler_weights=Path(config["upscaler_weights"]) if config.get("upscaler_weights") else None,
                     upscaler_tile=upscaler_tile, dog_batch=dog_batch,
                      upscaler_batch=upscaler_batch, track_link_seconds=track_link_seconds,
                      track_reconnect_seconds=track_reconnect_seconds,
                      track_presence_seconds=track_presence_seconds, track_presence_ratio=track_presence_ratio,
                      track_clip_presence_ratio=track_clip_presence_ratio, tracker_backend=tracker_backend,
                      distance_enabled=distance_enabled, distance_focal_mm=distance.focal_equivalent_mm,
                       distance_dog_extent_m=distance.dog_extent_m, distance_backend=distance_backend,
                       depth_python=Path(config.get("depth_python", cls.depth_python)),
                       depth_weights=Path(config.get("depth_weights", cls.depth_weights)))