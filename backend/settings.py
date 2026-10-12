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
    imgsz: int = 960
    breed_weights: Path | None = None
    breed_conf: float = 0.25

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
        imgsz = int(config.get("imgsz", 960))
        if imgsz < 32 or imgsz > 2048 or imgsz % 32:
            raise ValueError("Inference image size must be a multiple of 32 between 32 and 2048")
        if not config.get("breed_weights"):
            raise ValueError("Configure the crop-only breed checkpoint with backend/configure.py")
        breed_conf = float(config.get("breed_conf", 0.25))
        if not 0 < breed_conf <= 1:
            raise ValueError("Breed confidence threshold must be greater than zero and at most one")
        return cls(token=token, runtime=ROOT / "backend" / "runtime" / "jobs",
                   weights=Path(config["weights"]),
                   inference_python=Path(config["inference_python"]),
                   model_label=config.get("model_label", cls.model_label), device=device, imgsz=imgsz,
                   breed_weights=Path(config["breed_weights"]), breed_conf=breed_conf)