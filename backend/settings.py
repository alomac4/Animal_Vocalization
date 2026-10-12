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

    @classmethod
    def load(cls) -> Settings:
        config_path = Path(os.environ.get("BARK_CONFIG", ROOT / "backend" / "server.local.json"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        token = os.environ.get("BARK_API_TOKEN", config["token"])
        if len(token) < 32:
            raise ValueError("Use an API token with at least 32 characters")
        return cls(token=token, runtime=ROOT / "backend" / "runtime" / "jobs",
                   weights=Path(config["weights"]),
                   inference_python=Path(config["inference_python"]),
                   model_label=config.get("model_label", cls.model_label))