"""Workspace-relative locations, independent of the caller's working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
PRETRAINED = MODELS / "pretrained"
LOGS = ROOT / "logs"
OUTPUTS = ROOT / "outputs"
_configured = False


def configure_ultralytics() -> None:
    """Keep downloads/settings in this workspace without changing global settings."""
    global _configured
    if _configured:
        return
    for folder in (DATASETS, PRETRAINED, LOGS / "ultralytics", OUTPUTS):
        folder.mkdir(parents=True, exist_ok=True)
    # Set before importing Ultralytics to isolate its persistent settings.
    os.environ["YOLO_CONFIG_DIR"] = str(LOGS / "ultralytics")
    from ultralytics import settings
    import ultralytics.utils as utils

    settings.update({"datasets_dir": str(DATASETS), "weights_dir": str(PRETRAINED),
                     "runs_dir": str(MODELS / "breed_detection" / "runs")})
    # The installed version caches these locations at initial import. Updating
    # them also directs the AMP-check checkpoint to models/pretrained.
    utils.DATASETS_DIR = DATASETS
    utils.WEIGHTS_DIR = PRETRAINED
    utils.RUNS_DIR = MODELS / "breed_detection" / "runs"
    _configured = True