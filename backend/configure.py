"""Snapshot completed dog/breed checkpoints for CUDA crop-only inference."""
import json
import secrets
import shutil
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config_path = ROOT / "backend" / "server.local.json"
assets = ROOT / "backend" / "assets"
assets.mkdir(exist_ok=True)
run = ROOT / "models/dog_recognition/yolo12s_960_openimages_aug_20261011_150702"
source = run / "weights/best.pt"
state = json.loads((run / "training_state.json").read_text(encoding="utf-8"))
if state["status"] != "completed":
    raise RuntimeError("The selected training run has not completed; refusing to snapshot it")
weights = assets / "dog-openimages-aug-best.pt"
shutil.copy2(source, weights)
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if digest != hashlib.sha256(weights.read_bytes()).hexdigest():
    raise RuntimeError("Serving snapshot checksum mismatch")
breed_run = ROOT / "models/breed_recognition/yolo12s_960_full_701515_20261007_014943"
breed_source = breed_run / "weights/best.pt"
if json.loads((breed_run / "training_state.json").read_text(encoding="utf-8"))["status"] != "completed":
    raise RuntimeError("Breed training is not completed; refusing to snapshot it")
breed_weights = assets / "breed-full-best.pt"
shutil.copy2(breed_source, breed_weights)
breed_digest = hashlib.sha256(breed_source.read_bytes()).hexdigest()
if breed_digest != hashlib.sha256(breed_weights.read_bytes()).hexdigest():
    raise RuntimeError("Breed serving snapshot checksum mismatch")
config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"token": secrets.token_urlsafe(36)}
config.update(weights=str(weights), inference_python=str(ROOT / ".venv/Scripts/python.exe"),
              model_label="GPU: trained dog detector + breed detector on dog crops",
              device="cuda:0", imgsz=960, source_weights=str(source), weights_sha256=digest,
              breed_weights=str(breed_weights), breed_conf=0.25,
              breed_source_weights=str(breed_source), breed_weights_sha256=breed_digest)
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(f"Local config: {config_path}")
print("Token intentionally not printed. Find it locally in server.local.json.")