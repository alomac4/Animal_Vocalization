"""Generate a local API token and snapshot the stable pretrained checkpoint."""
import json
import secrets
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config_path = ROOT / "backend" / "server.local.json"
assets = ROOT / "backend" / "assets"
assets.mkdir(exist_ok=True)
weights = assets / "baseline-yolo12s.pt"
if not weights.exists():
    shutil.copy2(ROOT / "models" / "dog_recognition" / "yolo12s.pt", weights)
if not config_path.exists():
    config_path.write_text(json.dumps({
        "token": secrets.token_urlsafe(36), "weights": str(weights),
        "inference_python": str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "model_label": "Pretrained baseline — fine-tuned model not connected yet"
    }, indent=2), encoding="utf-8")
print(f"Local config: {config_path}")
print("Token intentionally not printed. Find it locally in server.local.json.")