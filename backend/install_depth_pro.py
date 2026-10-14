"""Install pinned official Depth Pro in an isolated overlay environment on this PC.

CUDA Torch/torchvision are read-only imports from the existing project environment.
Depth Pro's numpy<2 and additional packages live only in .depth-venv. No upgrades
or package writes are made to the serving/training environments.
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9e65e4dbe9568d23c546fcec53302b10445e109e"
WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"


def main():
    environment = ROOT / "backend/.depth-venv"
    source = ROOT / "backend/third_party/ml-depth-pro"
    assets = ROOT / "backend/assets"
    assets.mkdir(parents=True, exist_ok=True)
    if not (environment / "Scripts/python.exe").exists():
        venv.create(environment, with_pip=True)
    site = environment / "Lib/site-packages"
    shared = ROOT / ".venv/Lib/site-packages"
    (site / "bark_cuda_runtime.pth").write_text(str(shared) + "\n", encoding="utf-8")
    if not (source / "src/depth_pro/depth_pro.py").exists():
        archive = assets / "depth-pro-source.zip"
        print("Downloading pinned official source", flush=True)
        urllib.request.urlretrieve(f"https://api.github.com/repos/apple-aiml-research/ml-depth-pro/zipball/{REVISION}", archive)
        source.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            base = zipped.namelist()[0].split("/")[0]
            zipped.extractall(source.parent)
        (source.parent / base).rename(source)
        archive.unlink()
    (site / "bark_depth_pro.pth").write_text(str(source / "src") + "\n", encoding="utf-8")
    python = environment / "Scripts/python.exe"
    subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir",
                    "numpy==1.26.4", "opencv-python==4.11.0.86", "timm==0.9.16", "pillow-heif>=0.18,<2",
                    "huggingface-hub>=0.23,<1", "safetensors>=0.4,<1", "matplotlib>=3.8,<4"], check=True)
    weights = assets / "depth_pro.pt"
    if not weights.exists():
        partial = weights.with_suffix(".pt.partial")
        print("Downloading official Apple checkpoint (large file)", flush=True)
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "BARK-local-model-install"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", 0))
            received, next_notice = 0, 256 * 1024**2
            while chunk := response.read(8 * 1024**2):
                output.write(chunk)
                received += len(chunk)
                if received >= next_notice:
                    print(f"Checkpoint downloaded: {received / 1024**2:.0f} MiB", flush=True)
                    next_notice += 256 * 1024**2
            if expected and received != expected:
                raise RuntimeError("Incomplete Depth Pro checkpoint download")
        partial.rename(weights)
    digest = hashlib.sha256()
    with weights.open("rb") as file:
        while chunk := file.read(8 * 1024**2):
            digest.update(chunk)
    manifest = {"model": "apple_depth_pro", "source_revision": REVISION,
                "weights_url": WEIGHTS_URL, "sha256": digest.hexdigest(),
                "bytes": weights.stat().st_size, "python": str(python),
                "license": str(source / "LICENSE"),
                "checksum_note": "Local integrity fingerprint of HTTPS official download, not upstream signature verification"}
    (assets / "depth-pro-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([str(python), "-c", "import torch,depth_pro,numpy,timm; print('Depth Pro imports OK:',torch.__version__,numpy.__version__,timm.__version__); assert torch.cuda.is_available()"], check=True)
    print("Depth Pro isolated installation ready", flush=True)


if __name__ == "__main__":
    main()