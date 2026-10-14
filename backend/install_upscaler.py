"""Download official Real-ESRGAN weights and license without installing new packages."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"


def main():
    assets = ROOT / "assets"
    third_party = ROOT / "third_party"
    assets.mkdir(exist_ok=True)
    third_party.mkdir(exist_ok=True)
    weights = assets / "realesr-general-x4v3.pth"
    manifest_path = assets / "upscaler-manifest.json"
    # Refuse overwriting an existing hash-pinned installation silently.
    if weights.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError("Existing weights differ from their manifest; inspect before reinstalling")
    else:
        temporary = weights.with_suffix(".download")
        try:
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(weights)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    license_url = f"https://raw.githubusercontent.com/xinntao/Real-ESRGAN/{REVISION}/LICENSE"
    (third_party / "Real-ESRGAN-LICENSE.txt").write_bytes(urllib.request.urlopen(license_url, timeout=60).read())
    manifest_path.write_text(json.dumps({"model": "realesr-general-x4v3", "source": URL,
                                        "upstream_revision": REVISION, "sha256": digest,
                                        "license": "BSD-3-Clause", "native_scale": 4}, indent=2), encoding="utf-8")
    print(f"Official upscaler installed: {weights}\nSHA256: {digest}")


if __name__ == "__main__":
    main()