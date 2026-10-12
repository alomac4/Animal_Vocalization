"""Live baseline inference + H.264/audio integration check using a generated clip."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import imageio_ffmpeg
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.settings import Settings


def main():
    settings = Settings.load()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # Official PyTorch Hub dog example. Temporary test content only.
        image = root / "dog.jpg"
        response = httpx.get("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", follow_redirects=True, timeout=30)
        response.raise_for_status()
        image.write_bytes(response.content)
        source = root / "source.mp4"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(image),
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "1",
                        "-vf", "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2",
                        "-r", "5", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True)
        from dataclasses import replace
        config = replace(settings, runtime=root / "jobs")
        headers = {"Authorization": f"Bearer {config.token}", "Content-Type": "application/octet-stream"}
        with TestClient(create_app(config)) as client:
            response = client.post("/jobs", content=source.read_bytes(), headers=headers)
            assert response.status_code == 202, response.text
            job_id = response.json()["id"]
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                result = client.get(f"/jobs/{job_id}", headers=headers).json()
                if result["status"] in ("ready", "failed"):
                    break
                time.sleep(0.3)
            assert result["status"] == "ready", result
            assert result["detection_count"] > 0, result
            video = client.get(f"/jobs/{job_id}/video", headers=headers)
            assert video.status_code == 200
            output = root / "output.mp4"
            output.write_bytes(video.content)
            probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(output), "-f", "null", "-"],
                                   capture_output=True, text=True, check=True)
            assert "Video: h264" in probe.stderr, probe.stderr
            assert "Audio: aac" in probe.stderr, probe.stderr
            assert "Duration: 00:00:01." in probe.stderr, probe.stderr
            # Compare decoded frames: boxes really change pixels in the output.
            code = "import cv2,sys,numpy as np; a=cv2.VideoCapture(sys.argv[1]); b=cv2.VideoCapture(sys.argv[2]); ok,x=a.read(); ok2,y=b.read(); assert ok and ok2; y=cv2.resize(y,(x.shape[1],x.shape[0])); assert np.mean(np.abs(x.astype(float)-y.astype(float)))>0.1; print('Rendered frame pixels differ: boxes burned in'); a.release(); b.release()"
            subprocess.run([str(config.inference_python), "-c", code, str(source), str(output)], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            assert client.delete(f"/jobs/{job_id}", headers=headers).status_code == 204
            assert client.get(f"/jobs/{job_id}", headers=headers).status_code == 404
            print(f"PASS: {result['frames']} frames, {result['detection_count']} box detections, H.264 + AAC, download and deletion.")


if __name__ == "__main__":
    main()