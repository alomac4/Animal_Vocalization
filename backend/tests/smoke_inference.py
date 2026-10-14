"""Live configured GPU inference + H.264/audio integration check using a generated clip."""
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
            health = client.get("/health", headers=headers).json()
            assert health["device"].startswith("cuda:"), health
            assert health["breed_weights_available"] and health["pipeline"] == "dog_boxes_then_breed_crops", health
            expected_upscaler = "realesr-general-x4v3" if config.upscaler_weights else "opencv_lanczos4"
            assert health["imgsz"] == 1536 and health["crop_upscaler"] == expected_upscaler, health
            assert health["breed_batch"] == config.breed_batch and health["half"], health
            assert health["dog_batch"] == config.dog_batch and health["decode_prefetch"], health
            assert health["tracklet_linking"] == "conservative_motion_color", health
            expected_tracking = ("confirmed_motion_color_hungarian" if config.tracker_backend == "motion_color"
                                 else f"confirmed_{config.tracker_backend}_sequential")
            assert health["tracking"] == expected_tracking, health
            assert health["track_min_hits"] == 3 and health["track_buffer_seconds"] == 3, health
            assert health["breed_consensus"] == "clip_majority_non_unknown" and health["render_passes"] == 2, health
            assert health["video_labels"] == ("dog_id_and_estimated_distance" if config.distance_enabled else "dog_id_only"), health
            assert health["tracker_backend"] == config.tracker_backend, health
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
            assert result["performance"]["frames"] == result["frames"], result
            if config.distance_enabled:
                assert result["distance_estimation"]["approximate"], result
                assert result["performance"]["distance_estimated_boxes"] + result["performance"]["distance_unavailable_boxes"] == result["consensus_labeled_count"], result
            else:
                assert result["distance_estimation"] is None, result
                assert result["performance"]["depth_frames"] == 0, result
                assert result["performance"]["depth_seconds"] == 0, result
                assert result["performance"]["depth_load_seconds"] == 0, result
            print(f"Measured pipeline stages: {result['performance']}")
            if config.upscaler_weights:
                assert health["upscaler_weights_available"], health
                assert result["performance"]["neural_upscaled_crops"] > 0, result
                assert result["performance"]["upscaler_host_transfers"] == result["performance"]["neural_upscaled_crops"], result
                assert result["performance"]["upscaler_host_bytes"] > 0, result
            assert result["detection_count"] > 0, result
            assert result["breed_crop_count"] == result["detection_count"], result
            assert 0 <= result["breed_labeled_count"] <= result["breed_crop_count"], result
            assert result["frames"] == result["rendered_frames"] == 5, result
            assert result["tracked_dog_count"] == 1, result
            assert result["breed_labeled_track_count"] == 1, result
            assert len(result["dog_statistics"]) == 1, result
            assert result["dog_statistics"][0]["name"] == "Dog #1", result
            assert result["dog_statistics"][0]["breed"], result
            print(f"Dog statistics: {result['dog_statistics']}")
            assert result["consensus_labeled_count"] == result["detection_count"], result
            print(f"Tracking: {result['tracked_dog_count']} ID, {result['consensus_labeled_count']} consensus-labeled frames")
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
            print(f"PASS on {config.device}: {result['frames']} frames, {result['detection_count']} dog boxes, "
                  f"{result['breed_crop_count']} breed crops, {result['breed_labeled_count']} breed-labeled crops, "
                  "H.264 + AAC, download and deletion.")


if __name__ == "__main__":
    main()