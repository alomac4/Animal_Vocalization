"""API tests with a deterministic renderer; no training or model loading."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import LeasedVideoResponse, create_app
from backend.settings import Settings


def fake_renderer(job, update):
    update(job, status="detecting", progress=0.5, frames=1, detection_count=1,
           breed_crop_count=1, breed_labeled_count=1)
    (job.directory / "result.mp4").write_bytes(b"fake rendered video")
    (job.directory / "yolo-settings").mkdir()
    (job.directory / "track-observations.jsonl").write_text("temporary boxes and IDs")
    update(job, status="rendering", progress=None, tracked_dog_count=1,
           breed_labeled_track_count=1, consensus_labeled_count=1, rendered_frames=1,
           dog_statistics=[{"dog_id": 1, "name": "Dog #1", "breed": "beagle", "votes": 1, "known_votes": 1}],
           distance_estimation={"approximate": True, "focal_equivalent_mm": 26, "dog_extent_m": .65})


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        folder = Path(self.temp.name)
        self.settings = Settings(token="a" * 48, runtime=folder / "jobs",
                                 weights=folder / "weights.pt", inference_python=Path("python"))
        self.settings.weights.write_bytes(b"model")
        self.client = TestClient(create_app(self.settings, fake_renderer))
        self.client.__enter__()
        self.headers = {"Authorization": f"Bearer {self.settings.token}", "Content-Type": "application/octet-stream"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertEqual(list(self.settings.runtime.iterdir()), [])
        self.temp.cleanup()

    def ready_job(self):
        response = self.client.post("/jobs", content=b"input", headers=self.headers)
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        for _ in range(100):
            result = self.client.get(f"/jobs/{job_id}", headers=self.headers)
            if result.json()["status"] == "ready":
                return job_id
            time.sleep(0.01)
        self.fail("Job never finished")

    def test_authentication_required_everywhere(self):
        for method, path in [("GET", "/health"), ("POST", "/jobs"), ("GET", "/jobs/id"),
                             ("GET", "/jobs/id/video"), ("DELETE", "/jobs/id")]:
            self.assertEqual(self.client.request(method, path).status_code, 401)
        self.assertEqual(self.client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_health_reports_configured_gpu(self):
        response = self.client.get("/health", headers=self.headers)
        self.assertEqual(response.json()["device"], "cuda:0")
        self.assertEqual(response.json()["imgsz"], 1536)
        self.assertEqual(response.json()["breed_imgsz"], 960)
        self.assertEqual(response.json()["breed_batch"], self.settings.breed_batch)
        self.assertTrue(response.json()["half"])
        self.assertEqual(response.json()["crop_upscaler"], "opencv_lanczos4")
        self.assertEqual(response.json()["pipeline"], "dog_boxes_then_breed_crops")
        self.assertEqual(response.json()["tracking"], "confirmed_ocsort_sequential")
        self.assertEqual(response.json()["track_buffer_seconds"], 3)
        self.assertEqual(response.json()["track_min_hits"], 3)
        self.assertEqual(response.json()["track_new_conf"], .6)
        self.assertEqual(response.json()["breed_consensus"], "clip_majority_non_unknown")
        self.assertEqual(response.json()["render_passes"], 2)
        self.assertEqual(response.json()["video_labels"], "dog_id_only")
        self.assertIsNone(response.json()["distance_estimation"])
        self.assertEqual(response.json()["distance_backend"], "depth_pro")
        self.assertEqual(response.json()["identity_recovery"], "appearance_gallery_mutual_best_v2")
        self.assertEqual(response.json()["track_reconnect_seconds"], 15)
        self.assertEqual(response.json()["track_clip_presence_ratio"], .03)
        self.assertFalse(response.json()["breed_weights_available"])

    def test_full_lifecycle_and_range_playback(self):
        job_id = self.ready_job()
        metadata = self.client.get(f"/jobs/{job_id}", headers=self.headers).json()
        self.assertEqual(metadata["breed_crop_count"], 1)
        self.assertEqual(metadata["breed_labeled_count"], 1)
        self.assertEqual(metadata["tracked_dog_count"], 1)
        self.assertEqual(metadata["breed_labeled_track_count"], 1)
        self.assertEqual(metadata["consensus_labeled_count"], 1)
        self.assertEqual(metadata["rendered_frames"], 1)
        self.assertTrue(metadata["distance_estimation"]["approximate"])
        self.assertEqual(metadata["dog_statistics"], [{"dog_id": 1, "name": "Dog #1", "breed": "beagle", "votes": 1, "known_votes": 1}])
        result = self.client.get(f"/jobs/{job_id}/video", headers=self.headers)
        self.assertEqual(result.content, b"fake rendered video")
        self.assertEqual(result.headers["cache-control"], "no-store")
        result = self.client.get(f"/jobs/{job_id}/video", headers={**self.headers, "Range": "bytes=0-3"})
        self.assertEqual(result.status_code, 206)
        self.assertEqual(result.content, b"fake")
        # Wait for worker's post-completion housekeeping.
        for _ in range(100):
            if list((self.settings.runtime / job_id).iterdir()) == [self.settings.runtime / job_id / "result.mp4"]:
                break
            time.sleep(0.01)
        self.assertEqual([p.name for p in (self.settings.runtime / job_id).iterdir()], ["result.mp4"])
        self.assertEqual(self.client.delete(f"/jobs/{job_id}", headers=self.headers).status_code, 204)
        self.assertEqual(self.client.get(f"/jobs/{job_id}", headers=self.headers).status_code, 404)

    def test_limits_and_empty_uploads(self):
        self.assertEqual(self.client.post("/jobs", content=b"", headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post("/jobs", content=b"x", headers={**self.headers, "Content-Type": "text/plain"}).status_code, 415)
        self.assertEqual(self.client.post("/jobs", content=b"x", headers={**self.headers, "Content-Length": str(self.settings.max_upload_bytes + 1)}).status_code, 413)
        self.assertEqual(self.client.post("/jobs", content=b"x", headers={**self.headers, "Content-Length": "bad"}).status_code, 400)
        self.assertEqual(len(self.client.app.state.manager.jobs), 0)

    def test_download_lease_is_released_on_disconnect(self):
        job_id = self.ready_job()
        job = self.client.app.state.manager.jobs[job_id]
        released = []
        response = LeasedVideoResponse(job.directory / "result.mp4", release=lambda: released.append(True))

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise ConnectionError("Client disconnected")

        with self.assertRaises(ConnectionError):
            asyncio.run(response({"type": "http", "method": "GET", "headers": []}, receive, send))
        self.assertEqual(released, [True])

    def test_upload_lease_prevents_cleanup_while_file_is_open(self):
        manager = self.client.app.state.manager
        job = manager.reserve()
        job.created_at = time.time() - 3601
        with (job.directory / "input.video").open("wb"):
            manager.cleanup()
            self.assertTrue(job.cancel.is_set())
            self.assertTrue(job.directory.exists())
            self.assertIn(job.id, manager.jobs)
        job.status = "failed"
        manager.cleanup()
        self.assertNotIn(job.id, manager.jobs)

    def test_capacity_and_expiry(self):
        first = self.ready_job()
        second = self.ready_job()
        self.assertEqual(self.client.post("/jobs", content=b"x", headers=self.headers).status_code, 429)
        manager = self.client.app.state.manager
        manager.jobs[first].expires_at = time.time() - 1
        self.assertEqual(self.client.get(f"/jobs/{first}", headers=self.headers).status_code, 404)
        manager.cleanup()
        self.assertNotIn(first, manager.jobs)
        self.assertIn(second, manager.jobs)

    def test_missing_and_repeated_delete(self):
        self.assertEqual(self.client.get("/jobs/../../config", headers=self.headers).status_code, 404)
        for _ in range(2):
            self.assertEqual(self.client.delete("/jobs/unknown", headers=self.headers).status_code, 204)

    def test_render_failure_is_reported_and_input_removed(self):
        def fail(job, update):
            raise ValueError("Unreadable video")
        self.client.app.state.manager.renderer = fail
        response = self.client.post("/jobs", content=b"bad", headers=self.headers)
        job_id = response.json()["id"]
        for _ in range(100):
            response = self.client.get(f"/jobs/{job_id}", headers=self.headers)
            if response.json()["status"] == "failed":
                break
            time.sleep(0.01)
        self.assertEqual(response.json()["error"], "Unreadable video")
        self.assertEqual(self.client.get(f"/jobs/{job_id}/video", headers=self.headers).status_code, 409)


if __name__ == "__main__":
    unittest.main()